
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>
#include <type_traits>

// Helper to check CUDA errors
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Operator semantics, RNG and the decoded interpreter are shared with the
// fused PSO kernel so that every evaluator agrees on what a formula means.
#include "eval_core.cuh"

// Stack size for the classic per-sample evaluator. Programs that would exceed
// it are reported as errors instead of silently dropping pushed values.
#define STACK_SIZE RPN_EVAL_STACK

// TEMPLATED KERNEL
template <typename scalar_t>
__global__ void rpn_eval_kernel(
    const unsigned char* __restrict__ population,  // [B, L] (uint8)
    const scalar_t* __restrict__ x,          // [Vars, D]
    const scalar_t* __restrict__ constants,  // [B, K]
    scalar_t* __restrict__ out_preds,        // [B, D]
    int* __restrict__ out_sp,
    unsigned char* __restrict__ out_error,
    int B, int D, int L, int K, int num_vars,
    // ID Mappings passed as scalars
    int PAD_ID, 
    int id_x_start, 
    int id_C, int id_pi, int id_e,
    int id_0, int id_1, int id_2, int id_3, int id_4, int id_5, int id_6, int id_10,
    // Ops
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan,
    // Values
    double pi_val, double e_val,
    // Strict mode: 0 = protected (search), 1 = strict (validation)
    int strict_mode
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)B * D) return;

    long long b_idx = idx / D; // Population Index
    long long d_idx = idx % D; // Sample Index

    // Registers
    scalar_t stack[STACK_SIZE];
    int sp = 0;
    bool error = false;
    int c_idx = 0; // Constants pointer

    const unsigned char* my_prog = &population[b_idx * L];
    // 1e30 fits in both float32 (max ~3.4e38) and float64 — no truncation warning.
    const scalar_t ERROR_VAL = (scalar_t)1e30;

    for (int pc = 0; pc < L; ++pc) {
        int64_t token = (int64_t)my_prog[pc];
        
        
        if (token == PAD_ID) break;

        scalar_t val = (scalar_t)0.0;
        bool is_push = true;

        // Using jump table (switch) for O(1) instruction dispatch mapping
        // This dramatically reduces warp divergence compared to the previous if/else chain
        switch(token) {
            // --- Operands ---
            case 0: // Placeholder, actual match below
            default:
                if (token >= id_x_start && token < id_x_start + num_vars) {
                    int v_idx = token - id_x_start;
                    val = x[v_idx * D + d_idx];
                } else if (token == id_C) {
                    if (K > 0) {
                         int r_idx = c_idx;
                         if (r_idx >= K) r_idx = K - 1;
                         val = constants[b_idx * K + r_idx];
                         c_idx++;
                    } else {
                         val = (scalar_t)1.0;
                    }
                } else if (token == id_0) val = (scalar_t)0.0;
                else if (token == id_1) val = (scalar_t)1.0;
                else if (token == id_2) val = (scalar_t)2.0;
                else if (token == id_3) val = (scalar_t)3.0;
                else if (token == id_4) val = (scalar_t)4.0;
                else if (token == id_5) val = (scalar_t)5.0;
                else if (token == id_6) val = (scalar_t)6.0;
                else if (token == id_10) val = (scalar_t)10.0;
                else if (token == id_pi) val = (scalar_t)pi_val;
                else if (token == id_e) val = (scalar_t)e_val;
                else {
                    // It's an operator
                    is_push = false;
                }
                break;
        }

        if (is_push) {
            // A program deeper than the stack is invalid; dropping the value
            // would silently evaluate a different expression.
            if (sp >= STACK_SIZE) { error = true; break; }
            stack[sp++] = val;
            continue;
        }


        // Binary Operators — most-common first for branch predictor friendliness
        if (__builtin_expect(token == op_add || token == op_sub || token == op_mul || token == op_div || token == op_pow || token == op_mod, 1)) {
            if (__builtin_expect(sp < 2, 0)) { error = true; break; }
            scalar_t op2 = stack[--sp];
            scalar_t op1 = stack[--sp];
            scalar_t res = (scalar_t)0.0;
            
            if (__builtin_expect(token == op_add, 1)) res = op1 + op2;
            else if (__builtin_expect(token == op_sub, 1)) res = op1 - op2;
            else if (__builtin_expect(token == op_mul, 1)) res = op1 * op2;
            else if (token == op_div) res = strict_mode ? strict_div(op1, op2, error) : safe_div(op1, op2, error);
            else if (token == op_pow) res = strict_mode ? strict_pow(op1, op2, error) : safe_pow(op1, op2, error);
            else if (token == op_mod) res = strict_mode ? strict_mod(op1, op2, error) : safe_mod(op1, op2, error);
            
            if (__builtin_expect(error, 0)) break;

            stack[sp++] = res;
            continue;
        }
        
        // Unary operators ordered to reduce warp divergence.
        if (__builtin_expect(sp < 1, 0)) { error = true; break; }
        scalar_t op1 = stack[--sp];
        scalar_t res = (scalar_t)0.0;
        
        // Hot path: lgamma, fact, sqrt, exp, log used most in this problem
        if (__builtin_expect(token == op_lgamma, 1)) res = strict_mode ? strict_lgamma(op1, error) : safe_lgamma(op1, error);
        else if (__builtin_expect(token == op_fact, 1)) res = strict_mode ? strict_tgamma(op1 + (scalar_t)1.0, error) : safe_tgamma(op1 + (scalar_t)1.0, error);
        else if (__builtin_expect(token == op_sqrt, 1)) res = strict_mode ? strict_sqrt(op1, error) : safe_sqrt(op1, error);
        else if (__builtin_expect(token == op_exp, 1)) res = strict_mode ? strict_exp(op1, error) : safe_exp(op1, error);
        else if (__builtin_expect(token == op_log, 1)) res = strict_mode ? strict_log(op1, error) : safe_log(op1, error);
        else if (token == op_sin) res = sin(op1);
        else if (token == op_cos) res = cos(op1);
        else if (token == op_tan) res = tan(op1);
        else if (token == op_abs) res = fabs(op1);
        else if (token == op_neg) res = -op1;
        else if (token == op_floor) res = floor(op1);
        else if (token == op_ceil) res = ceil(op1);
        else if (token == op_sign) res = (op1 > (scalar_t)0.0) ? (scalar_t)1.0 : ((op1 < (scalar_t)0.0) ? (scalar_t)-1.0 : (scalar_t)0.0);
        else if (token == op_asin) res = strict_mode ? strict_asin(op1, error) : safe_asin(op1, error);
        else if (token == op_acos) res = strict_mode ? strict_acos(op1, error) : safe_acos(op1, error);
        else if (token == op_atan) res = atan(op1);
        else if (token == op_gamma) res = strict_mode ? strict_tgamma(op1, error) : safe_tgamma(op1, error);
        else { __builtin_expect(false, 0); error = true; break; }
        
        if (__builtin_expect(error, 0)) break;
        stack[sp++] = res;
    }

    
    out_sp[idx] = sp;
    out_error[idx] = error ? 1 : 0;
    
    if (sp > 0) out_preds[idx] = stack[sp-1];
    else out_preds[idx] = ERROR_VAL;
}

// Wrapper to launch kernel
void launch_rpn_kernel(
    const torch::Tensor& population,
    const torch::Tensor& x,
    const torch::Tensor& constants,
    torch::Tensor& out_preds,
    torch::Tensor& out_sp,
    torch::Tensor& out_error,
    int PAD_ID, 
    int id_x_start, 
    int id_C, int id_pi, int id_e,
    int id_0, int id_1, int id_2, int id_3, int id_4, int id_5, int id_6, int id_10,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan,
    double pi_val, double e_val,
    int strict_mode
) {
    CHECK_INPUT(population);
    CHECK_INPUT(x);
    if (constants.size(1) > 0) CHECK_INPUT(constants);
    
    int B = population.size(0);
    int L = population.size(1);
    
    // X is [Vars, D]
    int num_vars = x.size(0);
    int D = x.size(1);
    
    int K = constants.size(1);
    
    long long total_threads = (long long)B * D;
    if (total_threads == 0) return;
    const int block_size = 256;
    const long long grid_size = (total_threads + block_size - 1) / block_size;
    TORCH_CHECK(grid_size <= 2147483647LL, "rpn_eval_kernel: B*D too large for one launch");
    
    // Dispatch based on X type (float or double)
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "rpn_eval_kernel", ([&] {
        rpn_eval_kernel<scalar_t><<<grid_size, block_size>>>(
            population.data_ptr<unsigned char>(),
            x.data_ptr<scalar_t>(),
            (constants.size(1) > 0) ? constants.data_ptr<scalar_t>() : nullptr,
            out_preds.data_ptr<scalar_t>(),
            out_sp.data_ptr<int32_t>(),
            out_error.data_ptr<uint8_t>(),
            B, D, L, K, num_vars,
            PAD_ID, 
            id_x_start, 
            id_C, id_pi, id_e,
            id_0, id_1, id_2, id_3, id_4, id_5, id_6, id_10,
            op_add, op_sub, op_mul, op_div, op_pow, op_mod,
            op_sin, op_cos, op_tan,
            op_log, op_exp,
            op_sqrt, op_abs, op_neg,
            op_fact, op_floor, op_ceil, op_sign,
            op_gamma, op_lgamma,
            op_asin, op_acos, op_atan,
            pi_val, e_val,
            strict_mode
        );
    }));
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error in launch_rpn_kernel: ", cudaGetErrorString(err));
}

// --- Phase 2: Crossover & Mutation Kernels ---

__global__ void find_subtree_ranges_kernel(
    const unsigned char* __restrict__ population, // [B, L] (uint8)
    const int64_t* __restrict__ row_indices,      // [B] or nullptr
    const int* __restrict__ token_arities,  // [VocabSize]
    int64_t* __restrict__ out_starts,       // [B, L]
    int64_t* __restrict__ out_lengths,      // [B] or nullptr
    int B, int L, int vocab_size, int PAD_ID
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * L;
    if (idx >= total_threads) return;
    
    int b = idx / L; // Batch index
    int tid = idx % L; // Token index in sequence (0..L-1)
    
    int64_t source_row = row_indices != nullptr ? row_indices[b] : b;
    const unsigned char* my_pop = &population[source_row * L];
    int64_t* my_starts = &out_starts[b * L];
    
    int64_t token = (int64_t)my_pop[tid];
    
    // Default invalid
    my_starts[tid] = -1;
    
    if (token == PAD_ID) {
        if (out_lengths != nullptr && (tid == 0 || my_pop[tid - 1] != PAD_ID)) {
            out_lengths[b] = tid;
        }
        return;
    }

    if (out_lengths != nullptr && tid == L - 1) {
        out_lengths[b] = L;
    }
    
    // Get arity
    int arity = 0;
    if (token >= 0 && token < vocab_size) {
        arity = token_arities[token];
    }
    
    // Terminal (arity 0) -> Subtree is just itself
    if (arity == 0) {
        my_starts[tid] = tid;
        return;
    }
    
    // Operator -> Scan backwards to find bounds
    // We need to satisfy 'arity' arguments.
    int needed = arity;
    
    for (int j = tid - 1; j >= 0; --j) {
        int64_t t = (int64_t)my_pop[j];
        if (t == PAD_ID) break; // Invalid structure if we hit PAD
        
        int a = 0;
        if (t >= 0 && t < vocab_size) {
            a = token_arities[t];
        }
        
        // Token j produces 1 output, satisfies 1 need
        needed -= 1;
        // But token j requires 'a' inputs
        needed += a;
        
        if (needed == 0) {
            // Found the start
            my_starts[tid] = j;
            return;
        }
    }
}

void launch_find_subtree_ranges(
    const torch::Tensor& population,
    const torch::Tensor& token_arities,
    torch::Tensor& out_starts,
    int PAD_ID,
    torch::Tensor out_lengths = torch::Tensor()
) {
    CHECK_INPUT(population);
    CHECK_INPUT(token_arities);
    CHECK_INPUT(out_starts);
    if (out_lengths.defined() && out_lengths.numel() > 0) CHECK_INPUT(out_lengths);
    
    int B = population.size(0);
    int L = population.size(1);
    int vocab_size = token_arities.size(0);
    
    int threads = 256;
    int blocks = (B * L + threads - 1) / threads;
    int64_t* out_lengths_ptr = (out_lengths.defined() && out_lengths.numel() > 0) ? out_lengths.data_ptr<int64_t>() : nullptr;
    
    find_subtree_ranges_kernel<<<blocks, threads>>>(
        population.data_ptr<unsigned char>(),
        nullptr,
        token_arities.data_ptr<int32_t>(),
        out_starts.data_ptr<int64_t>(),
        out_lengths_ptr,
        B, L, vocab_size, PAD_ID
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error in find_subtree_ranges: ", cudaGetErrorString(err));
}

__global__ void select_subtree_points_kernel(
    const float* __restrict__ rand_vals,      // [N]
    const int64_t* __restrict__ lengths,      // [N]
    const int64_t* __restrict__ starts,       // [N, L]
    int64_t* __restrict__ out_start,          // [N]
    int64_t* __restrict__ out_end,            // [N]
    int N, int L
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    int64_t len = lengths[n];
    if (len < 1) len = 1;
    if (len > L) len = L;

    float r = rand_vals[n];
    int64_t e = (int64_t)(r * (float)len);
    if (e < 0) e = 0;
    if (e >= L) e = L - 1;
    if (e >= len) e = len - 1;

    int64_t s = starts[n * L + e];
    if (s < 0) s = e;
    out_start[n] = s;
    out_end[n] = e;
}

void launch_select_subtree_points(
    const torch::Tensor& rand_vals,
    const torch::Tensor& lengths,
    const torch::Tensor& starts,
    torch::Tensor& out_start,
    torch::Tensor& out_end
) {
    CHECK_INPUT(rand_vals);
    CHECK_INPUT(lengths);
    CHECK_INPUT(starts);
    CHECK_INPUT(out_start);
    CHECK_INPUT(out_end);

    int N = rand_vals.size(0);
    int L = starts.size(1);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    select_subtree_points_kernel<<<blocks, threads>>>(
        rand_vals.data_ptr<float>(),
        lengths.data_ptr<int64_t>(),
        starts.data_ptr<int64_t>(),
        out_start.data_ptr<int64_t>(),
        out_end.data_ptr<int64_t>(),
        N, L
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error in select_subtree_points: ", cudaGetErrorString(err));
}

__global__ void select_subtree_range_indirect_kernel(
    const unsigned char* __restrict__ population,
    const int64_t* __restrict__ row_indices,
    const int* __restrict__ token_arities,
    const float* __restrict__ rand_vals,
    int64_t* __restrict__ out_lengths,
    int64_t* __restrict__ out_start,
    int64_t* __restrict__ out_end,
    int N, int L, int vocab_size, int PAD_ID
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int64_t source_row = row_indices != nullptr ? row_indices[n] : n;
    const unsigned char* row = population + source_row * (int64_t)L;

    int len = 0;
    while (len < L && row[len] != PAD_ID) ++len;
    if (len < 1) len = 1;
    out_lengths[n] = len;

    int end = (int)(rand_vals[n] * (float)len);
    if (end < 0) end = 0;
    if (end >= len) end = len - 1;
    int start = end;
    int token = (int)row[end];
    int arity = (token >= 0 && token < vocab_size) ? token_arities[token] : 0;
    if (arity > 0) {
        int needed = arity;
        for (int j = end - 1; j >= 0; --j) {
            int t = (int)row[j];
            if (t == PAD_ID) break;
            int a = (t >= 0 && t < vocab_size) ? token_arities[t] : 0;
            needed += a - 1;
            if (needed == 0) {
                start = j;
                break;
            }
        }
    }
    out_start[n] = start;
    out_end[n] = end;
}

__global__ void validate_crossover_lengths_kernel(
    const int64_t* __restrict__ lengths1,
    const int64_t* __restrict__ lengths2,
    const int64_t* __restrict__ starts1,
    const int64_t* __restrict__ ends1,
    const int64_t* __restrict__ starts2,
    const int64_t* __restrict__ ends2,
    bool* __restrict__ valid,
    int N, int max_length
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int64_t new_len1 = starts1[idx] + (ends2[idx] - starts2[idx] + 1)
                     + (lengths1[idx] - (ends1[idx] + 1));
    int64_t new_len2 = starts2[idx] + (ends1[idx] - starts1[idx] + 1)
                     + (lengths2[idx] - (ends2[idx] + 1));
    valid[idx] = (new_len1 <= max_length) && (new_len2 <= max_length);
}

void launch_validate_crossover_lengths(
    const torch::Tensor& lengths1,
    const torch::Tensor& lengths2,
    const torch::Tensor& starts1,
    const torch::Tensor& ends1,
    const torch::Tensor& starts2,
    const torch::Tensor& ends2,
    torch::Tensor& valid,
    int max_length
) {
    int N = lengths1.size(0);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    validate_crossover_lengths_kernel<<<blocks, threads>>>(
        lengths1.data_ptr<int64_t>(), lengths2.data_ptr<int64_t>(),
        starts1.data_ptr<int64_t>(), ends1.data_ptr<int64_t>(),
        starts2.data_ptr<int64_t>(), ends2.data_ptr<int64_t>(),
        valid.data_ptr<bool>(), N, max_length
    );
}

__device__ __forceinline__ uint4 philox4x32_10(uint4 counter, uint2 key) {
    return rpn_philox4x32_10(counter, key);
}

// Point mutation. Constant tokens are never mutated into (or out of) a
// different terminal: constants are bound to their slot by position, so a
// C <-> terminal swap would silently shift every later constant of the
// formula. Constant values are explored by PSO / perturbation instead, and the
// terminal pool supplied for mutation excludes C.
__device__ __forceinline__ unsigned char mutate_token_philox(
    unsigned char token, uint64_t individual, int token_pos,
    const int* __restrict__ token_arities,
    const unsigned char* __restrict__ arity_0_ids, int n_0,
    const unsigned char* __restrict__ arity_1_ids, int n_1,
    const unsigned char* __restrict__ arity_2_ids, int n_2,
    float mutation_rate, int L, int vocab_size, int PAD_ID,
    uint64_t rng_seed, uint64_t generation, int id_C = -1
) {
    if ((int)token == PAD_ID || (int)token == id_C) return token;
    uint4 counter = make_uint4(
        (uint32_t)(individual * (uint64_t)L + (uint64_t)token_pos),
        (uint32_t)generation, (uint32_t)(generation >> 32), (uint32_t)individual);
    uint2 key = make_uint2((uint32_t)rng_seed, (uint32_t)(rng_seed >> 32));
    uint4 random = philox4x32_10(counter, key);
    float probability = ((float)random.x + 0.5f) * 2.3283064365386963e-10f;
    if (probability >= mutation_rate) return token;

    int arity = ((int)token < vocab_size) ? token_arities[(int)token] : 0;
    uint32_t selector = random.y;
    if (arity == 0 && n_0 > 0) {
        unsigned char repl = arity_0_ids[selector % n_0];
        return ((int)repl == id_C) ? token : repl;
    }
    if (arity == 1 && n_1 > 0) return arity_1_ids[selector % n_1];
    if (arity == 2 && n_2 > 0) return arity_2_ids[selector % n_2];
    return token;
}

__global__ void mutation_philox_kernel(
    unsigned char* __restrict__ population,
    const int64_t* __restrict__ individual_ids,
    const int* __restrict__ token_arities,
    const unsigned char* __restrict__ arity_0_ids, int n_0,
    const unsigned char* __restrict__ arity_1_ids, int n_1,
    const unsigned char* __restrict__ arity_2_ids, int n_2,
    float mutation_rate, int B, int L, int vocab_size, int PAD_ID,
    uint64_t rng_seed, uint64_t generation
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * L) return;
    int row = idx / L;
    int token_pos = idx - row * L;
    uint64_t individual = (uint64_t)individual_ids[row];
    population[idx] = mutate_token_philox(
        population[idx], individual, token_pos, token_arities,
        arity_0_ids, n_0, arity_1_ids, n_1, arity_2_ids, n_2,
        mutation_rate, L, vocab_size, PAD_ID, rng_seed, generation);
}

__global__ void mutation_philox_indirect_kernel(
    unsigned char* __restrict__ population,
    const int64_t* __restrict__ row_indices,
    const float* __restrict__ individual_rand,
    float individual_cut,
    const int* __restrict__ token_arities,
    const unsigned char* __restrict__ arity_0_ids, int n_0,
    const unsigned char* __restrict__ arity_1_ids, int n_1,
    const unsigned char* __restrict__ arity_2_ids, int n_2,
    float mutation_rate, int N, int L, int vocab_size, int PAD_ID,
    uint64_t rng_seed, uint64_t generation, int id_C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N * L) return;
    int n = idx / L;
    int token_pos = idx - n * L;
    int64_t row = row_indices[n];
    if (row == 0 || individual_rand[row] >= individual_cut) return;
    int64_t out_idx = row * (int64_t)L + token_pos;
    population[out_idx] = mutate_token_philox(
        population[out_idx], (uint64_t)row, token_pos, token_arities,
        arity_0_ids, n_0, arity_1_ids, n_1, arity_2_ids, n_2,
        mutation_rate, L, vocab_size, PAD_ID, rng_seed, generation, id_C);
}

__global__ void mutation_kernel(
    unsigned char* __restrict__ population,        // [B, L] (uint8)
    const float* __restrict__ rand_floats,   // [B, L] (0..1)
    const int64_t* __restrict__ rand_ints,   // [B, L] (random integers)
    const int* __restrict__ token_arities,   // [VocabSize]
    const unsigned char* __restrict__ arity_0_ids, int n_0,
    const unsigned char* __restrict__ arity_1_ids, int n_1,
    const unsigned char* __restrict__ arity_2_ids, int n_2,
    float mutation_rate,
    int B, int L, int vocab_size, int PAD_ID
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = B * L;
    if (idx >= total_threads) return;
    
    int64_t token = (int64_t)population[idx];
    
    if (token == PAD_ID) return;
    
    // Check mutation probability
    if (rand_floats[idx] >= mutation_rate) return;
    
    // Get arity
    int arity = 0;
    if (token >= 0 && token < vocab_size) {
        arity = token_arities[token];
    }
    
    // Select replacement
    int64_t new_token = token;
    uint64_t rand_val = (uint64_t)rand_ints[idx]; // Use raw bits
    
    if (arity == 0 && n_0 > 0) {
        new_token = (int64_t)arity_0_ids[rand_val % n_0];
    } else if (arity == 1 && n_1 > 0) {
        new_token = (int64_t)arity_1_ids[rand_val % n_1];
    } else if (arity == 2 && n_2 > 0) {
        new_token = (int64_t)arity_2_ids[rand_val % n_2];
    }
    
    population[idx] = (unsigned char)new_token;
}

void launch_mutation_kernel(
    torch::Tensor& population,
    const torch::Tensor& rand_floats,
    const torch::Tensor& rand_ints,
    const torch::Tensor& token_arities,
    const torch::Tensor& arity_0_ids,
    const torch::Tensor& arity_1_ids,
    const torch::Tensor& arity_2_ids,
    float mutation_rate,
    int PAD_ID
) {
    CHECK_INPUT(population);
    CHECK_INPUT(rand_floats);
    CHECK_INPUT(rand_ints);
    
    int B = population.size(0);
    int L = population.size(1);
    int vocab_size = token_arities.size(0);
    
    int threads = 256;
    int blocks = (B * L + threads - 1) / threads;
    
    mutation_kernel<<<blocks, threads>>>(
        population.data_ptr<unsigned char>(),
        rand_floats.data_ptr<float>(),
        rand_ints.data_ptr<int64_t>(),
        token_arities.data_ptr<int32_t>(),
        arity_0_ids.data_ptr<unsigned char>(), arity_0_ids.numel(),
        arity_1_ids.data_ptr<unsigned char>(), arity_1_ids.numel(),
        arity_2_ids.data_ptr<unsigned char>(), arity_2_ids.numel(),
        mutation_rate,
        B, L, vocab_size, PAD_ID
    );
}

__global__ void validate_rpn_batch_kernel(
    const unsigned char* __restrict__ population, // [B, L]
    const int* __restrict__ token_arities,        // [VocabSize]
    bool* __restrict__ out_valid,                 // [B]
    int B, int L, int vocab_size, int PAD_ID
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    
    const unsigned char* row = &population[b * L];
    int stack = 0;
    bool valid = true;
    bool pad_seen = false;
    
    for (int i = 0; i < L; ++i) {
        int64_t t = (int64_t)row[i];
        
        if (t == PAD_ID) {
            pad_seen = true;
            continue;
        }
        
        if (pad_seen) {
            // Found a non-PAD token after a PAD token -> Invalid (not contiguous)
            valid = false;
            break;
        }
        
        int arity = 0;
        if (t >= 0 && t < vocab_size) {
            arity = token_arities[t];
        }
        
        // Stack delta
        int delta = 1 - arity;
        stack += delta;
        
        if (stack < 1) {
            // Underflow
            valid = false;
            break;
        }
    }
    
    // Valid only if final stack is exactly 1
    if (valid && stack != 1) {
        valid = false;
    }
    
    out_valid[b] = valid;
}

void launch_validate_rpn_batch(
    const torch::Tensor& population,
    const torch::Tensor& token_arities,
    torch::Tensor& out_valid,
    int PAD_ID
) {
    CHECK_INPUT(population);
    CHECK_INPUT(token_arities);
    CHECK_INPUT(out_valid);
    
    int B = population.size(0);
    int L = population.size(1);
    int vocab_size = token_arities.size(0);
    
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    
    validate_rpn_batch_kernel<<<blocks, threads>>>(
        population.data_ptr<unsigned char>(),
        token_arities.data_ptr<int32_t>(),
        out_valid.data_ptr<bool>(),
        B, L, vocab_size, PAD_ID
    );
}

__global__ void crossover_splicing_kernel(
    const unsigned char* __restrict__ parent1, // [N, L] (uint8)
    const unsigned char* __restrict__ parent2, // [N, L] (uint8)
    const int64_t* __restrict__ parent1_indices, // [N] or nullptr
    const int64_t* __restrict__ parent2_indices, // [N] or nullptr
    const int64_t* __restrict__ child1_indices,  // [N] or nullptr
    const int64_t* __restrict__ child2_indices,  // [N] or nullptr
    const int64_t* __restrict__ starts1, // [N]
    const int64_t* __restrict__ ends1,   // [N]
    const int64_t* __restrict__ starts2, // [N]
    const int64_t* __restrict__ ends2,   // [N]
    const bool* __restrict__ cx_mask,     // [N] or nullptr
    unsigned char* __restrict__ child1,        // [N, L] (uint8)
    unsigned char* __restrict__ child2,        // [N, L] (uint8)
    int N_pairs, int L, int PAD_ID
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = N_pairs * L;
    if (idx >= total_threads) return;
    
    int n = idx / L; // Pair index
    int t = idx % L; // Token index in child
    int64_t p1_base = (parent1_indices != nullptr ? parent1_indices[n] : n) * (int64_t)L;
    int64_t p2_base = (parent2_indices != nullptr ? parent2_indices[n] : n) * (int64_t)L;
    int64_t c1_base = (child1_indices != nullptr ? child1_indices[n] : n) * (int64_t)L;
    int64_t c2_base = (child2_indices != nullptr ? child2_indices[n] : n) * (int64_t)L;

    if (cx_mask != nullptr && !cx_mask[n]) {
        child1[c1_base + t] = parent1[p1_base + t];
        child2[c2_base + t] = parent2[p2_base + t];
        return;
    }
    
    // --- Child 1 Construction ---
    // Child 1 = P1_Pre + P2_Sub + P1_Post
    int64_t s1 = starts1[n];
    int64_t e1 = ends1[n];
    int64_t s2 = starts2[n];
    int64_t e2 = ends2[n];
    
    int64_t len_pre1 = s1; // [0, s1-1]
    int64_t len_sub2 = e2 - s2 + 1;
    int64_t cut1 = len_pre1 + len_sub2;
    
    int64_t val_c1 = (int64_t)PAD_ID;
    
    if (t < len_pre1) {
        if (t >= 0 && t < L) val_c1 = (int64_t)parent1[p1_base + t];
    } else if (t < cut1) {
        // From Sub2
        int64_t src_idx = s2 + t - len_pre1;
        if (src_idx >= 0 && src_idx < L) {
            val_c1 = (int64_t)parent2[p2_base + src_idx];
        }
    } else {
        // From Post1
        int64_t src_idx = e1 + 1 + t - cut1;
        if (src_idx >= 0 && src_idx < L) {
            val_c1 = (int64_t)parent1[p1_base + src_idx];
        } else {
            val_c1 = (int64_t)PAD_ID;
        }
    }
    child1[c1_base + t] = (unsigned char)val_c1;
    
    // --- Child 2 Construction ---
    // Child 2 = P2_Pre + P1_Sub + P2_Post
    int64_t len_pre2 = s2;
    int64_t len_sub1 = e1 - s1 + 1;
    int64_t cut2 = len_pre2 + len_sub1;
    
    int64_t val_c2 = (int64_t)PAD_ID;
    
    if (t < len_pre2) {
        if (t >= 0 && t < L) val_c2 = (int64_t)parent2[p2_base + t];
    } else if (t < cut2) {
        int64_t src_idx = s1 + t - len_pre2;
        if (src_idx >= 0 && src_idx < L) {
            val_c2 = (int64_t)parent1[p1_base + src_idx];
        }
    } else {
        int64_t src_idx = e2 + 1 + t - cut2;
        if (src_idx >= 0 && src_idx < L) {
            val_c2 = (int64_t)parent2[p2_base + src_idx];
        } else {
            val_c2 = (int64_t)PAD_ID;
        }
    }
    child2[c2_base + t] = (unsigned char)val_c2;
}

// Main-generation fast path: splice and point-mutate while each output token
// is still in a register. This removes the compact/index-copy mutation pass.
__global__ void crossover_splicing_mutation_kernel(
    const unsigned char* __restrict__ parent1,
    const unsigned char* __restrict__ parent2,
    const int64_t* __restrict__ parent1_indices,
    const int64_t* __restrict__ parent2_indices,
    const int64_t* __restrict__ child1_indices,
    const int64_t* __restrict__ child2_indices,
    const int64_t* __restrict__ starts1,
    const int64_t* __restrict__ ends1,
    const int64_t* __restrict__ starts2,
    const int64_t* __restrict__ ends2,
    const bool* __restrict__ cx_mask,
    unsigned char* __restrict__ child1,
    unsigned char* __restrict__ child2,
    const float* __restrict__ individual_rand,
    float individual_cut,
    const int* __restrict__ token_arities,
    const unsigned char* __restrict__ arity_0_ids, int n_0,
    const unsigned char* __restrict__ arity_1_ids, int n_1,
    const unsigned char* __restrict__ arity_2_ids, int n_2,
    float mutation_rate, int vocab_size,
    uint64_t rng_seed, uint64_t generation,
    int N_pairs, int L, int PAD_ID, int id_C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_pairs * L) return;
    int n = idx / L;
    int t = idx - n * L;
    int64_t p1_row = parent1_indices != nullptr ? parent1_indices[n] : n;
    int64_t p2_row = parent2_indices != nullptr ? parent2_indices[n] : n;
    int64_t c1_row = child1_indices != nullptr ? child1_indices[n] : n;
    int64_t c2_row = child2_indices != nullptr ? child2_indices[n] : n;
    int64_t p1_base = p1_row * (int64_t)L;
    int64_t p2_base = p2_row * (int64_t)L;
    int64_t c1_base = c1_row * (int64_t)L;
    int64_t c2_base = c2_row * (int64_t)L;

    unsigned char val_c1;
    unsigned char val_c2;
    if (cx_mask != nullptr && !cx_mask[n]) {
        val_c1 = parent1[p1_base + t];
        val_c2 = parent2[p2_base + t];
    } else {
        int64_t s1 = starts1[n], e1 = ends1[n];
        int64_t s2 = starts2[n], e2 = ends2[n];
        int64_t cut1 = s1 + (e2 - s2 + 1);
        int64_t cut2 = s2 + (e1 - s1 + 1);

        int64_t src1;
        const unsigned char* src_parent1;
        if (t < s1) {
            src1 = t; src_parent1 = parent1 + p1_base;
        } else if (t < cut1) {
            src1 = s2 + t - s1; src_parent1 = parent2 + p2_base;
        } else {
            src1 = e1 + 1 + t - cut1; src_parent1 = parent1 + p1_base;
        }
        val_c1 = (src1 >= 0 && src1 < L) ? src_parent1[src1] : (unsigned char)PAD_ID;

        int64_t src2;
        const unsigned char* src_parent2;
        if (t < s2) {
            src2 = t; src_parent2 = parent2 + p2_base;
        } else if (t < cut2) {
            src2 = s1 + t - s2; src_parent2 = parent1 + p1_base;
        } else {
            src2 = e2 + 1 + t - cut2; src_parent2 = parent2 + p2_base;
        }
        val_c2 = (src2 >= 0 && src2 < L) ? src_parent2[src2] : (unsigned char)PAD_ID;
    }

    if (c1_row != 0 && individual_rand[c1_row] < individual_cut) {
        val_c1 = mutate_token_philox(
            val_c1, (uint64_t)c1_row, t, token_arities,
            arity_0_ids, n_0, arity_1_ids, n_1, arity_2_ids, n_2,
            mutation_rate, L, vocab_size, PAD_ID, rng_seed, generation, id_C);
    }
    if (c2_row != 0 && individual_rand[c2_row] < individual_cut) {
        val_c2 = mutate_token_philox(
            val_c2, (uint64_t)c2_row, t, token_arities,
            arity_0_ids, n_0, arity_1_ids, n_1, arity_2_ids, n_2,
            mutation_rate, L, vocab_size, PAD_ID, rng_seed, generation, id_C);
    }
    child1[c1_base + t] = val_c1;
    child2[c2_base + t] = val_c2;
}

void launch_crossover_splicing(
    const torch::Tensor& parent1,
    const torch::Tensor& parent2,
    const torch::Tensor& starts1,
    const torch::Tensor& ends1,
    const torch::Tensor& starts2,
    const torch::Tensor& ends2,
    torch::Tensor& child1,
    torch::Tensor& child2,
    int PAD_ID,
    const torch::Tensor& cx_mask = torch::Tensor()
) {
    int N = parent1.size(0);
    int L = parent1.size(1);
    const bool* cx_mask_ptr = (cx_mask.defined() && cx_mask.numel() > 0) ? cx_mask.data_ptr<bool>() : nullptr;
    
    int threads = 256;
    int blocks = (N * L + threads - 1) / threads;
    
    crossover_splicing_kernel<<<blocks, threads>>>(
        parent1.data_ptr<unsigned char>(),
        parent2.data_ptr<unsigned char>(),
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        starts1.data_ptr<int64_t>(),
        ends1.data_ptr<int64_t>(),
        starts2.data_ptr<int64_t>(),
        ends2.data_ptr<int64_t>(),
        cx_mask_ptr,
        child1.data_ptr<unsigned char>(),
        child2.data_ptr<unsigned char>(),
        N, L, PAD_ID
    );
}

// Structural mutation without host-side compaction: rows whose graft is
// enabled (mask && fits) receive pre[0,s) + bank_sub + post(e,len); every other
// row is copied unchanged. Writes into a separate buffer (no in-place hazard).
__global__ void graft_splice_masked_kernel(
    const unsigned char* __restrict__ src,      // [B, L]
    const unsigned char* __restrict__ bank,     // [Bank, L]
    const int64_t* __restrict__ bank_rows,      // [B]
    const int64_t* __restrict__ s_pop, const int64_t* __restrict__ e_pop,
    const int64_t* __restrict__ s_bank, const int64_t* __restrict__ e_bank,
    const bool* __restrict__ enabled,           // [B] mask && fits
    unsigned char* __restrict__ dst,            // [B, L]
    int B, int L, int PAD_ID
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)B * L) return;
    int64_t row = idx / L;
    int t = (int)(idx - row * L);
    const unsigned char* r = src + row * (int64_t)L;
    if (!enabled[row]) { dst[idx] = r[t]; return; }
    int64_t s = s_pop[row], e = e_pop[row];
    int64_t sb = s_bank[row], eb = e_bank[row];
    int64_t cut = s + (eb - sb + 1);
    unsigned char v;
    if (t < s) v = r[t];
    else if (t < cut) v = bank[bank_rows[row] * (int64_t)L + sb + (t - s)];
    else {
        int64_t src_i = e + 1 + (t - cut);
        v = (src_i < L) ? r[src_i] : (unsigned char)PAD_ID;
    }
    dst[idx] = v;
}

__global__ void graft_enable_kernel(
    const float* __restrict__ individual_rand, float lo, float hi,
    const int64_t* __restrict__ len_pop,
    const int64_t* __restrict__ s_pop, const int64_t* __restrict__ e_pop,
    const int64_t* __restrict__ s_bank, const int64_t* __restrict__ e_bank,
    bool* __restrict__ enabled, int B, int max_length
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= B) return;
    float u = individual_rand[n];
    bool selected = (n != 0) && u >= lo && u < hi;
    int64_t new_len = s_pop[n] + (e_bank[n] - s_bank[n] + 1) + (len_pop[n] - (e_pop[n] + 1));
    enabled[n] = selected && new_len >= 1 && new_len <= max_length;
}

// Per-row statistics in one pass: program length and number of distinct
// variables used (replaces several full-population PyTorch reductions).
__global__ void population_row_stats_kernel(
    const unsigned char* __restrict__ population, int B, int L, int PAD_ID,
    int id_x_start, int num_vars,
    float* __restrict__ out_len, int32_t* __restrict__ out_var_count
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    const unsigned char* row = population + (int64_t)b * L;
    unsigned long long mask = 0ULL;
    int len = 0;
    for (; len < L; ++len) {
        int t = (int)row[len];
        if (t == PAD_ID) break;
        int v = t - id_x_start;
        if (v >= 0 && v < num_vars && v < 64) mask |= (1ULL << v);
    }
    if (out_len != nullptr) out_len[b] = (float)len;
    if (out_var_count != nullptr) out_var_count[b] = __popcll(mask);
}

void launch_population_row_stats(
    const torch::Tensor& population,
    torch::Tensor& out_len,
    torch::Tensor& out_var_count,
    int PAD_ID, int id_x_start, int num_vars
) {
    CHECK_INPUT(population);
    int B = population.size(0);
    int L = population.size(1);
    float* len_ptr = nullptr;
    int32_t* vc_ptr = nullptr;
    if (out_len.defined() && out_len.numel() > 0) {
        CHECK_INPUT(out_len);
        TORCH_CHECK(out_len.scalar_type() == torch::kFloat32 && out_len.numel() == B, "out_len must be float32 [B]");
        len_ptr = out_len.data_ptr<float>();
    }
    if (out_var_count.defined() && out_var_count.numel() > 0) {
        CHECK_INPUT(out_var_count);
        TORCH_CHECK(out_var_count.scalar_type() == torch::kInt32 && out_var_count.numel() == B, "out_var_count must be int32 [B]");
        vc_ptr = out_var_count.data_ptr<int32_t>();
    }
    if (B == 0) return;
    const int threads = 256;
    population_row_stats_kernel<<<(B + threads - 1) / threads, threads>>>(
        population.data_ptr<unsigned char>(), B, L, PAD_ID, id_x_start, num_vars, len_ptr, vc_ptr);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error in population_row_stats: ", cudaGetErrorString(err));
}

// ---------------------------------------------------------------------------
// Offspring constants
// ---------------------------------------------------------------------------
// Constants are bound to formula positions (the i-th C token reads slot i).
// Crossover children therefore receive the constants that travel with their
// token segments. Copies (no crossover) keep their parent's constants exactly,
// except when both parents share the same structure: then slot j means the
// same thing in both, and SBX blending of the two constant vectors is a
// meaningful recombination.
__global__ void offspring_constants_kernel(
    const unsigned char* __restrict__ pop,      // [B, L] parents
    const float* __restrict__ consts,           // [B, K] parent constants
    const int64_t* __restrict__ p1_rows,
    const int64_t* __restrict__ p2_rows,
    const int64_t* __restrict__ c1_rows,
    const int64_t* __restrict__ c2_rows,
    const int64_t* __restrict__ starts1, const int64_t* __restrict__ ends1,
    const int64_t* __restrict__ starts2, const int64_t* __restrict__ ends2,
    const bool* __restrict__ cx_mask,
    float* __restrict__ out_consts,             // [B, K]
    int N_pairs, int L, int K, int id_C, int PAD_ID,
    float sbx_eta, float sbx_prob,
    uint64_t rng_seed, uint64_t generation
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N_pairs || K <= 0) return;
    const int64_t p1 = p1_rows[n], p2 = p2_rows[n];
    const int64_t c1 = c1_rows[n], c2 = c2_rows[n];
    const unsigned char* r1 = pop + p1 * (int64_t)L;
    const unsigned char* r2 = pop + p2 * (int64_t)L;
    const float* k1 = consts + p1 * (int64_t)K;
    const float* k2 = consts + p2 * (int64_t)K;
    float* o1 = out_consts + c1 * (int64_t)K;
    float* o2 = out_consts + c2 * (int64_t)K;

    if (!cx_mask[n]) {
        bool same = (p1 != p2);
        if (same) {
            for (int i = 0; i < L; ++i) {
                if (r1[i] != r2[i]) { same = false; break; }
                if ((int)r1[i] == PAD_ID) break;
            }
        }
        for (int j = 0; j < K; ++j) {
            float a = k1[j], b = k2[j];
            float ya = a, yb = b;
            if (same && sbx_prob > 0.0f) {
                uint4 r = rpn_random4(rng_seed ^ 0x5B5C0DE5ULL, generation, (uint64_t)n, (uint64_t)j);
                if (rpn_u01(r.x) < sbx_prob) {
                    float u = rpn_u01(r.y);
                    float inv = 1.0f / (sbx_eta + 1.0f);
                    float beta = (u <= 0.5f) ? powf(2.0f * u, inv)
                                             : powf(1.0f / (2.0f * fmaxf(1.0f - u, 1e-7f)), inv);
                    ya = 0.5f * ((1.0f + beta) * a + (1.0f - beta) * b);
                    yb = 0.5f * ((1.0f - beta) * a + (1.0f + beta) * b);
                }
            }
            o1[j] = ya;
            o2[j] = yb;
        }
        return;
    }

    const int64_t s1 = starts1[n], e1 = ends1[n];
    const int64_t s2 = starts2[n], e2 = ends2[n];
    float buf1[32];
    float buf2[32];
    const int KK = K < 32 ? K : 32;
    // Child 1 = P1[0, s1) + P2[s2, e2] + P1(e1, L)
    {
        int out = 0, pc = 0;
        for (int64_t i = 0; i < s1; ++i) if ((int)r1[i] == id_C) { if (out < KK) buf1[out++] = k1[pc < K ? pc : K - 1]; ++pc; }
        int qc = 0;
        for (int64_t i = 0; i < s2; ++i) if ((int)r2[i] == id_C) ++qc;
        for (int64_t i = s2; i <= e2; ++i) if ((int)r2[i] == id_C) { if (out < KK) buf1[out++] = k2[qc < K ? qc : K - 1]; ++qc; }
        for (int64_t i = s1; i <= e1; ++i) if ((int)r1[i] == id_C) ++pc;
        for (int64_t i = e1 + 1; i < L; ++i) {
            if ((int)r1[i] == PAD_ID) break;
            if ((int)r1[i] == id_C) { if (out < KK) buf1[out++] = k1[pc < K ? pc : K - 1]; ++pc; }
        }
        for (; out < KK; ++out) buf1[out] = k1[out];
    }
    // Child 2 = P2[0, s2) + P1[s1, e1] + P2(e2, L)
    {
        int out = 0, qc = 0;
        for (int64_t i = 0; i < s2; ++i) if ((int)r2[i] == id_C) { if (out < KK) buf2[out++] = k2[qc < K ? qc : K - 1]; ++qc; }
        int pc = 0;
        for (int64_t i = 0; i < s1; ++i) if ((int)r1[i] == id_C) ++pc;
        for (int64_t i = s1; i <= e1; ++i) if ((int)r1[i] == id_C) { if (out < KK) buf2[out++] = k1[pc < K ? pc : K - 1]; ++pc; }
        for (int64_t i = s2; i <= e2; ++i) if ((int)r2[i] == id_C) ++qc;
        for (int64_t i = e2 + 1; i < L; ++i) {
            if ((int)r2[i] == PAD_ID) break;
            if ((int)r2[i] == id_C) { if (out < KK) buf2[out++] = k2[qc < K ? qc : K - 1]; ++qc; }
        }
        for (; out < KK; ++out) buf2[out] = k2[out];
    }
    for (int j = 0; j < KK; ++j) { o1[j] = buf1[j]; o2[j] = buf2[j]; }
    for (int j = KK; j < K; ++j) { o1[j] = k1[j]; o2[j] = k2[j]; }
}

// Structural (bank) mutation constants: the grafted subtree's C tokens get
// fresh random values, and the constants of the suffix keep their own values
// even though their slot indices shift. Must run on the pre-graft tokens.
__global__ void graft_constants_kernel(
    const unsigned char* __restrict__ pop,      // [B, L] pre-graft offspring
    const unsigned char* __restrict__ bank,     // [Bank, L]
    const int64_t* __restrict__ rows,           // [N] offspring rows
    const int64_t* __restrict__ bank_rows,      // [N]
    const int64_t* __restrict__ s_pop, const int64_t* __restrict__ e_pop,
    const int64_t* __restrict__ s_bank, const int64_t* __restrict__ e_bank,
    const bool* __restrict__ valid,             // [N]
    float* __restrict__ consts,                 // [B, K] in-place
    int N, int L, int K, int id_C, int PAD_ID,
    float c_lo, float c_hi, uint64_t rng_seed, uint64_t generation
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || K <= 0 || !valid[n]) return;
    const int64_t row = rows != nullptr ? rows[n] : (int64_t)n;
    const unsigned char* r = pop + row * (int64_t)L;
    const unsigned char* g = bank + bank_rows[n] * (int64_t)L;
    float* k = consts + row * (int64_t)K;
    const int KK = K < 32 ? K : 32;
    float old[32];
    for (int j = 0; j < KK; ++j) old[j] = k[j];
    float buf[32];
    int out = 0, pc = 0;
    const int64_t s = s_pop[n], e = e_pop[n];
    for (int64_t i = 0; i < s; ++i) if ((int)r[i] == id_C) { if (out < KK) buf[out++] = old[pc < KK ? pc : KK - 1]; ++pc; }
    int fresh = 0;
    for (int64_t i = s_bank[n]; i <= e_bank[n]; ++i) {
        if ((int)g[i] == id_C) {
            uint4 rnd = rpn_random4(rng_seed ^ 0x6A09E667ULL, generation, (uint64_t)row, (uint64_t)fresh++);
            if (out < KK) buf[out++] = c_lo + (c_hi - c_lo) * rpn_u01(rnd.x);
        }
    }
    for (int64_t i = s; i <= e; ++i) if ((int)r[i] == id_C) ++pc;
    for (int64_t i = e + 1; i < L; ++i) {
        if ((int)r[i] == PAD_ID) break;
        if ((int)r[i] == id_C) { if (out < KK) buf[out++] = old[pc < KK ? pc : KK - 1]; ++pc; }
    }
    for (; out < KK; ++out) buf[out] = old[out];
    for (int j = 0; j < KK; ++j) k[j] = buf[j];
}

// Hoist mutation applied directly to the rows selected by the per-individual
// random draw (no host-side compaction). The chosen subtree end is uniform
// over the program, and constants are shifted so that the hoisted subtree's
// C tokens keep reading the values they used before.
__global__ void hoist_mutation_masked_kernel(
    unsigned char* __restrict__ population,     // [B, L]
    float* __restrict__ consts,                 // [B, K]
    const float* __restrict__ individual_rand,  // [B]
    float lo, float hi,
    const int* __restrict__ token_arities,
    int B, int L, int K, int vocab_size, int PAD_ID, int id_C,
    uint64_t rng_seed, uint64_t generation
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B || b == 0) return;
    float u = individual_rand[b];
    if (!(u >= lo && u < hi)) return;
    unsigned char* row = population + (int64_t)b * L;
    int len = 0;
    while (len < L && (int)row[len] != PAD_ID) ++len;
    if (len < 2) return;
    uint4 rnd = rpn_random4(rng_seed ^ 0x3C6EF372ULL, generation, (uint64_t)b, 0ULL);
    int end = (int)(rnd.x % (uint32_t)len);
    int token = (int)row[end];
    int arity = (token >= 0 && token < vocab_size) ? token_arities[token] : 0;
    int start = end;
    if (arity > 0) {
        int needed = arity;
        start = -1;
        for (int j = end - 1; j >= 0; --j) {
            int t = (int)row[j];
            int a = (t >= 0 && t < vocab_size) ? token_arities[t] : 0;
            needed += a - 1;
            if (needed == 0) { start = j; break; }
        }
        if (start < 0) return;
    }
    if (start == 0 && end == len - 1) return;
    int c_before = 0;
    for (int i = 0; i < start; ++i) if ((int)row[i] == id_C) ++c_before;
    int sub_len = end - start + 1;
    for (int i = 0; i < sub_len; ++i) row[i] = row[start + i];
    for (int i = sub_len; i < L; ++i) row[i] = (unsigned char)PAD_ID;
    if (K > 0 && c_before > 0) {
        float* k = consts + (int64_t)b * K;
        for (int j = 0; j + c_before < K; ++j) k[j] = k[j + c_before];
    }
}

// --- Constant Perturbation: in-place local search for numeric constants ---

__device__ __forceinline__ uint32_t perturb_hash_u32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__ float perturb_uniform01(uint32_t x) {
    return ((perturb_hash_u32(x) >> 8) + 1.0f) * (1.0f / 16777217.0f);
}

template <typename scalar_t>
__global__ void constant_perturbation_kernel(
    scalar_t* __restrict__ constants,
    int B,
    int K,
    float rate,
    float sigma,
    float c_min,
    float c_max,
    uint32_t seed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * K;
    if (idx >= total) return;

    int row = idx / K;
    int col = idx - row * K;
    uint32_t row_key = seed ^ ((uint32_t)(row + 1) * 0x9e3779b9u);
    if (perturb_uniform01(row_key) >= rate) return;

    uint32_t nkey1 = row_key ^ ((uint32_t)(col + 1) * 0x85ebca6bu);
    uint32_t nkey2 = row_key ^ ((uint32_t)(col + 1) * 0xc2b2ae35u);
    float u1 = fmaxf(perturb_uniform01(nkey1), 1e-7f);
    float u2 = perturb_uniform01(nkey2);
    float z = sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);

    float value = (float)constants[idx];
    float scale = fabsf(value) * sigma + 1e-4f;
    float next = value + z * scale;
    // Keep the search range, but never pull an existing out-of-range constant
    // (e.g. from a seed formula) back to the boundary: that would destroy it.
    next = fminf(fmaxf(next, fminf(c_min, value)), fmaxf(c_max, value));
    constants[idx] = (scalar_t)next;
}

void launch_constant_perturbation(
    torch::Tensor& constants,
    float rate,
    float sigma,
    float c_min,
    float c_max,
    uint32_t seed
) {
    CHECK_INPUT(constants);
    if (constants.numel() == 0 || rate <= 0.0f || sigma <= 0.0f) return;

    int B = constants.size(0);
    int K = constants.size(1);
    int total = B * K;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(constants.scalar_type(), "constant_perturbation_cuda", [&] {
        constant_perturbation_kernel<scalar_t><<<blocks, threads>>>(
            constants.data_ptr<scalar_t>(),
            B, K,
            rate, sigma,
            c_min, c_max,
            seed
        );
    });

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error in constant_perturbation: ", cudaGetErrorString(err));
}

// --- Phase 3: Tournament Selection ---

__global__ void tournament_selection_kernel(
    const float* __restrict__ fitness,      // [PopSize]
    const float* __restrict__ errors,       // [PopSize, N_data] or nullptr
    const int64_t* __restrict__ rand_idx,   // [PopSize, TourSize]
    const int* __restrict__ rand_cases,     // [PopSize] or nullptr
    int64_t* __restrict__ selected_idx,     // [PopSize]
    const float* __restrict__ lengths,      // [PopSize] or nullptr (NEW)
    const float* __restrict__ mad_eps,      // [N_data] or nullptr (NEW Phase 3)
    int pop_size, int tour_size, int n_data
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;
    
    const int64_t* my_candidates = &rand_idx[idx * tour_size];
    int case_idx = (rand_cases != nullptr) ? rand_cases[idx] : -1;
    
    int64_t best_idx = my_candidates[0];
    float best_val;
    float best_len = 1000000.0f;
    
    if (case_idx >= 0 && errors != nullptr) {
        best_val = errors[best_idx * n_data + case_idx];
    } else {
        best_val = fitness[best_idx];
    }
    if (lengths != nullptr) {
        best_len = lengths[best_idx];
    }
    
    for (int k = 1; k < tour_size; ++k) {
        int64_t candidate = my_candidates[k];
        float val;
        float len = 1000000.0f;
        
        if (case_idx >= 0 && errors != nullptr) {
            val = errors[candidate * n_data + case_idx];
        } else {
            val = fitness[candidate];
        }
        if (lengths != nullptr) {
            len = lengths[candidate];
        }
        
        bool improve = false;
        if (val < best_val) {
            improve = true;
        } else if (lengths != nullptr) {
            // PHASE 8: Add epsilon for Lexicase parsimony
            // If errors are extremely close, pick the shorter one.
            // Using a moderate epsilon (1e-3f) for Lexicase to naturally reject micro-optimizations that double the tree size,
            // while still allowing genuine small incremental improvements (0.1% error drops).
            // using mad_eps if available
            float epsilon = (case_idx >= 0 && mad_eps != nullptr) ? mad_eps[case_idx] : ((case_idx >= 0) ? 1e-3f : 1e-9f);
            if (fabsf(val - best_val) < epsilon && len < best_len) {
                improve = true;
            }
        }
        
        if (improve) {
            best_val = val;
            best_idx = candidate;
            best_len = len;
        }
    }
    
    selected_idx[idx] = best_idx;
}

__global__ void tournament_selection_offsets_kernel(
    const float* __restrict__ fitness,      // [PopSize]
    const float* __restrict__ errors,       // [PopSize, N_data] or nullptr
    const int* __restrict__ rand_offsets,   // [PopSize, TourSize], local island offsets or global indices
    const int* __restrict__ rand_cases,     // [PopSize] or nullptr
    int64_t* __restrict__ selected_idx,     // [PopSize]
    const float* __restrict__ lengths,      // [PopSize] or nullptr
    const float* __restrict__ mad_eps,      // [N_data] or nullptr
    int pop_size, int tour_size, int n_data,
    int island_size, int n_islands
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    int base = 0;
    int span = pop_size;
    if (n_islands > 1) {
        span = max(1, island_size);
        base = (idx / span) * span;
        span = min(span, pop_size - base);
        span = max(1, span);
    }

    const int* my_offsets = &rand_offsets[idx * tour_size];
    int case_idx = (rand_cases != nullptr) ? rand_cases[idx] : -1;

    int first_offset = my_offsets[0];
    int64_t best_idx = (int64_t)(base + (first_offset % span));
    if (best_idx < 0) best_idx = 0;
    if (best_idx >= pop_size) best_idx = pop_size - 1;

    float best_val = (case_idx >= 0 && errors != nullptr)
        ? errors[best_idx * n_data + case_idx]
        : fitness[best_idx];
    float best_len = (lengths != nullptr) ? lengths[best_idx] : 1000000.0f;

    for (int k = 1; k < tour_size; ++k) {
        int offset = my_offsets[k];
        int64_t candidate = (int64_t)(base + (offset % span));
        if (candidate < 0) candidate = 0;
        if (candidate >= pop_size) candidate = pop_size - 1;

        float val = (case_idx >= 0 && errors != nullptr)
            ? errors[candidate * n_data + case_idx]
            : fitness[candidate];
        float len = (lengths != nullptr) ? lengths[candidate] : 1000000.0f;

        bool improve = false;
        if (val < best_val) {
            improve = true;
        } else if (lengths != nullptr) {
            float epsilon = (case_idx >= 0 && mad_eps != nullptr) ? mad_eps[case_idx] : ((case_idx >= 0) ? 1e-3f : 1e-9f);
            if (fabsf(val - best_val) < epsilon && len < best_len) {
                improve = true;
            }
        }

        if (improve) {
            best_val = val;
            best_idx = candidate;
            best_len = len;
        }
    }

    selected_idx[idx] = best_idx;
}

void launch_tournament_selection(
    const torch::Tensor& fitness,
    const torch::Tensor& errors,
    const torch::Tensor& rand_idx,
    const torch::Tensor& rand_cases,
    torch::Tensor& selected_idx,
    const torch::Tensor& lengths,
    const torch::Tensor& mad_eps
) {
    // fitness: [B]
    // rand_idx: [B, K]
    // selected_idx: [B]
    
    CHECK_INPUT(fitness);
    CHECK_INPUT(rand_idx);
    CHECK_INPUT(selected_idx);
    if (errors.numel() > 0) CHECK_INPUT(errors);
    if (rand_cases.numel() > 0) CHECK_INPUT(rand_cases);
    
    int B = fitness.size(0);
    int K = rand_idx.size(1);
    int n_data = (errors.numel() > 0) ? errors.size(1) : 0;
    
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    
    const float* errors_ptr = (errors.numel() > 0) ? errors.data_ptr<float>() : nullptr;
    const int* cases_ptr = (rand_cases.numel() > 0) ? rand_cases.data_ptr<int>() : nullptr;

    const float* lengths_ptr = (lengths.numel() > 0) ? lengths.data_ptr<float>() : nullptr;
    const float* mad_eps_ptr = (mad_eps.numel() > 0) ? mad_eps.data_ptr<float>() : nullptr;

    tournament_selection_kernel<<<blocks, threads>>>(
        fitness.data_ptr<float>(),
        errors_ptr,
        rand_idx.data_ptr<int64_t>(),
        cases_ptr,
        selected_idx.data_ptr<int64_t>(),
        lengths_ptr,
        mad_eps_ptr,
        B, K, n_data
    );
     
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error in tournament_selection: ", cudaGetErrorString(err));
}

void launch_tournament_selection_offsets(
    const torch::Tensor& fitness,
    const torch::Tensor& errors,
    const torch::Tensor& rand_offsets,
    const torch::Tensor& rand_cases,
    torch::Tensor& selected_idx,
    const torch::Tensor& lengths,
    const torch::Tensor& mad_eps,
    int island_size,
    int n_islands
) {
    CHECK_INPUT(fitness);
    CHECK_INPUT(rand_offsets);
    CHECK_INPUT(selected_idx);
    if (errors.numel() > 0) CHECK_INPUT(errors);
    if (rand_cases.numel() > 0) CHECK_INPUT(rand_cases);

    int B = fitness.size(0);
    int K = rand_offsets.size(1);
    int n_data = (errors.numel() > 0) ? errors.size(1) : 0;

    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    const float* errors_ptr = (errors.numel() > 0) ? errors.data_ptr<float>() : nullptr;
    const int* cases_ptr = (rand_cases.numel() > 0) ? rand_cases.data_ptr<int>() : nullptr;
    const float* lengths_ptr = (lengths.numel() > 0) ? lengths.data_ptr<float>() : nullptr;
    const float* mad_eps_ptr = (mad_eps.numel() > 0) ? mad_eps.data_ptr<float>() : nullptr;

    tournament_selection_offsets_kernel<<<blocks, threads>>>(
        fitness.data_ptr<float>(),
        errors_ptr,
        rand_offsets.data_ptr<int>(),
        cases_ptr,
        selected_idx.data_ptr<int64_t>(),
        lengths_ptr,
        mad_eps_ptr,
        B, K, n_data,
        island_size, n_islands
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error in tournament_selection_offsets: ", cudaGetErrorString(err));
}

// --- Phase 4: C++ Orchestrator (evolve_generation) ---

// External PSO Launchers (from pso_kernels.cu)
void launch_pso_update(
    torch::Tensor& pos,
    torch::Tensor& vel,
    const torch::Tensor& pbest,
    const torch::Tensor& gbest,
    const torch::Tensor& r1,
    const torch::Tensor& r2,
    float w, float c1, float c2
);

void launch_pso_update_bests(
    const torch::Tensor& current_err,
    torch::Tensor& pbest_err,
    torch::Tensor& pbest_pos,
    const torch::Tensor& current_pos,
    torch::Tensor& gbest_err,
    torch::Tensor& gbest_pos
);

// RPN Eval (Forward Decl) - We use launch_rpn_kernel directly now
void launch_rpn_kernel(
    const torch::Tensor& population,
    const torch::Tensor& x,
    const torch::Tensor& constants,
    torch::Tensor& out_preds,
    torch::Tensor& out_sp,
    torch::Tensor& out_error,
    int PAD_ID, 
    int id_x_start, 
    int id_C, int id_pi, int id_e,
    int id_0, int id_1, int id_2, int id_3, int id_4, int id_5, int id_6, int id_10,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan,
    double pi_val, double e_val,
    int strict_mode
);

// NOTE: launch_find_subtree_ranges and launch_crossover_splicing are defined above


std::vector<torch::Tensor> evolve_generation(
    torch::Tensor population,      // [B, L]
    torch::Tensor constants,       // [B, K]
    torch::Tensor fitness,         // [B]
    torch::Tensor abs_errors,     // [B, N_data] or Empty
    torch::Tensor X,               // [Vars, N_data] (Transposed for RPN kernel)
    torch::Tensor Y_target,        // [N_data]
    torch::Tensor lengths,         // [B] float32 (for parsimony)
    torch::Tensor token_arities,   // [VocabSize] int32
    torch::Tensor arity_0_ids,     // [n0] int64
    torch::Tensor arity_1_ids,     // [n1] int64
    torch::Tensor arity_2_ids,     // [n2] int64
    torch::Tensor mutation_bank,   // [BankSize, L] or Empty
    torch::Tensor mad_eps,         // [N_data] float32 Phase 3 MAD epsilons
    float mutation_rate,
    float crossover_rate,
    int tournament_size,
    int pso_steps,
    int pso_particles,
    float pso_w, float pso_c1, float pso_c2,
    int PAD_ID,
    // OpCodes
    int id_x_start, 
    int id_C, int id_pi, int id_e,
    int id_0, int id_1, int id_2, int id_3, int id_4, int id_5, int id_6, int id_10,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan,
    double pi_val, double e_val,
    int n_islands,
    torch::Tensor cached_p1_src,
    torch::Tensor cached_p2_src,
    torch::Tensor cached_copy_src,
    torch::Tensor cached_island_base,
    uint64_t rng_seed,
    uint64_t generation,
    float sbx_eta,
    float sbx_prob,
    float graft_const_lo,
    float graft_const_hi
) {
    // Full Orchestrator: Selection + Crossover + Mutation + PSO
    
    int B = population.size(0);
    int L = population.size(1);
    int K = constants.size(1);
    int N_data = X.size(1);
    auto device = population.device();
    auto float_opt = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto long_opt = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto int_opt = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto byte_opt = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    
    // 1. Selection (Tournament / Lexicase-Approx)
    // --- ISLAND MODEL RESTORATION ---
    // Instead of global random indices, we generate indices relative to each island.
    // island_size = B / n_islands
    // ranges = [0, 0, 0...], [100, 100, 100...]
    // rand = ranges + randint(0, island_size)
    
    int selection_island_size = B;
    if (n_islands > 1) {
        int island_size = B / n_islands;
        if (island_size < 1) island_size = 1; // Safety
        selection_island_size = island_size;
    }
    torch::Tensor rand_offsets;
    if (n_islands > 1) {
        rand_offsets = torch::randint(0, selection_island_size, {B, tournament_size}, int_opt);
    } else {
        // Panmictic (Global)
        rand_offsets = torch::randint(0, B, {B, tournament_size}, int_opt);
    }
    
    auto winner_idx = torch::empty({B}, long_opt);
    
    // Lexicase Approximation: If abs_errors provided, each tournament picks a random test case.
    // The case index addresses the columns of abs_errors. Those columns may be a
    // random subsample of the dataset (lexicase sub-sampling), so the range
    // must come from abs_errors itself and never from X.
    torch::Tensor rand_cases;
    if (abs_errors.numel() > 0) {
        TORCH_CHECK(abs_errors.dim() == 2 && abs_errors.size(0) == B,
                    "abs_errors must have shape [B, cases]");
        const int n_cases = (int)abs_errors.size(1);
        TORCH_CHECK(!(mad_eps.defined() && mad_eps.numel() > 0) || mad_eps.numel() == n_cases,
                    "mad_eps must have one entry per abs_errors column");
        rand_cases = torch::randint(0, n_cases, {B}, int_opt);
    } else {
        rand_cases = torch::empty({0}, int_opt);
    }
    
    auto fit_f32 = (fitness.scalar_type() == torch::kFloat32) ? fitness : fitness.to(torch::kFloat32);
    auto err_f32 = (abs_errors.numel() > 0) ? 
        ((abs_errors.scalar_type() == torch::kFloat32) ? abs_errors : abs_errors.to(torch::kFloat32)) : 
        torch::empty({0}, float_opt);
    
    auto lengths_f32 = (lengths.numel() > 0 && lengths.scalar_type() != torch::kFloat32) ? lengths.to(torch::kFloat32) : lengths;
    auto mad_eps_f32 = (mad_eps.defined() && mad_eps.numel() > 0 && mad_eps.scalar_type() != torch::kFloat32) ? mad_eps.to(torch::kFloat32) : mad_eps;
    launch_tournament_selection_offsets(
        fit_f32, err_f32, rand_offsets, rand_cases, winner_idx, lengths_f32, mad_eps_f32,
        selection_island_size, n_islands
    );
    
    // --- Elitism: Preserve the best individual at index 0 ---
    auto best_idx = torch::argmin(fit_f32);
    winner_idx.index_put_({0}, best_idx);
    

    // 2. Crossover (Proper Subtree Crossover with SAFETY CHECKS)
    // BUG-ELT-1 Fix: Skip the elite (pos 0) from crossover to prevent structural contamination.
    // The elite will be preserved in next_pop[0].
    // To ensure strict island isolation, we group parent index pairs strictly within their islands.
    int island_size = B / n_islands;
    torch::Tensor p1_src_t, p2_src_t, copy_src_t;
    if (cached_p1_src.defined() && cached_p2_src.defined() && cached_copy_src.defined() &&
        cached_p1_src.numel() == cached_p2_src.numel() && cached_copy_src.numel() > 0) {
        p1_src_t = cached_p1_src;
        p2_src_t = cached_p2_src;
        copy_src_t = cached_copy_src;
    } else {
        std::vector<torch::Tensor> p1_parts;
        std::vector<torch::Tensor> p2_parts;
        std::vector<torch::Tensor> copy_parts;

        copy_parts.push_back(torch::zeros({1}, long_opt));

        for (int k = 0; k < n_islands; ++k) {
            int start = k * island_size;
            int end = (k + 1) * island_size;
            
            int avail_start = (k == 0) ? start + 1 : start;
            int avail_len = end - avail_start;
            int p_k = avail_len / 2;

            if (p_k > 0) {
                auto p1_part = torch::arange(avail_start, avail_start + 2 * p_k, 2, long_opt);
                p1_parts.push_back(p1_part);
                p2_parts.push_back(p1_part + 1);
            }

            if (avail_len % 2 != 0) {
                copy_parts.push_back(torch::full({1}, end - 1, long_opt));
            }
        }

        p1_src_t = p1_parts.empty() ? torch::empty({0}, long_opt) : torch::cat(p1_parts).contiguous();
        p2_src_t = p2_parts.empty() ? torch::empty({0}, long_opt) : torch::cat(p2_parts).contiguous();
        copy_src_t = torch::cat(copy_parts).contiguous();
    }
    auto c1_dest_t = p1_src_t;
    auto c2_dest_t = p2_src_t;
    auto copy_dest_t = copy_src_t;
    
    int n_pairs = p1_src_t.size(0);
    
    auto p1_winner_idx = winner_idx.index_select(0, p1_src_t);
    auto p2_winner_idx = winner_idx.index_select(0, p2_src_t);
    auto copy_winner_idx = winner_idx.index_select(0, copy_src_t);
    
    // Keep crossover selection as a device mask. Compacting with nonzero followed
    // by index_select/index_copy added allocator traffic and synchronization to
    // every generation; the splice kernels can cheaply copy masked-out parents.
    auto cx_prob = torch::rand({n_pairs}, float_opt);
    auto cx_candidate_mask = (cx_prob < crossover_rate);
    auto lengths1 = torch::empty({n_pairs}, long_opt);
    auto lengths2 = torch::empty({n_pairs}, long_opt);
    auto rand_e1 = torch::rand({n_pairs}, float_opt);
    auto rand_e2 = torch::rand({n_pairs}, float_opt);
    auto s1 = torch::empty({n_pairs}, long_opt);
    auto e1 = torch::empty({n_pairs}, long_opt);
    auto s2 = torch::empty({n_pairs}, long_opt);
    auto e2 = torch::empty({n_pairs}, long_opt);
    int threads_ranges = 256;
    int blocks_ranges = (n_pairs + threads_ranges - 1) / threads_ranges;
    select_subtree_range_indirect_kernel<<<blocks_ranges, threads_ranges>>>(
        population.data_ptr<unsigned char>(), p1_winner_idx.data_ptr<int64_t>(),
        token_arities.data_ptr<int32_t>(), rand_e1.data_ptr<float>(),
        lengths1.data_ptr<int64_t>(), s1.data_ptr<int64_t>(), e1.data_ptr<int64_t>(),
        n_pairs, L, token_arities.size(0), PAD_ID);
    select_subtree_range_indirect_kernel<<<blocks_ranges, threads_ranges>>>(
        population.data_ptr<unsigned char>(), p2_winner_idx.data_ptr<int64_t>(),
        token_arities.data_ptr<int32_t>(), rand_e2.data_ptr<float>(),
        lengths2.data_ptr<int64_t>(), s2.data_ptr<int64_t>(), e2.data_ptr<int64_t>(),
        n_pairs, L, token_arities.size(0), PAD_ID);

    auto cx_mask_flat = torch::empty({n_pairs}, torch::TensorOptions().dtype(torch::kBool).device(device));
    launch_validate_crossover_lengths(lengths1, lengths2, s1, e1, s2, e2, cx_mask_flat, L);
    cx_mask_flat.logical_and_(cx_candidate_mask);

    auto mut_rand = torch::rand({B}, float_opt);
    bool has_bank = (mutation_bank.numel() > 0);
    float point_cut = has_bank ? 0.5f : 0.8f;

    auto offspring = torch::empty_like(population);
    int threads_splice = 256;
    if (n_pairs > 0) {
        int blocks_splice = (int)(((int64_t)n_pairs * L + threads_splice - 1) / threads_splice);
        crossover_splicing_mutation_kernel<<<blocks_splice, threads_splice>>>(
            population.data_ptr<unsigned char>(), population.data_ptr<unsigned char>(),
            p1_winner_idx.data_ptr<int64_t>(), p2_winner_idx.data_ptr<int64_t>(),
            c1_dest_t.data_ptr<int64_t>(), c2_dest_t.data_ptr<int64_t>(),
            s1.data_ptr<int64_t>(), e1.data_ptr<int64_t>(),
            s2.data_ptr<int64_t>(), e2.data_ptr<int64_t>(),
            cx_mask_flat.data_ptr<bool>(), offspring.data_ptr<unsigned char>(),
            offspring.data_ptr<unsigned char>(), mut_rand.data_ptr<float>(), point_cut,
            token_arities.data_ptr<int32_t>(),
            arity_0_ids.data_ptr<unsigned char>(), arity_0_ids.numel(),
            arity_1_ids.data_ptr<unsigned char>(), arity_1_ids.numel(),
            arity_2_ids.data_ptr<unsigned char>(), arity_2_ids.numel(),
            mutation_rate, token_arities.size(0), rng_seed, generation,
            n_pairs, L, PAD_ID, id_C);
    }

    offspring.index_copy_(0, copy_dest_t, population.index_select(0, copy_winner_idx));
    int copy_mut_total = copy_dest_t.size(0) * L;
    int copy_mut_blocks = (copy_mut_total + threads_splice - 1) / threads_splice;
    mutation_philox_indirect_kernel<<<copy_mut_blocks, threads_splice>>>(
        offspring.data_ptr<unsigned char>(), copy_dest_t.data_ptr<int64_t>(),
        mut_rand.data_ptr<float>(), point_cut, token_arities.data_ptr<int32_t>(),
        arity_0_ids.data_ptr<unsigned char>(), arity_0_ids.numel(),
        arity_1_ids.data_ptr<unsigned char>(), arity_1_ids.numel(),
        arity_2_ids.data_ptr<unsigned char>(), arity_2_ids.numel(),
        mutation_rate, copy_dest_t.size(0), L, token_arities.size(0), PAD_ID,
        rng_seed, generation, id_C);

    // Offspring constants follow their token segments (see offspring_constants_kernel).
    auto offspring_consts = torch::empty_like(constants);
    if (n_pairs > 0 && K > 0) {
        int threads_consts = 128;
        int blocks_consts = (n_pairs + threads_consts - 1) / threads_consts;
        offspring_constants_kernel<<<blocks_consts, threads_consts>>>(
            population.data_ptr<unsigned char>(),
            constants.data_ptr<float>(),
            p1_winner_idx.data_ptr<int64_t>(), p2_winner_idx.data_ptr<int64_t>(),
            c1_dest_t.data_ptr<int64_t>(), c2_dest_t.data_ptr<int64_t>(),
            s1.data_ptr<int64_t>(), e1.data_ptr<int64_t>(),
            s2.data_ptr<int64_t>(), e2.data_ptr<int64_t>(),
            cx_mask_flat.data_ptr<bool>(),
            offspring_consts.data_ptr<float>(),
            n_pairs, L, K, id_C, PAD_ID,
            sbx_eta, sbx_prob, rng_seed, generation);
    }
    offspring_consts.index_copy_(0, copy_dest_t, constants.index_select(0, copy_winner_idx));

    // 3. Mutation
    // Point mutation is fused into the crossover writer. Structural (bank)
    // mutation and hoist mutation are per-individual operations:
    //   bank:    [0.5, 0.5 + 0.3*rate) of the individual draw
    //   hoist:   [0.8, 0.8 + 0.2*rate)
    // The elite slot 0 is never mutated.
    if (has_bank) {
        // Masked over all rows: no torch::nonzero, hence no host synchronisation.
        float struct_lo = 0.5f;
        float struct_hi = 0.5f + 0.3f * mutation_rate;
        int bank_size = mutation_bank.size(0);
        auto bank_indices = torch::randint(0, bank_size, {B}, long_opt);
        auto len_pop = torch::empty({B}, long_opt);
        auto len_bank = torch::empty({B}, long_opt);
        auto rand_e_pop = torch::rand({B}, float_opt);
        auto rand_e_bank = torch::rand({B}, float_opt);
        auto s_pop = torch::empty({B}, long_opt);
        auto e_pop = torch::empty({B}, long_opt);
        auto s_bank = torch::empty({B}, long_opt);
        auto e_bank = torch::empty({B}, long_opt);
        int row_blocks = (B + threads_ranges - 1) / threads_ranges;
        select_subtree_range_indirect_kernel<<<row_blocks, threads_ranges>>>(
            offspring.data_ptr<unsigned char>(), nullptr,
            token_arities.data_ptr<int32_t>(), rand_e_pop.data_ptr<float>(),
            len_pop.data_ptr<int64_t>(), s_pop.data_ptr<int64_t>(), e_pop.data_ptr<int64_t>(),
            B, L, token_arities.size(0), PAD_ID);
        select_subtree_range_indirect_kernel<<<row_blocks, threads_ranges>>>(
            mutation_bank.data_ptr<unsigned char>(), bank_indices.data_ptr<int64_t>(),
            token_arities.data_ptr<int32_t>(), rand_e_bank.data_ptr<float>(),
            len_bank.data_ptr<int64_t>(), s_bank.data_ptr<int64_t>(), e_bank.data_ptr<int64_t>(),
            B, L, token_arities.size(0), PAD_ID);

        // Grafts that would not fit are skipped instead of truncated
        // (a truncated RPN program is invalid). Slot 0 (elite) is excluded.
        auto graft_ok = torch::empty({B}, torch::TensorOptions().dtype(torch::kBool).device(device));
        graft_enable_kernel<<<row_blocks, threads_ranges>>>(
            mut_rand.data_ptr<float>(), struct_lo, struct_hi,
            len_pop.data_ptr<int64_t>(), s_pop.data_ptr<int64_t>(), e_pop.data_ptr<int64_t>(),
            s_bank.data_ptr<int64_t>(), e_bank.data_ptr<int64_t>(),
            graft_ok.data_ptr<bool>(), B, L);

        // Constants must be remapped from the pre-graft tokens.
        if (K > 0) {
            graft_constants_kernel<<<row_blocks, threads_ranges>>>(
                offspring.data_ptr<unsigned char>(), mutation_bank.data_ptr<unsigned char>(),
                nullptr, bank_indices.data_ptr<int64_t>(),
                s_pop.data_ptr<int64_t>(), e_pop.data_ptr<int64_t>(),
                s_bank.data_ptr<int64_t>(), e_bank.data_ptr<int64_t>(),
                graft_ok.data_ptr<bool>(), offspring_consts.data_ptr<float>(),
                B, L, K, id_C, PAD_ID,
                graft_const_lo, graft_const_hi, rng_seed, generation);
        }

        auto grafted = torch::empty_like(offspring);
        int64_t total = (int64_t)B * L;
        int splice_blocks = (int)((total + threads_splice - 1) / threads_splice);
        graft_splice_masked_kernel<<<splice_blocks, threads_splice>>>(
            offspring.data_ptr<unsigned char>(), mutation_bank.data_ptr<unsigned char>(),
            bank_indices.data_ptr<int64_t>(),
            s_pop.data_ptr<int64_t>(), e_pop.data_ptr<int64_t>(),
            s_bank.data_ptr<int64_t>(), e_bank.data_ptr<int64_t>(),
            graft_ok.data_ptr<bool>(), grafted.data_ptr<unsigned char>(),
            B, L, PAD_ID);
        offspring = grafted;
    }

    // Hoist mutation: masked kernel, no host-side compaction or sync.
    {
        float hoist_lo = 0.8f;
        float hoist_hi = 0.8f + 0.2f * mutation_rate;
        const int hoist_threads = 128;
        const int hoist_blocks = (B + hoist_threads - 1) / hoist_threads;
        hoist_mutation_masked_kernel<<<hoist_blocks, hoist_threads>>>(
            offspring.data_ptr<unsigned char>(),
            K > 0 ? offspring_consts.data_ptr<float>() : nullptr,
            mut_rand.data_ptr<float>(), hoist_lo, hoist_hi,
            token_arities.data_ptr<int32_t>(),
            B, L, K, token_arities.size(0), PAD_ID, id_C,
            rng_seed, generation);
    }

    // 4. NanoPSO (Constant Optimization)
    auto final_consts_out = offspring_consts;
    auto final_fit_out = torch::empty({0}, float_opt);

    if (pso_steps > 0) {
        auto gbest_pos = offspring_consts.clone();
        auto gbest_err = torch::full({B}, std::numeric_limits<float>::infinity(), float_opt);
        auto pop_expanded = offspring.repeat_interleave(pso_particles, 0); // [B*P, L]
        
        int total_particles = B * pso_particles;
        int N_data = X.size(1); // X is [Vars, N]
        
        // Initial Particles
        auto pos = offspring_consts.unsqueeze(1).repeat({1, pso_particles, 1}); // [B, P, K]
        auto jitter = torch::randn({B, pso_particles-1, K}, float_opt) * 1.0;
        
        using namespace torch::indexing;
        pos.index_put_({Slice(), Slice(1, None), Slice()}, pos.index({Slice(), Slice(1, None), Slice()}) + jitter);
        
        auto vel = torch::randn_like(pos) * 0.1;
        
        auto pbest_pos = pos.clone();
        auto pbest_err = torch::full({B, pso_particles}, std::numeric_limits<float>::infinity(), float_opt);
        
        // Pre-allocate eval outputs (Shape must match B*P x N_data)
        int num_evals = total_particles * N_data;
        auto preds = torch::empty({total_particles, N_data}, float_opt);
        auto sp = torch::empty({num_evals}, int_opt);
        auto error_flags = torch::empty({num_evals}, byte_opt);
        
        for(int step=0; step<pso_steps; ++step) {
            auto flat_pos = pos.view({-1, K}); 
            
            // Evaluate
            launch_rpn_kernel(
                pop_expanded, X, flat_pos, 
                preds, sp, error_flags,
                PAD_ID, id_x_start, 
                id_C, id_pi, id_e,
                id_0, id_1, id_2, id_3, id_4, id_5, id_6, id_10,
                op_add, op_sub, op_mul, op_div, op_pow, op_mod,
                op_sin, op_cos, op_tan,
                op_log, op_exp,
                op_sqrt, op_abs, op_neg,
                op_fact, op_floor, op_ceil, op_sign,
                op_gamma, op_lgamma,
                op_asin, op_acos, op_atan,
                pi_val, e_val,
                0  // strict_mode=0: always protected during search
            );
            
            // RMSE Logic
            auto diff = preds - Y_target.unsqueeze(0);
            auto mse = torch::mean(diff*diff, 1); // [B*P]
            auto rmse = torch::sqrt(mse);
            rmse = torch::where(torch::isnan(rmse), torch::full_like(rmse, std::numeric_limits<float>::infinity()), rmse);
            
            auto curr_err = rmse.view({B, pso_particles});
            
            launch_pso_update_bests(curr_err, pbest_err, pbest_pos, pos, gbest_err, gbest_pos);
            
            auto r1 = torch::rand({B, pso_particles, K}, float_opt);
            auto r2 = torch::rand({B, pso_particles, K}, float_opt);
            
            launch_pso_update(pos, vel, pbest_pos, gbest_pos, r1, r2, pso_w, pso_c1, pso_c2);
        }
        final_consts_out = gbest_pos;
        final_fit_out = gbest_err;
    }

    
    // Return: [NewPop, NewConsts, NewFitness, ParentIndex]. ParentIndex[i] is
    // the selected parent whose structure child i descends from (lineage for
    // age-layered selection).
    return {offspring, final_consts_out, final_fit_out, winner_idx};
}

// ============================================================
//  FUSED EVAL KERNEL — decoded program + RMSE in one pass
// ============================================================
//
//  WARP_MODE: one warp per individual (8 individuals per block). Lanes stride
//             over the samples, so any number of samples is supported.
//  BLOCK mode: one block per individual, threads stride over the samples.
//
//  Key properties:
//  1. The program is decoded once per individual by one warp (opcode, variable
//     index, resolved constant/literal value) and its stack discipline is
//     validated with a warp prefix scan. Invalid programs never execute.
//  2. Every thread of a warp runs the same program -> no divergence; dispatch
//     is a dense switch over pre-decoded opcodes, top of stack in a register.
//  3. RMSE is reduced in-kernel (warp shuffles); only [B] values are written.
//  4. Strict-mode domain errors stop the whole individual early.
//  5. SCALED: the fitness is the RMSE of the least squares fit a + b*f
//     (linear scaling). Pass 1 accumulates Welford statistics and caches the
//     predictions in shared memory (when they fit); pass 2 sums the residuals.
//  6. Fitness reuse: when reuse_parent is given, an individual whose tokens and
//     constants are bit-identical to its parent's copies the parent's fitness
//     and is not evaluated at all.
//
// ============================================================

#define FUSED_MAX_L   256    // Max formula length (decoded program in shared memory)
#define FUSED_WARPS_PER_BLOCK 8
#define FUSED_BLOCK_THREADS 256
// Shared memory budget for cached predictions of the scaled evaluator. Larger
// datasets re-evaluate the program in pass 2 instead of lowering occupancy.
#define FUSED_SCALE_CACHE_BYTES (16 * 1024)

template <typename scalar_t, bool WARP_MODE, bool STRICT, bool SCALED>
__global__ void __launch_bounds__(256)
rpn_eval_fused_kernel(
    const unsigned char* __restrict__ population,  // [B, L]
    const scalar_t* __restrict__ x,               // [Vars, D]
    const scalar_t* __restrict__ constants,        // [B, K] or nullptr
    const scalar_t* __restrict__ y_target,         // [D]
    scalar_t* __restrict__ out_rmse,               // [B]
    scalar_t* __restrict__ out_ab,                 // [B, 2] (a, b) or nullptr
    const int64_t* __restrict__ reuse_parent,      // [B] or nullptr
    const unsigned char* __restrict__ reuse_pop,   // [B_old, L]
    const scalar_t* __restrict__ reuse_consts,     // [B_old, K] or nullptr
    const scalar_t* __restrict__ reuse_fit,        // [B_old]
    int B, int D, int L, int K, int cache_D,
    RpnOpIds ids
) {
    extern __shared__ __align__(16) unsigned char fused_smem[];
    constexpr scalar_t INVALID_RMSE = std::is_same<scalar_t, double>::value
        ? (scalar_t)1e100 : (scalar_t)1e15;
    constexpr scalar_t MAX_METRIC_DIFF = std::is_same<scalar_t, double>::value
        ? (scalar_t)1e150 : (scalar_t)4e18;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t b = WARP_MODE ? ((int64_t)blockIdx.x * FUSED_WARPS_PER_BLOCK + warp)
                                : (int64_t)blockIdx.x;
    const int slot = WARP_MODE ? warp : 0;
    const int n_slots = WARP_MODE ? FUSED_WARPS_PER_BLOCK : 1;

    scalar_t* imm = reinterpret_cast<scalar_t*>(fused_smem) + (size_t)slot * L;
    unsigned char* code = fused_smem + (size_t)n_slots * L * sizeof(scalar_t) + (size_t)slot * 2 * L;
    unsigned char* aux = code + L;
    DecodedProgram<scalar_t> prog{code, aux, imm};
    scalar_t* pred_cache = nullptr;
    if (SCALED && cache_D > 0) {
        size_t off = (size_t)n_slots * L * (sizeof(scalar_t) + 2);
        off = (off + 15) & ~(size_t)15;
        pred_cache = reinterpret_cast<scalar_t*>(fused_smem + off) + (size_t)slot * cache_D;
    }

    __shared__ int s_len;
    __shared__ int s_same;
    __shared__ scalar_t s_sq[FUSED_BLOCK_THREADS / 32];
    __shared__ int s_flags[FUSED_BLOCK_THREADS / 32];
    __shared__ RpnScaleStats<scalar_t> s_stats[FUSED_BLOCK_THREADS / 32];

    if (WARP_MODE && b >= B) return;  // the whole warp leaves together

    // --- Fitness reuse: identical child of an already evaluated parent ---
    if (reuse_parent != nullptr) {
        const int64_t p = reuse_parent[b];
        bool same = (p >= 0);
        if (WARP_MODE || warp == 0) {
            if (same) {
                const unsigned char* row = population + b * (int64_t)L;
                const unsigned char* prow = reuse_pop + p * (int64_t)L;
                for (int i = lane; i < L; i += 32) same = same && (row[i] == prow[i]);
                if (K > 0 && constants != nullptr) {
                    const scalar_t* c = constants + b * (int64_t)K;
                    const scalar_t* pc = reuse_consts + p * (int64_t)K;
                    for (int k = lane; k < K; k += 32) same = same && rpn_bits_equal(c[k], pc[k]);
                }
            }
            same = __all_sync(RPN_FULL_MASK, same);
            if (!WARP_MODE && lane == 0) s_same = same ? 1 : 0;
        }
        if (!WARP_MODE) {
            __syncthreads();
            same = (s_same != 0);
        }
        if (same) {
            if ((WARP_MODE ? lane : threadIdx.x) == 0) out_rmse[b] = reuse_fit[p];
            return;
        }
    }

    int len;
    if (WARP_MODE) {
        len = rpn_decode_program_warp<scalar_t, true>(
            population + b * (int64_t)L, L, ids,
            (K > 0 && constants != nullptr) ? constants + b * (int64_t)K : nullptr, K,
            prog, lane);
        __syncwarp();
    } else {
        if (warp == 0) {
            int l = rpn_decode_program_warp<scalar_t, true>(
                population + b * (int64_t)L, L, ids,
                (K > 0 && constants != nullptr) ? constants + b * (int64_t)K : nullptr, K,
                prog, lane);
            if (lane == 0) s_len = l;
        }
        __syncthreads();
        len = s_len;
    }

    if (len == 0) {
        if ((WARP_MODE ? lane : threadIdx.x) == 0) {
            out_rmse[b] = INVALID_RMSE;
            if (out_ab != nullptr) { out_ab[2 * b] = (scalar_t)0; out_ab[2 * b + 1] = (scalar_t)1; }
        }
        return;
    }

    const int nthreads = WARP_MODE ? 32 : blockDim.x;
    const int tid = WARP_MODE ? lane : threadIdx.x;
    scalar_t sq = (scalar_t)0.0;
    bool invalid = false;
    bool overflow = false;
    RpnScaleStats<scalar_t> st = rpn_scale_empty<scalar_t>();

    for (int d0 = 0; d0 < D; d0 += nthreads) {
        const int d = d0 + tid;
        if (d < D) {
            scalar_t pred;
            bool ok = rpn_run_program<scalar_t, STRICT>(code, aux, imm, len, x, D, d, nullptr, pred);
            if (!ok || isnan(pred) || isinf(pred)) {
                invalid = true;
            } else if (SCALED) {
                if (fabs(pred) > MAX_METRIC_DIFF) {
                    overflow = true;
                } else {
                    rpn_scale_push(st, pred, y_target[d]);
                    // Each thread reads back only the samples it wrote, so the
                    // cache needs no synchronisation.
                    if (pred_cache != nullptr) pred_cache[d] = pred;
                }
            } else {
                scalar_t diff = pred - y_target[d];
                scalar_t ad = fabs(diff);
                if (isnan(diff) || isinf(diff) || ad > MAX_METRIC_DIFF) {
                    overflow = true;
                } else {
                    scalar_t s2 = diff * diff;
                    if (isinf(s2)) overflow = true;
                    else sq += s2;
                }
            }
        }
        // Any invalid sample invalidates the whole individual: stop early.
        bool any_invalid;
        if (WARP_MODE) any_invalid = __any_sync(RPN_FULL_MASK, invalid);
        else any_invalid = __syncthreads_or(invalid) != 0;
        if (any_invalid) { invalid = true; break; }
    }

    scalar_t scale_a = (scalar_t)0;
    scalar_t scale_b = (scalar_t)1;
    if (SCALED) {
        // Merge the statistics (and flags) of the whole individual.
        unsigned int f1 = (invalid ? 1u : 0u) | (overflow ? 2u : 0u);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) f1 |= __shfl_xor_sync(RPN_FULL_MASK, f1, off);
        st = rpn_scale_warp_reduce(st);
        if (!WARP_MODE) {
            if (lane == 0) { s_stats[warp] = st; s_flags[warp] = (int)f1; }
            __syncthreads();
            if (threadIdx.x == 0) {
                const int nw = blockDim.x / 32;
                RpnScaleStats<scalar_t> m = rpn_scale_empty<scalar_t>();
                unsigned int fm = 0u;
                for (int w = 0; w < nw; ++w) {
                    m = rpn_scale_merge(m, s_stats[w]);
                    fm |= (unsigned int)s_flags[w];
                }
                s_stats[0] = m;
                s_flags[0] = (int)fm;
            }
            __syncthreads();
            st = s_stats[0];
            f1 = (unsigned int)s_flags[0];
        }
        invalid = (f1 & 1u) != 0u;
        overflow = (f1 & 2u) != 0u;
        if (!invalid && !overflow && !rpn_scale_finite(st)) overflow = true;

        // Pass 2: residuals of the least squares fit.
        if (!invalid && !overflow) {
            scale_b = rpn_scale_slope(st);
            scale_a = st.my - scale_b * st.mf;
            for (int d = tid; d < D; d += nthreads) {
                scalar_t pred;
                if (pred_cache != nullptr) pred = pred_cache[d];
                else rpn_run_program<scalar_t, STRICT>(code, aux, imm, len, x, D, d, nullptr, pred);
                const scalar_t r = (y_target[d] - st.my) - scale_b * (pred - st.mf);
                sq += r * r;
            }
        }
        if (!WARP_MODE) __syncthreads();  // s_stats/s_flags are reused below
    }

    // Reduction.
    unsigned int flags = (invalid ? 1u : 0u) | (overflow ? 2u : 0u);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sq += __shfl_xor_sync(RPN_FULL_MASK, sq, off);
        flags |= __shfl_xor_sync(RPN_FULL_MASK, flags, off);
    }
    if (!WARP_MODE) {
        if (lane == 0) { s_sq[warp] = sq; s_flags[warp] = (int)flags; }
        __syncthreads();
        if (warp == 0) {
            const int nw = blockDim.x / 32;
            sq = (lane < nw) ? s_sq[lane] : (scalar_t)0.0;
            flags = (lane < nw) ? (unsigned int)s_flags[lane] : 0u;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                sq += __shfl_xor_sync(RPN_FULL_MASK, sq, off);
                flags |= __shfl_xor_sync(RPN_FULL_MASK, flags, off);
            }
        }
    }
    if ((WARP_MODE ? lane : threadIdx.x) == 0) {
        scalar_t rmse;
        if (flags & 1u) {
            rmse = INVALID_RMSE;
        } else if (flags & 2u) {
            rmse = sqrt(INVALID_RMSE);
        } else {
            rmse = sqrt(sq / (scalar_t)D);
            if (isnan(rmse) || isinf(rmse)) rmse = INVALID_RMSE;
        }
        out_rmse[b] = rmse;
        if (out_ab != nullptr) {
            out_ab[2 * b] = scale_a;
            out_ab[2 * b + 1] = scale_b;
        }
    }
}

static RpnOpIds make_op_ids(
    int PAD_ID, int id_x_start, int num_vars,
    int id_C, int id_pi, int id_e,
    int id_0, int id_1, int id_2, int id_3, int id_4, int id_5, int id_6, int id_10,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan, int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan
) {
    RpnOpIds ids;
    ids.pad = PAD_ID; ids.x_start = id_x_start; ids.num_vars = num_vars;
    ids.c = id_C; ids.pi = id_pi; ids.e = id_e;
    ids.l0 = id_0; ids.l1 = id_1; ids.l2 = id_2; ids.l3 = id_3;
    ids.l4 = id_4; ids.l5 = id_5; ids.l6 = id_6; ids.l10 = id_10;
    ids.add = op_add; ids.sub = op_sub; ids.mul = op_mul; ids.div = op_div;
    ids.pow = op_pow; ids.mod = op_mod;
    ids.sin = op_sin; ids.cos = op_cos; ids.tan = op_tan; ids.log = op_log; ids.exp = op_exp;
    ids.sqrt = op_sqrt; ids.abs = op_abs; ids.neg = op_neg;
    ids.fact = op_fact; ids.floor = op_floor; ids.ceil = op_ceil; ids.sign = op_sign;
    ids.gamma = op_gamma; ids.lgamma = op_lgamma;
    ids.asin = op_asin; ids.acos = op_acos; ids.atan = op_atan;
    return ids;
}

// ── Launcher ──
void launch_rpn_eval_fused(
    const torch::Tensor& population,   // [B, L] uint8
    const torch::Tensor& x,            // [Vars, D] float
    const torch::Tensor& constants,    // [B, K] float
    const torch::Tensor& y_target,     // [D] float
    torch::Tensor& out_rmse,           // [B] float  (pre-allocated)
    int PAD_ID, int id_x_start,
    int id_C, int id_pi, int id_e,
    int id_0, int id_1, int id_2, int id_3, int id_4, int id_5, int id_6, int id_10,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan, int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan,
    double pi_val, double e_val,
    int strict_mode,
    int launch_mode,
    int scaled,
    const torch::Tensor& out_ab,
    const torch::Tensor& reuse_parent,
    const torch::Tensor& reuse_pop,
    const torch::Tensor& reuse_consts,
    const torch::Tensor& reuse_fit
) {
    CHECK_INPUT(population);
    CHECK_INPUT(x);
    CHECK_INPUT(constants);
    CHECK_INPUT(y_target);
    CHECK_INPUT(out_rmse);

    int B = population.size(0);
    int L = population.size(1);
    int num_vars = x.size(0);
    int D = x.size(1);
    int K = (constants.dim() > 1) ? constants.size(1) : 0;

    const bool want_ab = out_ab.defined() && out_ab.numel() > 0;
    if (want_ab) {
        CHECK_INPUT(out_ab);
        TORCH_CHECK(out_ab.numel() == 2 * (int64_t)B && out_ab.scalar_type() == x.scalar_type(),
                    "out_ab must have shape [B, 2] and the dtype of x");
    }
    const bool reuse = reuse_parent.defined() && reuse_parent.numel() > 0;
    if (reuse) {
        CHECK_INPUT(reuse_parent);
        CHECK_INPUT(reuse_pop);
        CHECK_INPUT(reuse_fit);
        TORCH_CHECK(!want_ab, "out_ab cannot be combined with fitness reuse");
        TORCH_CHECK(reuse_parent.numel() == B && reuse_parent.scalar_type() == torch::kInt64,
                    "reuse_parent must be an int64 tensor with one entry per individual");
        TORCH_CHECK(reuse_pop.dim() == 2 && reuse_pop.size(1) == L &&
                    reuse_pop.scalar_type() == torch::kUInt8,
                    "reuse_pop must be uint8 with the same program length");
        TORCH_CHECK(reuse_fit.dim() == 1 && reuse_fit.numel() == reuse_pop.size(0) &&
                    reuse_fit.scalar_type() == x.scalar_type(),
                    "reuse_fit must hold one fitness per reuse_pop row, in the dtype of x");
        if (K > 0) {
            CHECK_INPUT(reuse_consts);
            TORCH_CHECK(reuse_consts.dim() == 2 && reuse_consts.size(0) == reuse_pop.size(0) &&
                        reuse_consts.size(1) == K && reuse_consts.scalar_type() == x.scalar_type(),
                        "reuse_consts must have shape [B_old, K] and the dtype of x");
        }
    }

    TORCH_CHECK(population.dim() == 2, "population must have shape [B, L]");
    TORCH_CHECK(population.scalar_type() == torch::kUInt8, "population must use uint8 tokens");
    TORCH_CHECK(x.dim() == 2, "x must have shape [Vars, D]");
    TORCH_CHECK(B > 0 && L > 0, "population must be non-empty");
    TORCH_CHECK(num_vars > 0 && num_vars <= 255, "fused evaluator supports 1..255 variables");
    TORCH_CHECK(L <= FUSED_MAX_L, "fused evaluator program length exceeds ", FUSED_MAX_L);
    TORCH_CHECK(D > 0, "fused evaluator needs at least one sample");
    TORCH_CHECK(constants.dim() == 2 && constants.size(0) == B,
                "constants must have shape [B, K]");
    TORCH_CHECK(y_target.dim() == 1 && y_target.numel() == D,
                "y_target must have shape [D]");
    TORCH_CHECK(out_rmse.dim() == 1 && out_rmse.numel() == B,
                "out_rmse must have shape [B]");
    TORCH_CHECK(constants.scalar_type() == x.scalar_type(),
                "constants dtype must match x");
    TORCH_CHECK(y_target.scalar_type() == x.scalar_type(),
                "y_target dtype must match x");
    TORCH_CHECK(out_rmse.scalar_type() == x.scalar_type(),
                "out_rmse dtype must match x");

    RpnOpIds ids = make_op_ids(
        PAD_ID, id_x_start, num_vars, id_C, id_pi, id_e,
        id_0, id_1, id_2, id_3, id_4, id_5, id_6, id_10,
        op_add, op_sub, op_mul, op_div, op_pow, op_mod,
        op_sin, op_cos, op_tan, op_log, op_exp,
        op_sqrt, op_abs, op_neg,
        op_fact, op_floor, op_ceil, op_sign,
        op_gamma, op_lgamma, op_asin, op_acos, op_atan);

    // launch_mode: 0 = block per individual, 1 = one warp per individual.
    const bool use_warp_mode = (launch_mode == 1);
    int block_dim;
    int64_t grid_dim;
    if (use_warp_mode) {
        block_dim = FUSED_BLOCK_THREADS;
        grid_dim = ((int64_t)B + FUSED_WARPS_PER_BLOCK - 1) / FUSED_WARPS_PER_BLOCK;
    } else {
        block_dim = ((D + 31) / 32) * 32;
        if (block_dim > FUSED_BLOCK_THREADS) block_dim = FUSED_BLOCK_THREADS;
        grid_dim = B;
    }
    TORCH_CHECK(grid_dim <= 2147483647LL, "fused evaluator: population too large for one launch");

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "rpn_eval_fused_kernel", ([&] {
        const int n_slots = use_warp_mode ? FUSED_WARPS_PER_BLOCK : 1;
        size_t smem = (size_t)n_slots * L * (sizeof(scalar_t) + 2);
        int cache_D = 0;
        if (scaled && (size_t)n_slots * D * sizeof(scalar_t) <= FUSED_SCALE_CACHE_BYTES) {
            cache_D = D;
            smem = ((smem + 15) & ~(size_t)15) + (size_t)n_slots * D * sizeof(scalar_t);
        }
        auto launch = [&](auto warp_tag, auto strict_tag, auto scaled_tag) {
            constexpr bool warp_mode = decltype(warp_tag)::value;
            constexpr bool strict = decltype(strict_tag)::value;
            constexpr bool sc = decltype(scaled_tag)::value;
            rpn_eval_fused_kernel<scalar_t, warp_mode, strict, sc><<<(unsigned int)grid_dim, block_dim, smem>>>(
                population.data_ptr<unsigned char>(),
                x.data_ptr<scalar_t>(),
                (constants.numel() > 0) ? constants.data_ptr<scalar_t>() : nullptr,
                y_target.data_ptr<scalar_t>(),
                out_rmse.data_ptr<scalar_t>(),
                want_ab ? out_ab.data_ptr<scalar_t>() : nullptr,
                reuse ? reuse_parent.data_ptr<int64_t>() : nullptr,
                reuse ? reuse_pop.data_ptr<unsigned char>() : nullptr,
                (reuse && K > 0) ? reuse_consts.data_ptr<scalar_t>() : nullptr,
                reuse ? reuse_fit.data_ptr<scalar_t>() : nullptr,
                B, D, L, K, cache_D, ids);
        };
        auto with_scale = [&](auto warp_tag, auto strict_tag) {
            if (scaled) launch(warp_tag, strict_tag, std::true_type{});
            else launch(warp_tag, strict_tag, std::false_type{});
        };
        if (use_warp_mode) {
            if (strict_mode) with_scale(std::true_type{}, std::true_type{});
            else with_scale(std::true_type{}, std::false_type{});
        } else {
            if (strict_mode) with_scale(std::false_type{}, std::true_type{});
            else with_scale(std::false_type{}, std::false_type{});
        }
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error in rpn_eval_fused: ", cudaGetErrorString(err));
}
