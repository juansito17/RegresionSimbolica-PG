
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>

// Helper to check CUDA errors
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Stack size for RPN
// Max observed stack depth for 112-token formulas is ~14.
// 32 slots = 2.3x safety margin vs observed max, frees 128 bytes/thread of registers vs STACK_SIZE=64.
// Higher SM occupancy = more active warps = better latency hiding.
// Tested: random GP populations rarely exceed depth 28 with MAX_FORMULA_LENGTH=128.
#define STACK_SIZE 32

// Templated Constants?
// We will cast inside functions

// Device functions for unary/binary ops (Templated)

// Device functions for unary/binary ops (Templated)
template <typename T>
__device__ __forceinline__ T safe_div(T a, T b, bool &error) {
    if (abs(b) < (T)1e-9) {
        return a; // protected division
    }
    return a / b;
}

template <typename T>
__device__ __forceinline__ T safe_mod(T a, T b, bool &error) {
    if (abs(b) < (T)1e-9) {
        return (T)0.0; // protected modulo
    }
    T r = fmod(a, b);
    if ((b > 0 && r < 0) || (b < 0 && r > 0)) {
        r += b;
    }
    return r;
}

template <typename T>
__device__ __forceinline__ T safe_log(T a, bool &error) {
    return log(abs(a) + (T)1e-9); // protected log
}

template <typename T>
__device__ __forceinline__ T safe_exp(T a, bool &error) {
    T x = a;
    if (x < (T)-80.0) x = (T)-80.0;
    if (x > (T)80.0) x = (T)80.0;
    return exp(x);
}

template <typename T>
__device__ __forceinline__ T safe_sqrt(T a, bool &error) {
    return sqrt(abs(a)); // protected sqrt
}

template <typename T>
__device__ __forceinline__ T safe_pow(T a, T b, bool &error) {
    if (a != a || b != b) { error = true; return (T)0.0; }
    
    // Case (0,0) -> 1.0 (limit)
    if (abs(a) < (T)1e-10 && abs(b) < (T)1e-10) return (T)1.0;
    
    // Protected negative-base handling
    if (a < (T)0.0) {
        T ib = round(b);
        if (abs(b - ib) > (T)1e-3) {
            a = abs(a);
        } else {
            b = ib;
        }
    }
    
    // Safety: prevent extreme overflow that kills the individual
    // If a > 1 and b > 100, or similar combinations
    if (abs(a) > (T)1.0 && b > (T)80.0) b = (T)80.0;
    if (abs(a) > (T)100.0 && b > (T)10.0) b = (T)10.0;

    T res = pow(a, b);
    if (res != res || isinf(res)) { error = true; return (T)0.0; }
    return res;
}

template <typename T>
__device__ __forceinline__ T safe_asin(T a, bool &error) {
    if (a < (T)-1.0) a = (T)-1.0;
    if (a > (T)1.0) a = (T)1.0;
    return asin(a);
}

template <typename T>
__device__ __forceinline__ T safe_acos(T a, bool &error) {
    if (a < (T)-1.0) a = (T)-1.0;
    if (a > (T)1.0) a = (T)1.0;
    return acos(a);
}

template <typename T>
__device__ __forceinline__ T safe_tgamma(T a, bool &error) {
    if (a <= (T)0.0 && floor(a) == a) return (T)0.0;
    T res = tgamma(a);
    if (res != res || isinf(res)) { error = true; return (T)0.0; }
    return res;
}

template <typename T>
__device__ __forceinline__ T safe_lgamma(T a, bool &error) {
    if (a <= (T)0.0 && floor(a) == a) return (T)0.0;
    T res = lgamma(a);
    if (res != res || isinf(res)) { error = true; return (T)0.0; }
    return res;
}

// ============================================================
//  STRICT math functions — real math, error on domain violations
// ============================================================

template <typename T>
__device__ __forceinline__ T strict_div(T a, T b, bool &error) {
    if (abs(b) < (T)1e-9) { error = true; return (T)0.0; }
    return a / b;
}

template <typename T>
__device__ __forceinline__ T strict_mod(T a, T b, bool &error) {
    if (abs(b) < (T)1e-9) { error = true; return (T)0.0; }
    T r = fmod(a, b);
    if ((b > 0 && r < 0) || (b < 0 && r > 0)) r += b;
    return r;
}

template <typename T>
__device__ __forceinline__ T strict_log(T a, bool &error) {
    if (a <= (T)0.0) { error = true; return (T)0.0; }
    return log(a);
}

template <typename T>
__device__ __forceinline__ T strict_exp(T a, bool &error) {
    T res = exp(a);
    if (isinf(res)) { error = true; return (T)0.0; }
    return res;
}

template <typename T>
__device__ __forceinline__ T strict_sqrt(T a, bool &error) {
    if (a < (T)0.0) { error = true; return (T)0.0; }
    return sqrt(a);
}

template <typename T>
__device__ __forceinline__ T strict_pow(T a, T b, bool &error) {
    if (a != a || b != b) { error = true; return (T)0.0; }
    if (abs(a) < (T)1e-10 && abs(b) < (T)1e-10) return (T)1.0;
    if (a < (T)0.0) {
        T ib = round(b);
        if (abs(b - ib) > (T)1e-3) { error = true; return (T)0.0; } // non-integer exp of negative base
        b = ib;
    }
    T res = pow(a, b);
    if (res != res || isinf(res)) { error = true; return (T)0.0; }
    return res;
}

template <typename T>
__device__ __forceinline__ T strict_asin(T a, bool &error) {
    if (a < (T)-1.0 || a > (T)1.0) { error = true; return (T)0.0; }
    return asin(a);
}

template <typename T>
__device__ __forceinline__ T strict_acos(T a, bool &error) {
    if (a < (T)-1.0 || a > (T)1.0) { error = true; return (T)0.0; }
    return acos(a);
}

template <typename T>
__device__ __forceinline__ T strict_tgamma(T a, bool &error) {
    if (a <= (T)0.0 && floor(a) == a) { error = true; return (T)0.0; }
    T res = tgamma(a);
    if (res != res || isinf(res)) { error = true; return (T)0.0; }
    return res;
}

template <typename T>
__device__ __forceinline__ T strict_lgamma(T a, bool &error) {
    if (a <= (T)0.0 && floor(a) == a) { error = true; return (T)0.0; }
    T res = lgamma(a);
    if (res != res || isinf(res)) { error = true; return (T)0.0; }
    return res;
}

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
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D) return;

    int b_idx = idx / D; // Population Index
    int d_idx = idx % D; // Sample Index

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
            if (sp < STACK_SIZE) {
                stack[sp++] = val;
            }
            continue;
        }

        // --- Operators (Binary & Unary) ---
        // Fast-path resolution via another switch block might be tricky because op_* are dynamic ints,
        // but we can compile them to switch statements if we pass them statically.
        // Since op_add, op_sub are dynamic arguments to the kernel (from python mappings),
        // we cannot use them in a native C++ switch(token) case op_add:.
        // To keep the speedup without hardcoding vocabulary IDs in CUDA, we do tiered checks.
        
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
        
        // Unary — lgamma/fact/sqrt first (most-used in N-Queens target)
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
        else if (token == op_abs) res = abs(op1);
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
    
    int total_threads = B * D;
    // 256 threads/block: con STACK_SIZE=28 y float32, cada hilo usa menos registros,
    // lo que permite más bloques concurrentes por SM en la RTX 3050 (112 SM, 64K regs/SM).
    const int block_size = 256;
    const int grid_size = (total_threads + block_size - 1) / block_size;
    
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
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

// --- Phase 2: Crossover & Mutation Kernels ---

__global__ void find_subtree_ranges_kernel(
    const unsigned char* __restrict__ population, // [B, L] (uint8)
    const int* __restrict__ token_arities,  // [VocabSize]
    int64_t* __restrict__ out_starts,       // [B, L]
    int B, int L, int vocab_size, int PAD_ID
) {
    int b = blockIdx.x; // Batch index
    int tid = threadIdx.x; // Token index in sequence (0..L-1)
    
    if (b >= B || tid >= L) return;
    
    const unsigned char* my_pop = &population[b * L];
    int64_t* my_starts = &out_starts[b * L];
    
    int64_t token = (int64_t)my_pop[tid];
    
    // Default invalid
    my_starts[tid] = -1;
    
    if (token == PAD_ID) return;
    
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
    // If loop finishes and needed > 0, it's an invalid subtree (incomplete)
}

void launch_find_subtree_ranges(
    const torch::Tensor& population,
    const torch::Tensor& token_arities,
    torch::Tensor& out_starts,
    int PAD_ID
) {
    CHECK_INPUT(population);
    CHECK_INPUT(token_arities);
    CHECK_INPUT(out_starts);
    
    int B = population.size(0);
    int L = population.size(1);
    int vocab_size = token_arities.size(0);
    
    // 1 Block per Individual, L threads per block (since L is usually 30-256)
    // If L > 1024, need loops, but typically L < 128 here.
    dim3 blocks(B);
    dim3 threads(L);
    if (L > 1024) threads.x = 1024; // Simple cap, logic assumes single block per row implies L <= threads
    
    find_subtree_ranges_kernel<<<blocks, threads>>>(
        population.data_ptr<unsigned char>(),
        token_arities.data_ptr<int32_t>(),
        out_starts.data_ptr<int64_t>(),
        B, L, vocab_size, PAD_ID
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in find_subtree_ranges: %s\n", cudaGetErrorString(err));
    }
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
    int b = blockIdx.x;
    int tid = threadIdx.x;
    if (b >= B || tid >= L) return;
    
    int idx = b * L + tid;
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
    
    dim3 blocks(B);
    dim3 threads(L);
    if (L > 1024) threads.x = 1024;
    
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
    const int64_t* __restrict__ starts1, // [N]
    const int64_t* __restrict__ ends1,   // [N]
    const int64_t* __restrict__ starts2, // [N]
    const int64_t* __restrict__ ends2,   // [N]
    unsigned char* __restrict__ child1,        // [N, L] (uint8)
    unsigned char* __restrict__ child2,        // [N, L] (uint8)
    int N_pairs, int L, int PAD_ID
) {
    int n = blockIdx.x; // Pair index
    int t = threadIdx.x; // Token index in child
    
    if (n >= N_pairs || t >= L) return;
    
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
        if (t >= 0 && t < L) val_c1 = (int64_t)parent1[n * L + t];
    } else if (t < cut1) {
        // From Sub2
        int64_t src_idx = s2 + t - len_pre1;
        if (src_idx >= 0 && src_idx < L) {
            val_c1 = (int64_t)parent2[n * L + src_idx];
        }
    } else {
        // From Post1
        int64_t src_idx = e1 + 1 + t - cut1;
        if (src_idx >= 0 && src_idx < L) {
            val_c1 = (int64_t)parent1[n * L + src_idx];
        } else {
            val_c1 = (int64_t)PAD_ID;
        }
    }
    child1[n * L + t] = (unsigned char)val_c1;
    
    // --- Child 2 Construction ---
    // Child 2 = P2_Pre + P1_Sub + P2_Post
    int64_t len_pre2 = s2;
    int64_t len_sub1 = e1 - s1 + 1;
    int64_t cut2 = len_pre2 + len_sub1;
    
    int64_t val_c2 = (int64_t)PAD_ID;
    
    if (t < len_pre2) {
        if (t >= 0 && t < L) val_c2 = (int64_t)parent2[n * L + t];
    } else if (t < cut2) {
        int64_t src_idx = s1 + t - len_pre2;
        if (src_idx >= 0 && src_idx < L) {
            val_c2 = (int64_t)parent1[n * L + src_idx];
        }
    } else {
        int64_t src_idx = e2 + 1 + t - cut2;
        if (src_idx >= 0 && src_idx < L) {
            val_c2 = (int64_t)parent2[n * L + src_idx];
        } else {
            val_c2 = (int64_t)PAD_ID;
        }
    }
    child2[n * L + t] = (unsigned char)val_c2;
}

__global__ void crossover_constants_kernel(
    const unsigned char* __restrict__ p1,     // [N, L]
    const unsigned char* __restrict__ p2,     // [N, L]
    const float* __restrict__ consts1,        // [N, K]
    const float* __restrict__ consts2,        // [N, K]
    const int64_t* __restrict__ starts1,      // [N]
    const int64_t* __restrict__ ends1,        // [N]
    const int64_t* __restrict__ starts2,      // [N]
    const int64_t* __restrict__ ends2,        // [N]
    float* __restrict__ child1_consts,        // [N, K]
    float* __restrict__ child2_consts,        // [N, K]
    int N_pairs, int L, int K, int id_C
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N_pairs) return;

    int64_t s1 = starts1[n];
    int64_t e1 = ends1[n];
    int64_t s2 = starts2[n];
    int64_t e2 = ends2[n];
    
    // --- Child 1 Construction ---
    int c1_idx = 0;
    
    // P1_pre [0, s1-1]
    int p1_c = 0;
    for (int i = 0; i < s1; ++i) {
        if (p1[n * L + i] == id_C) {
            if (c1_idx < K && p1_c < K) child1_consts[n * K + c1_idx++] = consts1[n * K + p1_c];
            p1_c++;
        }
    }
    
    // P2_sub [s2, e2]
    int p2_c = 0;
    for (int i = 0; i < s2; ++i) {
        if (p2[n * L + i] == id_C) p2_c++;
    }
    for (int i = s2; i <= e2; ++i) {
        if (p2[n * L + i] == id_C) {
            if (c1_idx < K && p2_c < K) child1_consts[n * K + c1_idx++] = consts2[n * K + p2_c];
            p2_c++;
        }
    }
    
    // P1_post [e1+1, L-1]
    for (int i = s1; i <= e1; ++i) {
        if (p1[n * L + i] == id_C) p1_c++;
    }
    for (int i = e1 + 1; i < L; ++i) {
        if (p1[n * L + i] == id_C) {
            if (c1_idx < K && p1_c < K) child1_consts[n * K + c1_idx++] = consts1[n * K + p1_c];
            p1_c++;
        }
    }
    
    while (c1_idx < K) {
        int fill_idx = c1_idx;
        child1_consts[n * K + c1_idx] = consts1[n * K + (fill_idx % K)];
        c1_idx++;
    }
    
    // --- Child 2 Construction ---
    int c2_idx = 0;
    
    p2_c = 0;
    for (int i = 0; i < s2; ++i) {
        if (p2[n * L + i] == id_C) {
            if (c2_idx < K && p2_c < K) child2_consts[n * K + c2_idx++] = consts2[n * K + p2_c];
            p2_c++;
        }
    }
    
    p1_c = 0;
    for (int i = 0; i < s1; ++i) {
        if (p1[n * L + i] == id_C) p1_c++;
    }
    for (int i = s1; i <= e1; ++i) {
        if (p1[n * L + i] == id_C) {
            if (c2_idx < K && p1_c < K) child2_consts[n * K + c2_idx++] = consts1[n * K + p1_c];
            p1_c++;
        }
    }
    
    for (int i = s2; i <= e2; ++i) {
        if (p2[n * L + i] == id_C) p2_c++;
    }
    for (int i = e2 + 1; i < L; ++i) {
        if (p2[n * L + i] == id_C) {
            if (c2_idx < K && p2_c < K) child2_consts[n * K + c2_idx++] = consts2[n * K + p2_c];
            p2_c++;
        }
    }
    
    while (c2_idx < K) {
        int fill_idx = c2_idx;
        child2_consts[n * K + c2_idx] = consts2[n * K + (fill_idx % K)];
        c2_idx++;
    }
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
    int PAD_ID
) {
    int N = parent1.size(0);
    int L = parent1.size(1);
    
    dim3 blocks(N);
    dim3 threads(L);
    if (L > 1024) threads.x = 1024;
    

    crossover_splicing_kernel<<<blocks, threads>>>(
        parent1.data_ptr<unsigned char>(),
        parent2.data_ptr<unsigned char>(),
        starts1.data_ptr<int64_t>(),
        ends1.data_ptr<int64_t>(),
        starts2.data_ptr<int64_t>(),
        ends2.data_ptr<int64_t>(),
        child1.data_ptr<unsigned char>(),
        child2.data_ptr<unsigned char>(),
        N, L, PAD_ID
    );
}

// --- Hoist Mutation Kernel ---
__global__ void hoist_mutation_kernel(
    unsigned char* __restrict__ population, // [B, L]
    const int64_t* __restrict__ starts,     // [B, L] (starts of subtree ending at i)
    const float* __restrict__ rand_floats,  // [B]
    const int64_t* __restrict__ rand_ints,  // [B]
    float hoist_rate,
    int B, int L, int PAD_ID
) {
    // OPTIMIZED: 128 threads/block en vez de 1 → mejor ocupación SM (era <<<B,1>>>)
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    
    // 1. Check Probability
    if (rand_floats[b] >= hoist_rate) return;
    
    // 2. Select a Random Valid Subtree (Reservoir Sampling)
    int selected_end = -1;
    int count = 0;
    
    // Simple LCG 
    uint64_t rng = (uint64_t)rand_ints[b];
    
    for (int i = 0; i < L; ++i) {
        if (starts[b*L + i] != -1) {
            count++;
            rng = rng * 6364136223846793005ULL + 1;
            if ((rng % count) == 0) {
                selected_end = i;
            }
        }
    }
    
    if (selected_end == -1) return;
    
    int start_idx = (int)starts[b*L + selected_end];
    int end_idx = selected_end;
    int subtree_len = end_idx - start_idx + 1;
    
    // 3. Hoist (Move [start, end] to [0, len])
    unsigned char* row = &population[b*L];
    
    // Create temp buffer to avoid overwrite issues during shift?
    // Case: shift left [start, end] -> [0, len].
    // start >= 0. So target index i is always <= source index start_idx + i.
    // Safe to copy forward directly.
    
    for (int i = 0; i < subtree_len; ++i) {
        row[i] = row[start_idx + i];
    }
    
    // 4. Pad Remainder
    for (int i = subtree_len; i < L; ++i) {
        row[i] = (unsigned char)PAD_ID;
    }
}

void launch_hoist_mutation(
    torch::Tensor& population,
    const torch::Tensor& starts,
    const torch::Tensor& rand_floats,
    const torch::Tensor& rand_ints,
    float hoist_rate,
    int PAD_ID
) {
    int B = population.size(0);
    int L = population.size(1);
    
    CHECK_INPUT(population);
    CHECK_INPUT(starts);
    CHECK_INPUT(rand_floats);
    CHECK_INPUT(rand_ints);
    
    // OPTIMIZED: 128 threads/block → 128x mejor ocupación de SM (era <<<B,1>>>)
    const int hoist_threads = 128;
    const int hoist_blocks = (B + hoist_threads - 1) / hoist_threads;
    hoist_mutation_kernel<<<hoist_blocks, hoist_threads>>>(
        population.data_ptr<unsigned char>(),
        starts.data_ptr<int64_t>(),
        rand_floats.data_ptr<float>(),
        rand_ints.data_ptr<int64_t>(),
        hoist_rate,
        B, L, PAD_ID
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in hoist_mutation: %s\n", cudaGetErrorString(err));
    }
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
    if (err != cudaSuccess) {
        printf("CUDA Error in tournament_selection: %s\n", cudaGetErrorString(err));
    }
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
    int n_islands
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
    
    torch::Tensor rand_idx;
    if (n_islands > 1) {
        int island_size = B / n_islands;
        if (island_size < 1) island_size = 1; // Safety
        
        auto arange = torch::arange(B, long_opt);
        // Fixed: Use explicit floor division to stay in integer domain
        auto island_base = torch::div(arange, island_size, "floor") * island_size; 
        
        auto rand_offsets = torch::randint(0, island_size, {B, tournament_size}, long_opt);
        rand_idx = (island_base.unsqueeze(1) + rand_offsets).to(torch::kInt64).clamp(0, B-1).contiguous();
    } else {
        // Panmictic (Global)
        rand_idx = torch::randint(0, B, {B, tournament_size}, long_opt);
    }
    
    auto winner_idx = torch::empty({B}, long_opt);
    
    // Lexicase Approximation: If abs_errors provided, each tournament picks a random test case.
    torch::Tensor rand_cases;
    if (abs_errors.numel() > 0) {
        rand_cases = torch::randint(0, N_data, {B}, int_opt);
    } else {
        rand_cases = torch::empty({0}, int_opt);
    }
    
    auto fit_f32 = (fitness.scalar_type() == torch::kFloat32) ? fitness : fitness.to(torch::kFloat32);
    auto err_f32 = (abs_errors.numel() > 0) ? 
        ((abs_errors.scalar_type() == torch::kFloat32) ? abs_errors : abs_errors.to(torch::kFloat32)) : 
        torch::empty({0}, float_opt);
    
    auto lengths_f32 = (lengths.numel() > 0 && lengths.scalar_type() != torch::kFloat32) ? lengths.to(torch::kFloat32) : lengths;
    auto mad_eps_f32 = (mad_eps.defined() && mad_eps.numel() > 0 && mad_eps.scalar_type() != torch::kFloat32) ? mad_eps.to(torch::kFloat32) : mad_eps;
    launch_tournament_selection(fit_f32, err_f32, rand_idx, rand_cases, winner_idx, lengths_f32, mad_eps_f32);
    
    // --- Elitism: Preserve the best individual at index 0 ---
    auto best_idx = torch::argmin(fit_f32);
    winner_idx.index_put_({0}, best_idx);
    
    auto parents = population.index_select(0, winner_idx);
    auto parent_consts = constants.index_select(0, winner_idx);
    
    // 2. Crossover (Proper Subtree Crossover with SAFETY CHECKS)
    // BUG-ELT-1 Fix: Skip the elite (pos 0) from crossover to prevent structural contamination.
    // The elite will be preserved in the next_pop[0] without mixing as a parent.
    int n_available = B - 1;
    int n_pairs = n_available / 2;
    auto parents1 = parents.slice(0, 1, 1 + 2*n_pairs, 2).contiguous();
    auto parents2 = parents.slice(0, 2, 1 + 2*n_pairs, 2).contiguous();
    
    // Find Subtree Starts using kernel [N, L]
    auto all_starts1 = torch::zeros({n_pairs, L}, long_opt);
    auto all_starts2 = torch::zeros({n_pairs, L}, long_opt);
    
    launch_find_subtree_ranges(parents1, token_arities, all_starts1, PAD_ID);
    launch_find_subtree_ranges(parents2, token_arities, all_starts2, PAD_ID);
    
    // Actual lengths (used for safety check)
    auto lengths1 = (parents1 != PAD_ID).sum(1).to(torch::kLong);
    auto lengths2 = (parents2 != PAD_ID).sum(1).to(torch::kLong);
    
    // Safe rand index: avoid choosing 0-length if possible
    auto e1 = (torch::rand({n_pairs}, float_opt) * lengths1.to(torch::kFloat32)).to(torch::kLong).clamp(0, L-1);
    auto e2 = (torch::rand({n_pairs}, float_opt) * lengths2.to(torch::kFloat32)).to(torch::kLong).clamp(0, L-1);
    
    // Get corresponding starts
    auto s1 = all_starts1.gather(1, e1.unsqueeze(1)).squeeze(1);
    auto s2 = all_starts2.gather(1, e2.unsqueeze(1)).squeeze(1);
    
    // --- Length Safety Check ---
    // P1 new: P1_Pre + P2_Sub + P1_Post
    // P2 new: P2_Pre + P1_Sub + P2_Post
    
    auto len_pre1 = s1;
    auto len_sub2 = e2 - s2 + 1;
    auto len_post1 = lengths1 - (e1 + 1);
    auto new_len1 = len_pre1 + len_sub2 + len_post1;
    
    auto len_pre2 = s2;
    auto len_sub1 = e1 - s1 + 1;
    auto len_post2 = lengths2 - (e2 + 1);
    auto new_len2 = len_pre2 + len_sub1 + len_post2;
    
    // Mask of valid crossovers (outcome <= L)
    auto safe_mask = (new_len1 <= L) & (new_len2 <= L);
    
    // Create children
    auto child1 = torch::full_like(parents1, PAD_ID);
    auto child2 = torch::full_like(parents2, PAD_ID);
    
    // Splice! (Only efficient to do all, then revert unsafe ones)
    launch_crossover_splicing(parents1, parents2, s1, e1, s2, e2, child1, child2, PAD_ID);
    
    // Apply crossover rate mask AND Safety Mask
    // If unsafe, we treat it as "crossover failed" -> keep parents
    auto cx_prob = torch::rand({n_pairs, 1}, float_opt);
    auto want_cx = (cx_prob < crossover_rate);
    auto cx_mask = want_cx & safe_mask.unsqueeze(1);
    
    auto final_c1 = torch::where(cx_mask, child1, parents1);
    auto final_c2 = torch::where(cx_mask, child2, parents2);
    
    // --- SOTA P0: Fallback Headless Chicken Crossover ---
    // If trees are too bloated to cross (safe_mask = false) but wanted to,
    // inject a random tree from the mutation bank to immediately break stagnation.
    if (mutation_bank.numel() > 0) {
        auto fallback_mask = want_cx & (~safe_mask.unsqueeze(1));
        auto bank_size = mutation_bank.size(0);
        
        auto rand_bank_idx1 = torch::randint(0, bank_size, {n_pairs}, long_opt);
        auto rand_bank1 = mutation_bank.index_select(0, rand_bank_idx1);
        
        auto rand_bank_idx2 = torch::randint(0, bank_size, {n_pairs}, long_opt);
        auto rand_bank2 = mutation_bank.index_select(0, rand_bank_idx2);
        
        final_c1 = torch::where(fallback_mask, rand_bank1, final_c1);
        final_c2 = torch::where(fallback_mask, rand_bank2, final_c2);
    }
    
    // Fixed: Interleave back to original order [0, 1, 2, 3...]
    // cat gives [0, 2, 4...] followed by [1, 3, 5...] which scrambles islands!
    // BUG-ELT-1 Fix: Prepend the elite (parents[0]) and interleave the rest.
    auto children = torch::stack({final_c1, final_c2}, 1).reshape({2*n_pairs, L});
    auto offspring = torch::cat({parents.slice(0, 0, 1), children}, 0);
    
    auto consts1_orig = parent_consts.slice(0, 1, 1 + 2*n_pairs, 2).contiguous();
    auto consts2_orig = parent_consts.slice(0, 2, 1 + 2*n_pairs, 2).contiguous();
    
    // --- SOTA I1: Simulated Binary Crossover (SBX) for Constants ---
    // Blend parent constants into a continuous real-valued landscape before structural splicing.
    float sbx_eta = 2.0;
    auto u_sbx = torch::rand({n_pairs, K}, float_opt);
    auto beta_sbx = torch::where(u_sbx <= 0.5, 
        torch::pow(2.0 * u_sbx, 1.0 / (sbx_eta + 1.0)), 
        torch::pow(1.0 / (2.0 * torch::clamp(1.0 - u_sbx, 1e-7, 1.0)), 1.0 / (sbx_eta + 1.0)));
    
    auto mask_sbx = (torch::rand({n_pairs, K}, float_opt) < 0.5); // 50% probability to blend each constant
    auto consts1 = torch::where(mask_sbx, 0.5 * ((1.0 + beta_sbx) * consts1_orig + (1.0 - beta_sbx) * consts2_orig), consts1_orig).contiguous();
    auto consts2 = torch::where(mask_sbx, 0.5 * ((1.0 - beta_sbx) * consts1_orig + (1.0 + beta_sbx) * consts2_orig), consts2_orig).contiguous();

    
    auto child1_consts = torch::empty_like(consts1);
    auto child2_consts = torch::empty_like(consts2);
    
    int threads_consts = 256;
    int blocks_consts = (n_pairs + threads_consts - 1) / threads_consts;
    crossover_constants_kernel<<<blocks_consts, threads_consts>>>(
        parents1.data_ptr<unsigned char>(),
        parents2.data_ptr<unsigned char>(),
        consts1.data_ptr<float>(),
        consts2.data_ptr<float>(),
        s1.data_ptr<int64_t>(),
        e1.data_ptr<int64_t>(),
        s2.data_ptr<int64_t>(),
        e2.data_ptr<int64_t>(),
        child1_consts.data_ptr<float>(),
        child2_consts.data_ptr<float>(),
        n_pairs, L, K, id_C
    );
    
    // Fallback mask for offspring where crossover failed
    auto cx_mask_consts = cx_mask.squeeze(1).unsqueeze(1); // shape [n_pairs, 1]
    
    auto final_c1_consts = torch::where(cx_mask_consts, child1_consts, consts1);
    auto final_c2_consts = torch::where(cx_mask_consts, child2_consts, consts2);

    // If Headless Chicken Crossover happened, substitute with bank
    if (mutation_bank.numel() > 0) {
        auto fallback_mask_consts = (want_cx & (~safe_mask.unsqueeze(1))).squeeze(1).unsqueeze(1);
        
        // Randomly initialized constants for Headless Chicken Trees
        auto rand_c1_consts = torch::empty_like(consts1).uniform_(-10.0f, +10.0f);
        auto rand_c2_consts = torch::empty_like(consts2).uniform_(-10.0f, +10.0f);
        
        final_c1_consts = torch::where(fallback_mask_consts, rand_c1_consts, final_c1_consts);
        final_c2_consts = torch::where(fallback_mask_consts, rand_c2_consts, final_c2_consts);
    }
    
    auto children_consts = torch::stack({final_c1_consts, final_c2_consts}, 1).reshape({2*n_pairs, K});
    auto offspring_consts = torch::cat({parent_consts.slice(0, 0, 1), children_consts}, 0);
    
    // Elitism Check: Ensure first row of offspring is the global best from previous generation
    // Index 0 is now guaranteed to be the elite, and we force it again just in case of padding.
    offspring.index_put_({0}, parents.index({0}));
    offspring_consts.index_put_({0}, parent_consts.index({0}));
    
    if (offspring.size(0) < B) {
        auto last_p = parents.slice(0, B-1, B);
        auto last_c = parent_consts.slice(0, B-1, B);
        offspring = torch::cat({offspring, last_p}, 0);
        offspring_consts = torch::cat({offspring_consts, last_c}, 0);
    }
    if (offspring.size(0) > B) {
        offspring = offspring.slice(0, 0, B);
        offspring_consts = offspring_consts.slice(0, 0, B);
    }
    
    // 3. Mutation 
    // Types: Point (Standard), Structural (Bank), Hoist (New)
    // Budget Split:
    // If Bank > 0: 50% Point, 30% Structural, 20% Hoist
    // Else:        80% Point,                20% Hoist
    
    auto mut_rand = torch::rand({B}, float_opt);
    bool has_bank = (mutation_bank.numel() > 0);
    torch::Tensor point_mask, struct_mask, hoist_mask;
    
    if (has_bank) {
        point_mask = (mut_rand < 0.5);
        struct_mask = (mut_rand >= 0.5) & (mut_rand < 0.8);
        hoist_mask = (mut_rand >= 0.8);
    } else {
        point_mask = (mut_rand < 0.8);
        struct_mask = torch::zeros({B}, torch::TensorOptions().dtype(torch::kBool).device(device));
        hoist_mask = (mut_rand >= 0.8);
    }

    // BUG-ELT-1 Fix: Exclude elite (index 0) from all mutations to preserve pure best.
    point_mask.index_put_({0}, false);
    struct_mask.index_put_({0}, false);
    hoist_mask.index_put_({0}, false);
    
    // Apply mutation rate to masks? No, we apply rate INSIDE the kernels usually,
    // OR we filter here.
    // The original code passed `mutation_rate` to kernels.
    // Let's keep passing `mutation_rate` to Point and Structural.
    // For Hoist, we can handle it similarly.
    
    // A. Point Mutation Path
    // OPTIMIZED: eliminado .any().item<bool>() — cada llamada forzaba sync GPU→CPU.
    // nonzero() devuelve tensor vacío si mask=False, y index_select sobre tensor vacío es no-op.
    {
        auto point_idx = torch::nonzero(point_mask).squeeze(1);
        auto point_pop = offspring.index_select(0, point_idx);
        auto r_floats = torch::rand({point_pop.size(0), L}, float_opt);
        auto r_ints = torch::randint(0, 1000000, {point_pop.size(0), L}, long_opt);
        
        launch_mutation_kernel(point_pop, r_floats, r_ints, token_arities,
                        arity_0_ids, arity_1_ids, arity_2_ids,
                        mutation_rate, PAD_ID);
        
        offspring.index_copy_(0, point_idx, point_pop);
    }
    
    // B. Structural Mutation Path (Grafting from Bank)
    // OPTIMIZED: eliminado .any().item<bool>() y .sum().item<int>() — ambos forzaban sync GPU→CPU.
    // n_struct se obtiene de struct_idx.size(0) tras nonzero() sin sync adicional.
    {
        auto struct_idx = torch::nonzero(struct_mask).squeeze(1);
        int n_struct = (int)struct_idx.size(0);
        if (n_struct > 0) {
        auto struct_pop = offspring.index_select(0, struct_idx);
        
        int bank_size = mutation_bank.size(0);
        auto bank_indices = torch::randint(0, bank_size, {n_struct}, long_opt);
        auto bank_trees = mutation_bank.index_select(0, bank_indices);
        
        auto starts_pop = torch::zeros({n_struct, L}, long_opt);
        auto starts_bank = torch::zeros({n_struct, L}, long_opt);
        launch_find_subtree_ranges(struct_pop, token_arities, starts_pop, PAD_ID);
        launch_find_subtree_ranges(bank_trees, token_arities, starts_bank, PAD_ID);
        
        auto len_pop = (struct_pop != PAD_ID).sum(1).to(torch::kLong);
        auto len_bank = (bank_trees != PAD_ID).sum(1).to(torch::kLong);
        
        auto e_pop = (torch::rand({n_struct}, float_opt) * len_pop.to(torch::kFloat32)).to(torch::kLong).clamp(0, L-1);
        auto e_bank = (torch::rand({n_struct}, float_opt) * len_bank.to(torch::kFloat32)).to(torch::kLong).clamp(0, L-1);
        
        auto s_pop = starts_pop.gather(1, e_pop.unsqueeze(1)).squeeze(1);
        auto s_bank = starts_bank.gather(1, e_bank.unsqueeze(1)).squeeze(1);
        
        auto child = torch::full_like(struct_pop, PAD_ID);
        auto dummy_child = torch::full_like(struct_pop, PAD_ID); 
        
        launch_crossover_splicing(struct_pop, bank_trees, s_pop, e_pop, s_bank, e_bank, child, dummy_child, PAD_ID);
        
        auto m_mask = (torch::rand({n_struct}, float_opt) < mutation_rate);
        auto final_struct = torch::where(m_mask.unsqueeze(1), child, struct_pop);
        
        offspring.index_copy_(0, struct_idx, final_struct);
        } // end if (n_struct > 0)
    } // end struct_mask scope

    // C. Hoist Mutation Path (NEW)
    // OPTIMIZED: eliminado .any().item<bool>() — forzaba sync GPU→CPU.
    {
        auto hoist_idx = torch::nonzero(hoist_mask).squeeze(1);
        if (hoist_idx.size(0) > 0) {
        auto hoist_pop = offspring.index_select(0, hoist_idx);
        
        // Find subtree ranges for hoist candidates
        auto starts_hoist = torch::zeros({hoist_pop.size(0), L}, long_opt);
        launch_find_subtree_ranges(hoist_pop, token_arities, starts_hoist, PAD_ID);
        
        auto r_floats = torch::rand({hoist_pop.size(0)}, float_opt);
        auto r_ints = torch::randint(0, 1000000, {hoist_pop.size(0)}, long_opt);
        
        // Pass mutation_rate to kernel to decide if we hoist
        launch_hoist_mutation(hoist_pop, starts_hoist, r_floats, r_ints, mutation_rate, PAD_ID);
        
        offspring.index_copy_(0, hoist_idx, hoist_pop);
        } // end if (hoist_idx.size(0) > 0)
    } // end hoist_mask scope
    
    // 4. NanoPSO (Constant Optimization)
    auto gbest_pos = offspring_consts.clone();
    auto gbest_err = torch::full({B}, std::numeric_limits<float>::infinity(), float_opt);

    if (pso_steps > 0) {
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
    }

    
    // Return: [NewPop, NewConsts, NewFitness]
    return {offspring, gbest_pos, gbest_err};
}

// ============================================================
//  FUSED EVAL KERNEL — Block-per-individual + RMSE in one pass
// ============================================================
//
//  Layout: blockIdx.x = individual index (b_idx)
//          threadIdx.x = data point index (d_idx)
//          blockDim.x = WARP_DIM (≥ D, padded to next 32 multiple)
//
//  Key properties:
//  1. All threads in a block execute the SAME program → 0 warp divergence
//  2. Program loaded into __shared__ memory → 17× less global reads
//  3. RMSE computed via warp shuffle reduction → outputs only [B] floats
//     instead of [B×D] predictions (saves ~153 MB of bandwidth per call)
//  4. No integer division/modulo in the hot path
//
// ============================================================

#define FUSED_MAX_L   256    // Max formula length (matches MAX_FORMULA_LENGTH)
#define FUSED_MAX_VARS  4    // Max variables (x0, x1, x2, x3)

template <typename scalar_t>
__global__ void rpn_eval_fused_kernel(
    const unsigned char* __restrict__ population,  // [B, L]
    const scalar_t* __restrict__ x,               // [Vars, D]
    const scalar_t* __restrict__ constants,        // [B, K]
    const scalar_t* __restrict__ y_target,         // [D]
    scalar_t* __restrict__ out_rmse,               // [B]
    int B, int D, int L, int K, int num_vars,
    int PAD_ID,
    int id_x_start,
    int id_C, int id_pi, int id_e,
    int id_0, int id_1, int id_2, int id_3, int id_4, int id_5, int id_6, int id_10,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan, int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan,
    double pi_val, double e_val,
    int strict_mode
) {
    const int b_idx = blockIdx.x;
    const int d_idx = threadIdx.x;

    if (b_idx >= B) return;

    // ── 1. Load program into shared memory (strided: each thread loads multiple bytes) ──
    // BUG FIX: With blockDim=32 and L=112, a single `if (d_idx < L)` only loads bytes 0..31!
    // Use a strided loop so all L bytes are covered, regardless of blockDim.
    __shared__ unsigned char prog[FUSED_MAX_L];
    #pragma unroll 4
    for (int i = d_idx; i < L && i < FUSED_MAX_L; i += blockDim.x) {
        prog[i] = population[b_idx * L + i];
    }
    __syncthreads();

    // ── 2. Preload x values for this thread's data point ──
    const bool active = (d_idx < D);
    scalar_t xv[FUSED_MAX_VARS];
    if (active) {
        for (int v = 0; v < num_vars && v < FUSED_MAX_VARS; ++v) {
            xv[v] = x[v * D + d_idx];
        }
    }

    // ── 3. Execute RPN program (no warp divergence – same program for all threads) ──
    scalar_t stack[STACK_SIZE];
    int sp = 0;
    bool error = false;
    int c_idx = 0;

    const scalar_t ERROR_VAL = (scalar_t)1e30;

    for (int pc = 0; pc < L && pc < FUSED_MAX_L; ++pc) {
        int64_t token = (int64_t)prog[pc];
        if (token == PAD_ID) break;

        scalar_t val = (scalar_t)0.0;
        bool is_push = true;

        // Terminal dispatch
        if (token >= id_x_start && token < id_x_start + num_vars) {
            int vi = token - id_x_start;
            val = active ? (vi < FUSED_MAX_VARS ? xv[vi] : (scalar_t)0.0) : (scalar_t)0.0;
        } else if (token == id_C) {
            int r = (c_idx < K) ? c_idx : K - 1;
            val = (K > 0) ? constants[b_idx * K + r] : (scalar_t)1.0;
            c_idx++;
        } else if (token == id_0)  val = (scalar_t)0.0;
        else if (token == id_1)    val = (scalar_t)1.0;
        else if (token == id_2)    val = (scalar_t)2.0;
        else if (token == id_3)    val = (scalar_t)3.0;
        else if (token == id_4)    val = (scalar_t)4.0;
        else if (token == id_5)    val = (scalar_t)5.0;
        else if (token == id_6)    val = (scalar_t)6.0;
        else if (token == id_10)   val = (scalar_t)10.0;
        else if (token == id_pi)   val = (scalar_t)pi_val;
        else if (token == id_e)    val = (scalar_t)e_val;
        else is_push = false;

        if (is_push) {
            if (sp < STACK_SIZE) stack[sp++] = val;
            continue;
        }

        // ── Binary operators (hot path first) ──
        if (__builtin_expect(token == op_add || token == op_sub || token == op_mul
                             || token == op_div || token == op_pow || token == op_mod, 1)) {
            if (__builtin_expect(sp < 2, 0)) { error = true; break; }
            scalar_t op2 = stack[--sp];
            scalar_t op1 = stack[--sp];
            scalar_t res;
            if      (__builtin_expect(token == op_add, 1)) res = op1 + op2;
            else if (__builtin_expect(token == op_sub, 1)) res = op1 - op2;
            else if (__builtin_expect(token == op_mul, 1)) res = op1 * op2;
            else if (token == op_div) res = strict_mode ? strict_div(op1, op2, error) : safe_div(op1, op2, error);
            else if (token == op_pow) res = strict_mode ? strict_pow(op1, op2, error) : safe_pow(op1, op2, error);
            else                      res = strict_mode ? strict_mod(op1, op2, error) : safe_mod(op1, op2, error);
            if (__builtin_expect(error, 0)) break;
            stack[sp++] = res;
            continue;
        }

        // ── Unary operators (hot path: lgamma/fact/sqrt/exp/log first for N-Queens) ──
        if (__builtin_expect(sp < 1, 0)) { error = true; break; }
        scalar_t op1 = stack[--sp];
        scalar_t res;
        if      (__builtin_expect(token == op_lgamma, 1)) res = strict_mode ? strict_lgamma(op1, error)                             : safe_lgamma(op1, error);
        else if (__builtin_expect(token == op_fact,   1)) res = strict_mode ? strict_tgamma(op1 + (scalar_t)1.0, error)             : safe_tgamma(op1 + (scalar_t)1.0, error);
        else if (__builtin_expect(token == op_sqrt,   1)) res = strict_mode ? strict_sqrt(op1, error)                               : safe_sqrt(op1, error);
        else if (__builtin_expect(token == op_exp,    1)) res = strict_mode ? strict_exp(op1, error)                                : safe_exp(op1, error);
        else if (__builtin_expect(token == op_log,    1)) res = strict_mode ? strict_log(op1, error)                                : safe_log(op1, error);
        else if (token == op_sin)    res = sin(op1);
        else if (token == op_cos)    res = cos(op1);
        else if (token == op_tan)    res = tan(op1);
        else if (token == op_abs)    res = abs(op1);
        else if (token == op_neg)    res = -op1;
        else if (token == op_gamma)  res = strict_mode ? strict_tgamma(op1, error)  : safe_tgamma(op1, error);
        else if (token == op_asin)   res = strict_mode ? strict_asin(op1, error)    : safe_asin(op1, error);
        else if (token == op_acos)   res = strict_mode ? strict_acos(op1, error)    : safe_acos(op1, error);
        else if (token == op_atan)   res = atan(op1);
        else if (token == op_floor)  res = floor(op1);
        else if (token == op_ceil)   res = ceil(op1);
        else if (token == op_sign)   res = (op1 > (scalar_t)0.0) ? (scalar_t)1.0 : ((op1 < (scalar_t)0.0) ? (scalar_t)-1.0 : (scalar_t)0.0);
        else { error = true; break; }

        if (__builtin_expect(error, 0)) break;
        stack[sp++] = res;
    }

    // ── 4. Compute this thread's squared error ──
    // Invalid = any math error, stack broken, NaN/Inf pred
    scalar_t sq_err;
    bool this_invalid;
    if (active) {
        bool valid = (!error) && (sp == 1);
        scalar_t pred = valid ? stack[sp - 1] : ERROR_VAL;
        valid = valid && !isnan(pred) && !isinf(pred);
        scalar_t diff = valid ? (pred - y_target[d_idx]) : ERROR_VAL;
        sq_err = valid ? (diff * diff) : ERROR_VAL;
        this_invalid = !valid;
    } else {
        sq_err = (scalar_t)0.0;   // Idle threads contribute nothing
        this_invalid = false;
    }

    // ── 5. Warp-shuffle reduction: sum sq_err and OR any_invalid ──
    // All 32 threads in the warp participate. Idle threads (d_idx >= D) have sq_err=0.
    // Full mask (all 32 lanes participate)
    unsigned int full_mask = 0xFFFFFFFF;
    uint32_t any_invalid_u = (uint32_t)this_invalid;

    // Butterfly reduction
    for (int off = 16; off > 0; off >>= 1) {
        sq_err      += __shfl_xor_sync(full_mask, sq_err, off);
        any_invalid_u |= __shfl_xor_sync(full_mask, any_invalid_u, off);
    }

    // ── 6. Thread 0 writes output ──
    if (d_idx == 0) {
        scalar_t rmse;
        if (any_invalid_u) {
            rmse = (scalar_t)1e15;
        } else {
            scalar_t mse = sq_err / (scalar_t)D;
            rmse = sqrt(mse);
            if (isnan(rmse) || isinf(rmse)) rmse = (scalar_t)1e15;
        }
        out_rmse[b_idx] = rmse;
    }
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
    int strict_mode
) {
    CHECK_INPUT(population);
    CHECK_INPUT(x);
    CHECK_INPUT(y_target);
    if (out_rmse.numel() > 0) { /* pre-allocated */ }

    int B = population.size(0);
    int L = population.size(1);
    int num_vars = x.size(0);
    int D = x.size(1);
    int K = (constants.dim() > 1) ? constants.size(1) : 0;

    // blockDim = next multiple of 32 >= D, capped at 32 (D is always small for this problem)
    // With D=17 → 32 threads per block (17 active, 15 idle but no divergence since same program)
    const int block_dim = ((D + 31) / 32) * 32;  // Round up to warp size
    const int grid_dim  = B;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "rpn_eval_fused_kernel", ([&] {
        rpn_eval_fused_kernel<scalar_t><<<grid_dim, block_dim>>>(
            population.data_ptr<unsigned char>(),
            x.data_ptr<scalar_t>(),
            (constants.numel() > 0) ? constants.data_ptr<scalar_t>() : nullptr,
            y_target.data_ptr<scalar_t>(),
            out_rmse.data_ptr<scalar_t>(),
            B, D, L, K, num_vars,
            PAD_ID, id_x_start,
            id_C, id_pi, id_e,
            id_0, id_1, id_2, id_3, id_4, id_5, id_6, id_10,
            op_add, op_sub, op_mul, op_div, op_pow, op_mod,
            op_sin, op_cos, op_tan, op_log, op_exp,
            op_sqrt, op_abs, op_neg,
            op_fact, op_floor, op_ceil, op_sign,
            op_gamma, op_lgamma, op_asin, op_acos, op_atan,
            pi_val, e_val,
            strict_mode
        );
    }));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in rpn_eval_fused: %s\n", cudaGetErrorString(err));
    }
}
