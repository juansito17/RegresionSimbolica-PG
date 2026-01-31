
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
#define STACK_SIZE 64

// Templated Constants?
// We will cast inside functions

// Device functions for unary/binary ops (Templated)

// Device functions for unary/binary ops (Templated)
template <typename T>
__device__ __forceinline__ T safe_div(T a, T b) {
    if (abs(b) < 1e-9) return nan(""); // Enforce NaN on division by zero
    return a / b;
}

template <typename T>
__device__ __forceinline__ T safe_mod(T a, T b) {
    if (abs(b) < 1e-9) return nan(""); // Enforce NaN on modulo by zero
    // Python-style Modulo: Result has sign of divisor (b)
    T r = fmod(a, b);
    if ((b > 0 && r < 0) || (b < 0 && r > 0)) {
        r += b;
    }
    return r;
}



template <typename T>
__device__ __forceinline__ T safe_log(T a) {
    if (a <= 0.0) return nan(""); 
    return log(a);
}



template <typename T>
__device__ __forceinline__ T safe_exp(T a) {
    if (a < -100.0) return (T)0.0;
    return exp(a); // Let overflow result in Inf/NaN
}

template <typename T>
__device__ __forceinline__ T safe_sqrt(T a) {
    if (a < 0) return nan(""); // Enforce NaN
    return sqrt(a);
}

template <typename T>
__device__ __forceinline__ T safe_pow(T a, T b) {
    // pow(neg, float) -> NaN usually.
    T res = pow(a, b);
    // If we want to be strict, built-in pow is usually strictly NaN for negative base with fractional exp.
    return res;
}


template <typename T>
__device__ __forceinline__ T safe_asin(T a) {
    return asin(a);
}

template <typename T>
__device__ __forceinline__ T safe_acos(T a) {
    return acos(a);
}


template <typename T>
__device__ __forceinline__ T safe_tgamma(T a) {
    // poles: negative integers or 0
    if (a <= 0.0 && floor(a) == a) return nan(""); 
    return tgamma(a);
}

template <typename T>
__device__ __forceinline__ T safe_lgamma(T a) {
    // poles: negative integers or 0
    if (a <= 0.0 && floor(a) == a) return nan(""); 
    return lgamma(a); 
}

template <typename T>
__device__ __forceinline__ T safe_fact(T a) {
    // Standard Factorial: x! = gamma(x + 1)
    // Poles at x = -1, -2, ... => a + 1 <= 0 integer
    T arg = a + 1.0;
    if (arg <= 0.0 && floor(arg) == arg) return nan(""); 
    return tgamma(arg); 
}

// TEMPLATED KERNEL
template <typename scalar_t>
__global__ void rpn_eval_kernel(
    const int64_t* __restrict__ population,  // [B, L]
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
    int id_0, int id_1, int id_2, int id_3, int id_5, int id_10,
    // Ops
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan,
    // Values
    double pi_val, double e_val
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

    const int64_t* my_prog = &population[b_idx * L];
    const scalar_t ERROR_VAL = (scalar_t)1e300; // Match Python INF_VAL for float64

    for (int pc = 0; pc < L; ++pc) {
        int64_t token = my_prog[pc];
        
        if (token == PAD_ID) continue;

        scalar_t val = (scalar_t)0.0;
        bool is_push = true;

        // --- Operands ---
        if (token >= id_x_start && token < id_x_start + num_vars) {
            int v_idx = token - id_x_start;
            val = x[v_idx * D + d_idx]; 
        }
        else if (token == id_0) val = (scalar_t)0.0;
        else if (token == id_1) val = (scalar_t)1.0;
        else if (token == id_2) val = (scalar_t)2.0;
        else if (token == id_3) val = (scalar_t)3.0;
        else if (token == id_5) val = (scalar_t)5.0;
        else if (token == id_10) val = (scalar_t)10.0;
        else if (token == id_pi) val = (scalar_t)pi_val;
        else if (token == id_e) val = (scalar_t)e_val;
        else if (token == id_C) {
             if (K > 0) {
                 int r_idx = c_idx;
                 if (r_idx >= K) r_idx = K - 1;
                 val = constants[b_idx * K + r_idx];
                 c_idx++;
             } else {
                 val = (scalar_t)1.0;
             }
        }
        else {
            is_push = false;
        }
        
        if (is_push) {
            if (sp < STACK_SIZE) {
                stack[sp++] = val;
            }
            continue;
        }

        // --- Operators ---
        // Binary
        if (token == op_add || token == op_sub || token == op_mul || token == op_div || token == op_pow || token == op_mod) {
            if (sp < 2) { error = true; break; }
            scalar_t op2 = stack[--sp];
            scalar_t op1 = stack[--sp];
            scalar_t res = (scalar_t)0.0;
            
            if (token == op_add) res = op1 + op2;
            else if (token == op_sub) res = op1 - op2;
            else if (token == op_mul) res = op1 * op2;
            else if (token == op_div) res = safe_div(op1, op2);
            else if (token == op_pow) res = safe_pow(op1, op2);
            else if (token == op_mod) res = safe_mod(op1, op2);
            
            stack[sp++] = res;
            continue;
        }
        
        // Unary
        if (sp < 1) { error = true; break; }
        scalar_t op1 = stack[--sp];
        scalar_t res = (scalar_t)0.0;
        
        if (token == op_sin) res = sin(op1);
        else if (token == op_cos) res = cos(op1);
        else if (token == op_tan) res = tan(op1);
        else if (token == op_abs) res = abs(op1);
        else if (token == op_neg) res = -op1;
        else if (token == op_sqrt) res = safe_sqrt(op1);
        else if (token == op_log) res = safe_log(op1);
        else if (token == op_exp) res = safe_exp(op1);
        else if (token == op_floor) res = floor(op1);
        else if (token == op_ceil) res = ceil(op1);
        else if (token == op_sign) res = (op1 > 0) ? (scalar_t)1.0 : ((op1 < 0) ? (scalar_t)-1.0 : (scalar_t)0.0);
        else if (token == op_asin) res = safe_asin(op1);
        else if (token == op_acos) res = safe_acos(op1);
        else if (token == op_atan) res = atan(op1);
        else if (token == op_fact) res = safe_tgamma(op1 + (scalar_t)1.0);
        else if (token == op_gamma) res = safe_tgamma(op1);
        else if (token == op_lgamma) res = safe_lgamma(op1);
        
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
    int id_0, int id_1, int id_2, int id_3, int id_5, int id_10,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan,
    double pi_val, double e_val
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
    const int block_size = 256;
    const int grid_size = (total_threads + block_size - 1) / block_size;
    
    // Dispatch based on X type (float or double)
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "rpn_eval_kernel", ([&] {
        rpn_eval_kernel<scalar_t><<<grid_size, block_size>>>(
            population.data_ptr<int64_t>(),
            x.data_ptr<scalar_t>(),
            constants.data_ptr<scalar_t>(),
            out_preds.data_ptr<scalar_t>(),
            out_sp.data_ptr<int32_t>(),
            out_error.data_ptr<uint8_t>(),
            B, D, L, K, num_vars,
            PAD_ID, 
            id_x_start, 
            id_C, id_pi, id_e,
            id_0, id_1, id_2, id_3, id_5, id_10,
            op_add, op_sub, op_mul, op_div, op_pow, op_mod,
            op_sin, op_cos, op_tan,
            op_log, op_exp,
            op_sqrt, op_abs, op_neg,
            op_fact, op_floor, op_ceil, op_sign,
            op_gamma, op_lgamma,
            op_asin, op_acos, op_atan,
            pi_val, e_val
        );
    }));
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

// --- Phase 2: Crossover & Mutation Kernels ---

__global__ void find_subtree_ranges_kernel(
    const int64_t* __restrict__ population, // [B, L]
    const int* __restrict__ token_arities,  // [VocabSize]
    int64_t* __restrict__ out_starts,       // [B, L]
    int B, int L, int vocab_size, int PAD_ID
) {
    int b = blockIdx.x; // Batch index
    int tid = threadIdx.x; // Token index in sequence (0..L-1)
    
    if (b >= B || tid >= L) return;
    
    const int64_t* my_pop = &population[b * L];
    int64_t* my_starts = &out_starts[b * L];
    
    int64_t token = my_pop[tid];
    
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
        int64_t t = my_pop[j];
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
        population.data_ptr<int64_t>(),
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
    int64_t* __restrict__ population,        // [B, L]
    const float* __restrict__ rand_floats,   // [B, L] (0..1)
    const int64_t* __restrict__ rand_ints,   // [B, L] (random integers)
    const int* __restrict__ token_arities,   // [VocabSize]
    const int64_t* __restrict__ arity_0_ids, int n_0,
    const int64_t* __restrict__ arity_1_ids, int n_1,
    const int64_t* __restrict__ arity_2_ids, int n_2,
    float mutation_rate,
    int B, int L, int vocab_size, int PAD_ID
) {
    int b = blockIdx.x;
    int tid = threadIdx.x;
    if (b >= B || tid >= L) return;
    
    int idx = b * L + tid;
    int64_t token = population[idx];
    
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
        new_token = arity_0_ids[rand_val % n_0];
    } else if (arity == 1 && n_1 > 0) {
        new_token = arity_1_ids[rand_val % n_1];
    } else if (arity == 2 && n_2 > 0) {
        new_token = arity_2_ids[rand_val % n_2];
    }
    
    population[idx] = new_token;
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
        population.data_ptr<int64_t>(),
        rand_floats.data_ptr<float>(),
        rand_ints.data_ptr<int64_t>(),
        token_arities.data_ptr<int32_t>(),
        arity_0_ids.data_ptr<int64_t>(), arity_0_ids.numel(),
        arity_1_ids.data_ptr<int64_t>(), arity_1_ids.numel(),
        arity_2_ids.data_ptr<int64_t>(), arity_2_ids.numel(),
        mutation_rate,
        B, L, vocab_size, PAD_ID
    );
}

__global__ void crossover_splicing_kernel(
    const int64_t* __restrict__ parent1, // [N, L]
    const int64_t* __restrict__ parent2, // [N, L]
    const int64_t* __restrict__ starts1, // [N]
    const int64_t* __restrict__ ends1,   // [N]
    const int64_t* __restrict__ starts2, // [N]
    const int64_t* __restrict__ ends2,   // [N]
    int64_t* __restrict__ child1,        // [N, L]
    int64_t* __restrict__ child2,        // [N, L]
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
    
    int64_t val_c1 = PAD_ID;
    
    if (t < len_pre1) {
        val_c1 = parent1[n * L + t];
    } else if (t < cut1) {
        // From Sub2
        // t corresponds to index relative to start of sub2
        // offset = t - len_pre1
        // src_idx = s2 + offset
        val_c1 = parent2[n * L + (s2 + t - len_pre1)];
    } else {
        // From Post1
        // offset = t - cut1
        // src_idx = e1 + 1 + offset
        int64_t src_idx = e1 + 1 + t - cut1;
        if (src_idx < L) {
            val_c1 = parent1[n * L + src_idx];
        } else {
            val_c1 = PAD_ID;
        }
    }
    child1[n * L + t] = val_c1;
    
    // --- Child 2 Construction ---
    // Child 2 = P2_Pre + P1_Sub + P2_Post
    int64_t len_pre2 = s2;
    int64_t len_sub1 = e1 - s1 + 1;
    int64_t cut2 = len_pre2 + len_sub1;
    
    int64_t val_c2 = PAD_ID;
    
    if (t < len_pre2) {
        val_c2 = parent2[n * L + t];
    } else if (t < cut2) {
        val_c2 = parent1[n * L + (s1 + t - len_pre2)];
    } else {
        int64_t src_idx = e2 + 1 + t - cut2;
        if (src_idx < L) {
            val_c2 = parent2[n * L + src_idx];
        } else {
            val_c2 = PAD_ID;
        }
    }
    child2[n * L + t] = val_c2;
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
        parent1.data_ptr<int64_t>(),
        parent2.data_ptr<int64_t>(),
        starts1.data_ptr<int64_t>(),
        ends1.data_ptr<int64_t>(),
        starts2.data_ptr<int64_t>(),
        ends2.data_ptr<int64_t>(),
        child1.data_ptr<int64_t>(),
        child2.data_ptr<int64_t>(),
        N, L, PAD_ID
    );
}

// --- Phase 3: Tournament Selection ---

__global__ void tournament_selection_kernel(
    const float* __restrict__ fitness,      // [PopSize]
    const int64_t* __restrict__ rand_idx,   // [PopSize, TourSize]
    int64_t* __restrict__ selected_idx,     // [PopSize]
    int pop_size, int tour_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;
    
    // My tournament candidates
    // Access rand_idx[idx, 0..k]
    // We assume rand_idx is flattened [PopSize * TourSize]
    
    const int64_t* my_candidates = &rand_idx[idx * tour_size];
    
    int64_t best_idx = my_candidates[0];
    float best_fit = fitness[best_idx];
    
    for (int k = 1; k < tour_size; ++k) {
        int64_t candidate = my_candidates[k];
        float fit = fitness[candidate];
        // Smaller RMSE is better
        if (fit < best_fit) {
            best_fit = fit;
            best_idx = candidate;
        }
    }
    
    selected_idx[idx] = best_idx;
}

void launch_tournament_selection(
    const torch::Tensor& fitness,
    const torch::Tensor& rand_idx,
    torch::Tensor& selected_idx
) {
    // fitness: [B]
    // rand_idx: [B, K]
    // selected_idx: [B]
    
    CHECK_INPUT(fitness);
    CHECK_INPUT(rand_idx);
    CHECK_INPUT(selected_idx);
    
    int B = fitness.size(0);
    int K = rand_idx.size(1);
    
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    
    // Dispatch based on fitness type? Assuming float32 fitness for now based on Plan
    // If double, need template. Implementation assumes float.
    
    tournament_selection_kernel<<<blocks, threads>>>(
        fitness.data_ptr<float>(),
        rand_idx.data_ptr<int64_t>(),
        selected_idx.data_ptr<int64_t>(),
        B, K
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
    int id_0, int id_1, int id_2, int id_3, int id_5, int id_10,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan,
    double pi_val, double e_val
);

// NOTE: launch_find_subtree_ranges and launch_crossover_splicing are defined above


std::vector<torch::Tensor> evolve_generation(
    torch::Tensor population,      // [B, L]
    torch::Tensor constants,       // [B, K]
    torch::Tensor fitness,         // [B]
    torch::Tensor X,               // [Vars, N_data] (Transposed for RPN kernel)
    torch::Tensor Y_target,        // [N_data]
    torch::Tensor token_arities,   // [VocabSize] int32
    torch::Tensor arity_0_ids,     // [n0] int64
    torch::Tensor arity_1_ids,     // [n1] int64
    torch::Tensor arity_2_ids,     // [n2] int64
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
    int id_0, int id_1, int id_2, int id_3, int id_5, int id_10,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan,
    double pi_val, double e_val
) {
    // Full Orchestrator: Selection + Crossover + Mutation + PSO
    
    int B = population.size(0);
    int L = population.size(1);
    int K = constants.size(1);
    auto device = population.device();
    auto float_opt = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto long_opt = torch::TensorOptions().dtype(torch::kInt64).device(device);
    auto int_opt = torch::TensorOptions().dtype(torch::kInt32).device(device);
    auto byte_opt = torch::TensorOptions().dtype(torch::kUInt8).device(device);
    
    // 1. Selection (Tournament)
    auto rand_idx = torch::randint(0, B, {B, tournament_size}, long_opt);
    auto winner_idx = torch::empty({B}, long_opt);
    
    auto fit_f32 = fitness.to(torch::kFloat32);
    launch_tournament_selection(fit_f32, rand_idx, winner_idx);
    
    auto parents = population.index_select(0, winner_idx);
    auto parent_consts = constants.index_select(0, winner_idx);
    
    // 2. Crossover (Proper Subtree Crossover)
    int n_pairs = B / 2;
    auto parents1 = parents.slice(0, 0, 2*n_pairs, 2).contiguous();
    auto parents2 = parents.slice(0, 1, 2*n_pairs, 2).contiguous();
    
    // Find Subtree Starts using kernel [N, L]
    auto all_starts1 = torch::zeros({n_pairs, L}, long_opt);
    auto all_starts2 = torch::zeros({n_pairs, L}, long_opt);
    
    launch_find_subtree_ranges(parents1, token_arities, all_starts1, PAD_ID);
    launch_find_subtree_ranges(parents2, token_arities, all_starts2, PAD_ID);
    
    // Pick random crossover points (ends) that are NOT PAD
    auto lengths1 = (parents1 != PAD_ID).sum(1).to(torch::kLong);
    auto lengths2 = (parents2 != PAD_ID).sum(1).to(torch::kLong);
    
    // Safe rand index: avoid choosing 0-length if possible
    auto e1 = (torch::rand({n_pairs}, float_opt) * lengths1.to(torch::kFloat32)).to(torch::kLong).clamp(0, L-1);
    auto e2 = (torch::rand({n_pairs}, float_opt) * lengths2.to(torch::kFloat32)).to(torch::kLong).clamp(0, L-1);
    
    // Get corresponding starts
    auto s1 = all_starts1.gather(1, e1.unsqueeze(1)).squeeze(1);
    auto s2 = all_starts2.gather(1, e2.unsqueeze(1)).squeeze(1);
    
    // Create children
    auto child1 = torch::full_like(parents1, PAD_ID);
    auto child2 = torch::full_like(parents2, PAD_ID);
    
    // Splice!
    launch_crossover_splicing(parents1, parents2, s1, e1, s2, e2, child1, child2, PAD_ID);
    
    // Apply crossover rate mask
    auto cx_mask = (torch::rand({n_pairs, 1}, float_opt) < crossover_rate);
    auto final_c1 = torch::where(cx_mask, child1, parents1);
    auto final_c2 = torch::where(cx_mask, child2, parents2);
    
    auto offspring = torch::cat({final_c1, final_c2}, 0);
    
    auto consts1 = parent_consts.slice(0, 0, 2*n_pairs, 2);
    auto consts2 = parent_consts.slice(0, 1, 2*n_pairs, 2);
    auto offspring_consts = torch::cat({consts1, consts2}, 0);
    
    // Handle odd population
    if (B % 2 != 0) {
        auto last_p = parents.slice(0, B-1, B);
        auto last_c = parent_consts.slice(0, B-1, B);
        offspring = torch::cat({offspring, last_p}, 0);
        offspring_consts = torch::cat({offspring_consts, last_c}, 0);
    }
    
    // 3. Mutation
    auto rand_floats = torch::rand({B, L}, float_opt);
    auto rand_ints = torch::randint(0, 100, {B, L}, long_opt);
    
    launch_mutation_kernel(offspring, rand_floats, rand_ints, token_arities,
                    arity_0_ids, arity_1_ids, arity_2_ids,
                    mutation_rate, PAD_ID);
    
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
        
        // Pre-allocate eval outputs
        auto preds = torch::empty({total_particles, N_data}, float_opt);
        auto sp = torch::empty({total_particles, STACK_SIZE}, int_opt);
        auto error_flags = torch::empty({total_particles}, byte_opt);
        
        for(int step=0; step<pso_steps; ++step) {
            auto flat_pos = pos.view({-1, K}); 
            
            // Evaluate
            launch_rpn_kernel(
                pop_expanded, X, flat_pos, 
                preds, sp, error_flags,
                PAD_ID, id_x_start, 
                id_C, id_pi, id_e,
                id_0, id_1, id_2, id_3, id_5, id_10,
                op_add, op_sub, op_mul, op_div, op_pow, op_mod,
                op_sin, op_cos, op_tan,
                op_log, op_exp,
                op_sqrt, op_abs, op_neg,
                op_fact, op_floor, op_ceil, op_sign,
                op_gamma, op_lgamma,
                op_asin, op_acos, op_atan,
                pi_val, e_val
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

