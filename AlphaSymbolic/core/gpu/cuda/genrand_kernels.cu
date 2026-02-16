/*
 * GPU-Native Random RPN Population Generation Kernel
 * 
 * Replaces the Python for-loop (30 iters × ~15 kernel launches/iter = ~450 launches)
 * with a single kernel launch. Each thread generates one valid RPN formula.
 *
 * Uses Xorshift64 PRNG per thread for fast random generation without
 * requiring pre-allocated random tensors.
 *
 * Stack balance invariant maintained per-thread:
 *   terminal(arity 0) -> stack += 1
 *   unary  (arity 1)  -> stack += 0
 *   binary (arity 2)  -> stack -= 1
 *
 * Constraints at each position j:
 *   remaining = L - j - 1
 *   new_stack >= 1
 *   new_stack <= 1 + remaining
 *   At last position: new_stack must be exactly 1
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define MAX_CATEGORY_SIZE 128  // Max tokens per category (terminals, unary, binary)
#define PAD_ID_CONST 0

// --- Xorshift64 PRNG ---
__device__ __forceinline__ float xorshift_uniform(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return (float)(x & 0xFFFFFF) / 16777216.0f;
}

__device__ __forceinline__ int xorshift_int(uint64_t* state, int n) {
    float u = xorshift_uniform(state);
    int r = (int)(u * n);
    return min(r, n - 1);  // clamp to [0, n-1]
}


/*
 * Each thread generates one valid RPN formula of max length L.
 * Terminal, unary, and binary token pools are passed in constant memory.
 *
 * Parameters:
 *   out_pop:        [B, L] int64, output population
 *   terminal_ids:   [n_terminals] int64, pool of terminal token IDs
 *   unary_ids:      [n_unary] int64, pool of unary operator IDs
 *   binary_ids:     [n_binary] int64, pool of binary operator IDs
 *   n_terminals, n_unary, n_binary: sizes of each pool
 *   B, L:           population size and max formula length
 *   seed:           base seed for PRNG (each thread adds its index)
 */
__global__ void generate_random_rpn_kernel(
    int64_t* __restrict__ out_pop,
    const int64_t* __restrict__ terminal_ids,
    const int64_t* __restrict__ unary_ids,
    const int64_t* __restrict__ binary_ids,
    int n_terminals, int n_unary, int n_binary,
    int B, int L,
    uint64_t seed
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    
    // Initialize per-thread PRNG with unique seed
    uint64_t rng_state = seed + (uint64_t)b * 6364136223846793005ULL + 1442695040888963407ULL;
    // Warm up the RNG a bit
    xorshift_uniform(&rng_state);
    xorshift_uniform(&rng_state);
    
    int64_t* row = out_pop + (int64_t)b * L;
    int stack = 0;
    int actual_len = 0;
    
    for (int j = 0; j < L; j++) {
        int remaining = L - j - 1;
        
        // Determine valid categories
        bool can_terminal = ((stack + 1) >= 1) && ((stack + 1) <= 1 + remaining);
        bool can_unary = (n_unary > 0) && (stack >= 1) && (stack <= 1 + remaining);
        bool can_binary = (n_binary > 0) && ((stack - 1) >= 1) && ((stack - 1) <= 1 + remaining);
        
        // Last position: must end at stack=1
        if (remaining == 0) {
            can_terminal = can_terminal && ((stack + 1) == 1);
            can_unary = can_unary && (stack == 1);
            can_binary = can_binary && ((stack - 1) == 1);
        }
        
        // Compute weights (equal weight per category for simplicity)
        float w_t = can_terminal ? 1.0f : 0.0f;
        float w_u = can_unary ? 1.0f : 0.0f;
        float w_b = can_binary ? 1.0f : 0.0f;
        float total_w = w_t + w_u + w_b;
        
        // Fallback to terminal if nothing valid
        if (total_w < 0.5f) {
            w_t = 1.0f;
            total_w = 1.0f;
            can_terminal = true;
            can_unary = false;
            can_binary = false;
        }
        
        // Select category using random number
        float r = xorshift_uniform(&rng_state);
        float p_t = w_t / total_w;
        float p_u = w_u / total_w;
        
        int64_t chosen;
        int delta;
        
        if (r < p_t) {
            // Terminal
            int idx = xorshift_int(&rng_state, n_terminals);
            chosen = terminal_ids[idx];
            delta = 1;
        } else if (r < p_t + p_u) {
            // Unary
            int idx = xorshift_int(&rng_state, n_unary);
            chosen = unary_ids[idx];
            delta = 0;
        } else {
            // Binary
            int idx = xorshift_int(&rng_state, n_binary);
            chosen = binary_ids[idx];
            delta = -1;
        }
        
        row[j] = chosen;
        stack += delta;
        actual_len = j + 1;
        
        // If stack == 1 and we've written enough, pad the rest
        if (stack == 1 && j > 0) {
            // Check if remaining positions can only be PAD
            // (i.e., the formula is already complete)
            // Actually, we should continue to make formulas of varying lengths.
            // Only stop early if stack == 1 and this is a reasonable stopping point.
            // For diversity, let's not stop early — always fill L positions.
            // The Python version fills all L positions too.
        }
    }
    
    // Final validation: if stack != 1, replace with simple "x0" formula
    if (stack != 1) {
        // Use first terminal as fallback
        row[0] = terminal_ids[0];
        for (int j = 1; j < L; j++) {
            row[j] = PAD_ID_CONST;
        }
    }
}


// ======================== C++ Launch Wrapper ========================

void launch_generate_random_rpn(
    torch::Tensor& population,
    const torch::Tensor& terminal_ids,
    const torch::Tensor& unary_ids,
    const torch::Tensor& binary_ids,
    uint64_t seed
) {
    CHECK_INPUT(population);
    CHECK_INPUT(terminal_ids);
    // unary_ids and binary_ids may be empty but must be on GPU
    CHECK_CUDA(unary_ids);
    CHECK_CUDA(binary_ids);
    
    int B = population.size(0);
    int L = population.size(1);
    int n_terminals = terminal_ids.size(0);
    int n_unary = unary_ids.numel();
    int n_binary = binary_ids.numel();
    
    TORCH_CHECK(n_terminals > 0, "Must have at least one terminal token");
    
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    
    generate_random_rpn_kernel<<<blocks, threads>>>(
        population.data_ptr<int64_t>(),
        terminal_ids.data_ptr<int64_t>(),
        n_unary > 0 ? unary_ids.data_ptr<int64_t>() : nullptr,
        n_binary > 0 ? binary_ids.data_ptr<int64_t>() : nullptr,
        n_terminals, n_unary, n_binary,
        B, L,
        seed
    );
}
