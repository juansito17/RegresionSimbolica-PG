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
/*
 * OPTIMIZED: Añadidos parámetros de peso por categoría.
 * term_weight/unary_weight/bin_weight vienen de GpuGlobals.OPERATOR_WEIGHTS
 * (calculados en Python). Resuelve el bug donde TERMINAL_VS_VARIABLE_PROB
 * y OPERATOR_WEIGHTS no se aplicaban en el kernel CUDA.
 */
__global__ void generate_random_rpn_kernel(
    uint8_t* __restrict__ out_pop,              // [B, L] uint8
    const uint8_t* __restrict__ terminal_ids,   // weighted pool (repeated ids = weight)
    const uint8_t* __restrict__ unary_ids,
    const uint8_t* __restrict__ binary_ids,
    int n_terminals, int n_unary, int n_binary,
    int B, int L,
    uint64_t seed,
    float term_weight,
    float unary_weight,
    float bin_weight,
    int min_len,
    int max_len_target
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;

    uint64_t rng_state = seed + (uint64_t)b * 6364136223846793005ULL + 1442695040888963407ULL;
    xorshift_uniform(&rng_state);
    xorshift_uniform(&rng_state);

    uint8_t* row = out_pop + (int64_t)b * L;

    // Ramped target size. Each formula is generated with exactly T tokens
    // (a uniform draw in [min_len, max_len_target]), which yields a spread of
    // sizes and shapes instead of stopping at the first time the stack
    // returns to one (that rule produced mostly 2-token formulas).
    // max_len_target <= 0 keeps the legacy "stop at first completion" mode.
    const bool legacy = (max_len_target <= 0);
    int T = L;
    if (!legacy) {
        int lo = min_len < 1 ? 1 : min_len;
        int hi = max_len_target < L ? max_len_target : L;
        if (hi < lo) hi = lo;
        if (lo > L) lo = hi = L;
        T = lo + xorshift_int(&rng_state, hi - lo + 1);
        // Without unary operators an expression always has odd length.
        if (n_unary == 0 && (T % 2) == 0) T = (T + 1 <= hi) ? T + 1 : T - 1;
        if (T < 1) T = 1;
    }

    int stack = 0;
    bool is_completed = false;

    for (int j = 0; j < L; j++) {
        if (is_completed || j >= T) {
            row[j] = (uint8_t)PAD_ID_CONST;
            continue;
        }

        int remaining = T - j - 1;

        // A state (s, r) can still finish at depth 1 iff s >= 1, s - 1 <= r
        // and, without unary operators, the parity of r - (s - 1) is even.
        auto feasible = [&](int s_new) {
            if (s_new < 1 || s_new - 1 > remaining) return false;
            if (n_unary == 0 && ((remaining - (s_new - 1)) & 1)) return false;
            return true;
        };
        bool can_terminal = feasible(stack + 1);
        bool can_unary = (n_unary > 0) && stack >= 1 && feasible(stack);
        bool can_binary = (n_binary > 0) && feasible(stack - 1);

        float w_t = can_terminal ? term_weight : 0.0f;
        float w_u = can_unary ? unary_weight : 0.0f;
        float w_b = can_binary ? bin_weight : 0.0f;
        float total_w = w_t + w_u + w_b;
        if (total_w <= 1e-6f) {
            // Only reachable for degenerate weights: fall back to any feasible move.
            w_t = can_terminal ? 1.0f : 0.0f;
            w_u = can_unary ? 1.0f : 0.0f;
            w_b = can_binary ? 1.0f : 0.0f;
            total_w = w_t + w_u + w_b;
            if (total_w <= 0.0f) { w_t = 1.0f; total_w = 1.0f; }
        }

        float r = xorshift_uniform(&rng_state);
        float p_t = w_t / total_w;
        float p_u = w_u / total_w;

        uint8_t chosen;
        int delta;
        if (r < p_t) {
            chosen = terminal_ids[xorshift_int(&rng_state, n_terminals)];
            delta = 1;
        } else if (r < p_t + p_u) {
            chosen = unary_ids[xorshift_int(&rng_state, n_unary)];
            delta = 0;
        } else {
            chosen = binary_ids[xorshift_int(&rng_state, n_binary)];
            delta = -1;
        }

        row[j] = chosen;
        stack += delta;

        if (legacy && stack == 1 && j > 0) is_completed = true;
    }

    if (stack != 1) {
        row[0] = terminal_ids[0];
        for (int j = 1; j < L; j++) row[j] = PAD_ID_CONST;
    }
}


// ======================== C++ Launch Wrapper ========================

void launch_generate_random_rpn(
    torch::Tensor& population,
    const torch::Tensor& terminal_ids,
    const torch::Tensor& unary_ids,
    const torch::Tensor& binary_ids,
    uint64_t seed,
    float term_weight,
    float unary_weight,
    float bin_weight,
    int min_len,
    int max_len_target
) {
    CHECK_INPUT(population);
    CHECK_INPUT(terminal_ids);
    CHECK_CUDA(unary_ids);
    CHECK_CUDA(binary_ids);

    int B = population.size(0);
    int L = population.size(1);
    int n_terminals = terminal_ids.size(0);
    int n_unary = unary_ids.numel();
    int n_binary = binary_ids.numel();

    TORCH_CHECK(n_terminals > 0, "Must have at least one terminal token");
    if (B == 0) return;

    int threads = 256;
    int blocks = (B + threads - 1) / threads;

    generate_random_rpn_kernel<<<blocks, threads>>>(
        population.data_ptr<uint8_t>(),
        terminal_ids.data_ptr<uint8_t>(),
        n_unary > 0 ? unary_ids.data_ptr<uint8_t>() : nullptr,
        n_binary > 0 ? binary_ids.data_ptr<uint8_t>() : nullptr,
        n_terminals, n_unary, n_binary,
        B, L,
        seed,
        term_weight,
        unary_weight,
        bin_weight,
        min_len,
        max_len_target
    );
}
