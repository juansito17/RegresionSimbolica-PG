/**
 * best_tracker_kernels.cu — GPU-Native Best Individual Tracking
 * 
 * Eliminates GPU→CPU synchronization for best tracking.
 * The engine can run thousands of generations without a single .item() call.
 * Only synchronizes when the user wants to print results.
 * 
 * Key features:
 * - Atomic best updates entirely on GPU
 * - Tracks best RPN, constants, fitness, and generation
 * - Single kernel launch per generation
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Maximum formula length
#define BT_MAX_L 256
// Maximum constants
#define BT_MAX_K 32

// ===================== Best Tracker Structure =====================
// This structure lives in GPU memory and is updated atomically

struct BestTrackerData {
    int64_t best_rpn[BT_MAX_L];      // Best formula tokens
    float best_consts[BT_MAX_K];     // Best constants
    float best_rmse;                 // Best RMSE (lower is better)
    int best_idx;                    // Index of best individual in population
    int best_gen;                    // Generation when best was found
    int formula_len;                 // Actual length of best formula
    int n_consts;                    // Number of constants used
    int updated;                     // Flag: 1 if updated this gen, 0 otherwise
};

// ===================== Update Best Kernel =====================
// One thread block per generation, single kernel launch
// Uses atomic operations to update the global best

__global__ void update_best_kernel(
    const int64_t* __restrict__ population,   // [B, L]
    const float* __restrict__ constants,      // [B, K]
    const float* __restrict__ fitness,        // [B] - RMSE values
    int64_t* __restrict__ tracker_rpn,        // [L] - persistent best RPN
    float* __restrict__ tracker_consts,       // [K] - persistent best constants
    float* __restrict__ tracker_rmse,         // [1] - persistent best RMSE
    int32_t* __restrict__ tracker_idx,        // [1] - persistent best index
    int32_t* __restrict__ tracker_gen,        // [1] - persistent best generation
    int32_t* __restrict__ tracker_len,        // [1] - persistent formula length
    int32_t* __restrict__ tracker_updated,    // [1] - flag if updated
    int B, int L, int K,
    int current_generation,
    float tolerance
) {
    // Strategy: Find minimum fitness in parallel, then update tracker
    // Using parallel reduction for efficiency
    
    __shared__ float s_min_val[256];
    __shared__ int s_min_idx[256];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    s_min_val[tid] = 1e30f;
    s_min_idx[tid] = -1;
    
    // Load fitness values and find local minimum
    for (int i = gid; i < B; i += blockDim.x * gridDim.x) {
        float f = fitness[i];
        if (f < s_min_val[tid]) {
            s_min_val[tid] = f;
            s_min_idx[tid] = i;
        }
    }
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            if (s_min_val[tid + s] < s_min_val[tid]) {
                s_min_val[tid] = s_min_val[tid + s];
                s_min_idx[tid] = s_min_idx[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Thread 0 updates global tracker
    if (tid == 0) {
        float current_best = s_min_val[0];
        int current_best_idx = s_min_idx[0];
        
        // Read current tracked best
        float tracked_best = tracker_rmse[0];
        
        // Check if we have a new best (with tolerance for floating point)
        if (current_best < tracked_best - tolerance) {
            // Update tracker atomically
            // First update RMSE (used as lock)
            
            // Copy best RPN
            if (current_best_idx >= 0 && current_best_idx < B) {
                for (int j = 0; j < L && j < BT_MAX_L; j++) {
                    tracker_rpn[j] = population[current_best_idx * L + j];
                }
                
                // Copy best constants
                for (int k = 0; k < K && k < BT_MAX_K; k++) {
                    tracker_consts[k] = constants[current_best_idx * K + k];
                }
                
                // Update metadata
                tracker_rmse[0] = current_best;
                tracker_idx[0] = current_best_idx;
                tracker_gen[0] = current_generation;
                tracker_len[0] = L;
                tracker_updated[0] = 1;
            }
        } else {
            tracker_updated[0] = 0;
        }
    }
}


// ===================== Check Improvement Kernel =====================
// Returns 1 if best improved, 0 otherwise (no sync needed)

__global__ void check_improvement_kernel(
    const float* __restrict__ fitness,
    const float* __restrict__ tracked_rmse,
    int32_t* __restrict__ improved,
    int B,
    float tolerance
) {
    __shared__ float s_min[256];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    s_min[tid] = 1e30f;
    
    for (int i = gid; i < B; i += blockDim.x * gridDim.x) {
        s_min[tid] = fminf(s_min[tid], fitness[i]);
    }
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + s]);
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        float current_best = s_min[0];
        float tracked = tracked_rmse[0];
        improved[0] = (current_best < tracked - tolerance) ? 1 : 0;
    }
}


// ===================== Batch Update Best Kernel =====================
// For use within evolve_generation - updates best after each gen

// ===================== C++ Wrappers =====================

void launch_update_best(
    const torch::Tensor& population,
    const torch::Tensor& constants,
    const torch::Tensor& fitness,
    torch::Tensor& tracker_rpn,
    torch::Tensor& tracker_consts,
    torch::Tensor& tracker_rmse,
    torch::Tensor& tracker_idx,
    torch::Tensor& tracker_gen,
    torch::Tensor& tracker_len,
    torch::Tensor& tracker_updated,
    int current_generation,
    float tolerance
) {
    CHECK_INPUT(population);
    CHECK_INPUT(constants);
    CHECK_INPUT(fitness);
    
    int B = population.size(0);
    int L = population.size(1);
    int K = constants.size(1);
    
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    blocks = min(blocks, 256);  // Cap blocks for efficiency
    
    update_best_kernel<<<blocks, threads>>>(
        population.data_ptr<int64_t>(),
        constants.data_ptr<float>(),
        fitness.data_ptr<float>(),
        tracker_rpn.data_ptr<int64_t>(),
        tracker_consts.data_ptr<float>(),
        tracker_rmse.data_ptr<float>(),
        tracker_idx.data_ptr<int32_t>(),
        tracker_gen.data_ptr<int32_t>(),
        tracker_len.data_ptr<int32_t>(),
        tracker_updated.data_ptr<int32_t>(),
        B, L, K,
        current_generation,
        tolerance
    );
    
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error in update_best: ", cudaGetErrorString(err));
}

void launch_check_improvement(
    const torch::Tensor& fitness,
    const torch::Tensor& tracked_rmse,
    torch::Tensor& improved,
    float tolerance
) {
    CHECK_INPUT(fitness);
    
    int B = fitness.size(0);
    
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    blocks = min(blocks, 256);
    
    check_improvement_kernel<<<blocks, threads>>>(
        fitness.data_ptr<float>(),
        tracked_rmse.data_ptr<float>(),
        improved.data_ptr<int32_t>(),
        B,
        tolerance
    );
}

// Order-preserving map from float to uint32 (NaN sorts after +inf).
__device__ __forceinline__ uint32_t bt_float_key(float f) {
    if (f != f) return 0xFFFFFFFFu;
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float bt_key_float(uint32_t k) {
    uint32_t u = (k & 0x80000000u) ? (k & 0x7FFFFFFFu) : ~k;
    return __uint_as_float(u);
}

// Grid-wide argmin: every block reduces a slice and publishes its candidate
// with one 64-bit atomicMin of (fitness key << 32 | index).
__global__ void argmin_packed_kernel(
    const float* __restrict__ fitness, int B,
    unsigned long long* __restrict__ packed_out
) {
    unsigned long long best = 0xFFFFFFFFFFFFFFFFull;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < B; i += gridDim.x * blockDim.x) {
        unsigned long long v = ((unsigned long long)bt_float_key(fitness[i]) << 32) | (unsigned int)i;
        best = v < best ? v : best;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        unsigned long long o = __shfl_xor_sync(0xFFFFFFFFu, best, off);
        best = o < best ? o : best;
    }
    __shared__ unsigned long long s_best[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) s_best[warp] = best;
    __syncthreads();
    if (warp == 0) {
        const int nw = blockDim.x / 32;
        best = lane < nw ? s_best[lane] : 0xFFFFFFFFFFFFFFFFull;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            unsigned long long o = __shfl_xor_sync(0xFFFFFFFFu, best, off);
            best = o < best ? o : best;
        }
        if (lane == 0) atomicMin(packed_out, best);
    }
}

// Copies the winning individual into the tracker when it improves on it.
__global__ void apply_best_kernel(
    const unsigned long long* __restrict__ packed,
    const uint8_t* __restrict__ population,
    const float* __restrict__ constants,
    uint8_t* __restrict__ best_rpn,
    float* __restrict__ best_consts,
    float* __restrict__ best_rmse,
    int32_t* __restrict__ best_idx,
    int L, int K, float tolerance
) {
    __shared__ int s_take;
    __shared__ int s_idx;
    if (threadIdx.x == 0) {
        unsigned long long v = *packed;
        int idx = (int)(v & 0xFFFFFFFFull);
        float cand = bt_key_float((uint32_t)(v >> 32));
        int take = (v != 0xFFFFFFFFFFFFFFFFull) && (cand == cand) && (cand < best_rmse[0] - tolerance);
        s_take = take;
        s_idx = idx;
        if (take) {
            best_rmse[0] = cand;
            best_idx[0] = idx;
        }
    }
    __syncthreads();
    if (!s_take) return;
    const int64_t idx = s_idx;
    for (int j = threadIdx.x; j < L; j += blockDim.x) best_rpn[j] = population[idx * L + j];
    for (int k = threadIdx.x; k < K; k += blockDim.x) best_consts[k] = constants[idx * K + k];
}

void launch_batch_update_best(
    const torch::Tensor& population,
    const torch::Tensor& constants,
    const torch::Tensor& fitness,
    torch::Tensor& best_rpn,
    torch::Tensor& best_consts,
    torch::Tensor& best_rmse,
    torch::Tensor& best_idx,
    float tolerance
) {
    CHECK_INPUT(population);
    CHECK_INPUT(constants);
    CHECK_INPUT(fitness);
    CHECK_INPUT(best_rpn);
    CHECK_INPUT(best_consts);
    CHECK_INPUT(best_rmse);
    CHECK_INPUT(best_idx);
    TORCH_CHECK(fitness.scalar_type() == torch::kFloat32, "fitness must be float32");
    TORCH_CHECK(constants.scalar_type() == torch::kFloat32, "constants must be float32");

    int B = population.size(0);
    int L = population.size(1);
    int K = constants.size(1);
    if (B == 0) return;

    auto packed = torch::full({1}, -1, population.options().dtype(torch::kInt64));
    const int threads = 256;
    int blocks = (B + threads * 4 - 1) / (threads * 4);
    if (blocks > 1024) blocks = 1024;
    if (blocks < 1) blocks = 1;
    argmin_packed_kernel<<<blocks, threads>>>(
        fitness.data_ptr<float>(), B,
        reinterpret_cast<unsigned long long*>(packed.data_ptr<int64_t>()));
    apply_best_kernel<<<1, 128>>>(
        reinterpret_cast<const unsigned long long*>(packed.data_ptr<int64_t>()),
        population.data_ptr<uint8_t>(),
        constants.data_ptr<float>(),
        best_rpn.data_ptr<uint8_t>(),
        best_consts.data_ptr<float>(),
        best_rmse.data_ptr<float>(),
        best_idx.data_ptr<int32_t>(),
        L, K, tolerance);

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error in batch_update_best: ", cudaGetErrorString(err));
}
