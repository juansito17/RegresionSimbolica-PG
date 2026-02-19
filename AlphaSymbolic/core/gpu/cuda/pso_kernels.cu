
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// --- Pseudo Random Generator (Xorshift) for Kernel ---
__device__ __forceinline__ float rand_uniform(uint64_t* state) {
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return (float)(x & 0xFFFFFF) / 16777216.0f;
}

// Optimization: Use pre-generated random numbers passed from PyTorch to avoid managing RNG state per thread if possible.
// Or just passing tensors of rand numbers.

// --- PSO Update Kernel ---
// V = w*V + c1*r1*(Pbest - X) + c2*r2*(Gbest - X)
// X = X + V
template <typename scalar_t>
__global__ void pso_update_kernel(
    scalar_t* __restrict__ pos,        // [B, P, K] (Flattened B*P*K)
    scalar_t* __restrict__ vel,        // [B, P, K]
    const scalar_t* __restrict__ pbest, // [B, P, K]
    const scalar_t* __restrict__ gbest, // [B, K] -> Broadcast to [B, P, K]
    const scalar_t* __restrict__ r1,    // [B, P, K]
    const scalar_t* __restrict__ r2,    // [B, P, K]
    float w, float c1, float c2,
    int B, int P, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = B * P * K;
    if (idx >= total) return;
    
    // Decompose index to find which Gbest to use
    // idx = ((b * P) + p) * K + k
    
    int k = idx % K;
    int tmp = idx / K;
    int p = tmp % P;
    int b = tmp / P;
    
    scalar_t my_pos = pos[idx];
    scalar_t my_vel = vel[idx];
    scalar_t my_pbest = pbest[idx];
    
    // Gbest is [B, K], so index is b * K + k
    scalar_t my_gbest = gbest[b * K + k];
    
    scalar_t my_r1 = r1[idx];
    scalar_t my_r2 = r2[idx];
    
    // Update Vel
    scalar_t new_vel = w * my_vel + c1 * my_r1 * (my_pbest - my_pos) + c2 * my_r2 * (my_gbest - my_pos);
    
    // Clamp velocity? (Optional, usually good practice)
    
    // Update Pos
    scalar_t new_pos = my_pos + new_vel;
    
    vel[idx] = new_vel;
    pos[idx] = new_pos;
}

// --- Update Bests Kernel ---
template <typename scalar_t>
__global__ void pso_update_bests_kernel(
    const scalar_t* __restrict__ current_err, // [B, P]
    scalar_t* __restrict__ pbest_err,         // [B, P]
    scalar_t* __restrict__ pbest_pos,         // [B, P, K]
    const scalar_t* __restrict__ current_pos, // [B, P, K]
    scalar_t* __restrict__ gbest_err,         // [B]
    scalar_t* __restrict__ gbest_pos,         // [B, K]
    int B, int P, int K
) {
    // We parallelize over particles: [B, P]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * P) return;
    
    int b = idx / P;
    // int p = idx % P;
    
    scalar_t curr = current_err[idx];
    scalar_t best = pbest_err[idx];
    
    // 1. Update PBest
    if (curr < best) {
        pbest_err[idx] = curr;
        // Copy pos
        int start_k = idx * K;
        for (int k = 0; k < K; ++k) {
            pbest_pos[start_k + k] = current_pos[start_k + k];
        }
    }
    
    // 2. Update GBest (Atomic? or use a reduction later?)
    // Trying to do atomic float min is hard. 
    // Alternate strategy: Each block handles one 'b'?
    // If P is small (20), one thread can scan all P for a batch 'b'.
}

// Specialized kernel for Gbest update (One block per individual B, threads loop over P)
template <typename scalar_t>
__global__ void pso_update_gbest_kernel(
    const scalar_t* __restrict__ pbest_err, // [B, P]
    const scalar_t* __restrict__ pbest_pos, // [B, P, K]
    scalar_t* __restrict__ gbest_err,       // [B]
    scalar_t* __restrict__ gbest_pos,       // [B, K]
    int B, int P, int K
) {
    int b = blockIdx.x;
    if (b >= B) return;
    
    int tid = threadIdx.x;
    scalar_t local_best_err = (scalar_t)1e30;
    int best_p = -1;

    // Load error for this thread's particle, if it exists
    if (tid < P) {
        local_best_err = pbest_err[b * P + tid];
        best_p = tid;
    }

    // Warp-level reduction for minimum error (assuming 32 threads per block)
    for (int offset = 16; offset > 0; offset /= 2) {
        scalar_t other_err = __shfl_down_sync(0xffffffff, local_best_err, offset);
        int other_p = __shfl_down_sync(0xffffffff, best_p, offset);
        if (other_err < local_best_err) {
            local_best_err = other_err;
            best_p = other_p;
        }
    }

    // Thread 0 collects the global minimum of the warp
    if (tid == 0) {
        // Only update if it beats the current global best
        if (best_p >= 0 && local_best_err < gbest_err[b]) {
            gbest_err[b] = local_best_err;
            // Copy pos
            for (int k = 0; k < K; ++k) {
                gbest_pos[b * K + k] = pbest_pos[(b * P + best_p) * K + k];
            }
        }
    }
}

void launch_pso_update(
    torch::Tensor& pos,
    torch::Tensor& vel,
    const torch::Tensor& pbest,
    const torch::Tensor& gbest,
    const torch::Tensor& r1,
    const torch::Tensor& r2,
    float w, float c1, float c2
) {
    CHECK_INPUT(pos);
    CHECK_INPUT(vel);
    
    int B_dim = pos.size(0);
    int P_dim = pos.size(1);
    int K_dim = pos.size(2);
    
    int total = B_dim * P_dim * K_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(pos.scalar_type(), "pso_update_kernel", ([&] {
        pso_update_kernel<scalar_t><<<blocks, threads>>>(
            pos.data_ptr<scalar_t>(),
            vel.data_ptr<scalar_t>(),
            pbest.data_ptr<scalar_t>(),
            gbest.data_ptr<scalar_t>(),
            r1.data_ptr<scalar_t>(),
            r2.data_ptr<scalar_t>(),
            w, c1, c2,
            B_dim, P_dim, K_dim
        );
    }));
}

void launch_pso_update_bests(
    const torch::Tensor& current_err,
    torch::Tensor& pbest_err,
    torch::Tensor& pbest_pos,
    const torch::Tensor& current_pos,
    torch::Tensor& gbest_err,
    torch::Tensor& gbest_pos
) {
    // Shapes:
    // current_err: [B, P]
    // pbest_pos: [B, P, K]
    
    int B = pbest_err.size(0);
    int P = pbest_err.size(1);
    int K = pbest_pos.size(2);
    
    // 1. Update PBests
    int total = B * P;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(pbest_err.scalar_type(), "pso_update_bests_kernel", ([&] {
        pso_update_bests_kernel<scalar_t><<<blocks, threads>>>(
            current_err.data_ptr<scalar_t>(),
            pbest_err.data_ptr<scalar_t>(),
            pbest_pos.data_ptr<scalar_t>(),
            current_pos.data_ptr<scalar_t>(),
            gbest_err.data_ptr<scalar_t>(),
            gbest_pos.data_ptr<scalar_t>(),
            B, P, K
        );
    }));
    
    // 2. Update GBests (Reduce)
    // One block per B, 32 threads per block (one warp)
    AT_DISPATCH_FLOATING_TYPES(pbest_err.scalar_type(), "pso_update_gbest_kernel", ([&] {
        pso_update_gbest_kernel<scalar_t><<<B, 32>>>(
            pbest_err.data_ptr<scalar_t>(),
            pbest_pos.data_ptr<scalar_t>(),
            gbest_err.data_ptr<scalar_t>(),
            gbest_pos.data_ptr<scalar_t>(),
            B, P, K
        );
    }));
}
