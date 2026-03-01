/**
 * Diversity Kernels for GPU Genetic Programming
 * 
 * Implements:
 * 1. Structural hash computation (polynomial hash)
 * 2. Structural deduplication with atomic tracking
 * 3. Skip evaluation marking for duplicates
 * 
 * Goal: Eliminate CPU-GPU sync points in deduplication
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// Helper macros
#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// ============================================================
// Kernel 1: Compute Structural Hash for Population
// ============================================================
// Uses polynomial rolling hash: hash = sum(token[i] * base^i) mod 2^64
// Identical RPN formulas produce identical hashes (ignoring PAD tokens)

__global__ void compute_population_hashes_kernel(
    const unsigned char* __restrict__ population,  // [B, L] uint8
    int64_t* __restrict__ hashes,                  // [B] output hashes
    int32_t* __restrict__ var_presence,            // [B] optional variable bitmask (or nullptr)
    int B, int L, int PAD_ID,
    int id_x_start, int num_vars
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    
    const unsigned char* row = &population[b * L];
    
    // Polynomial hash with base 31 (common choice for string hashing)
    // hash = token[0]*31^0 + token[1]*31^1 + ... 
    // Using unsigned 64-bit for natural modulo behavior
    
    uint64_t hash = 0;
    uint64_t base_pow = 1;
    const uint64_t BASE = 31ULL;
    
    // Variable presence bitmask
    int32_t var_mask = 0;
    
    for (int i = 0; i < L; ++i) {
        int token = (int)row[i];
        
        if (token == PAD_ID) break;  // Stop at first PAD
        
        // Add to hash
        hash += (uint64_t)token * base_pow;
        base_pow *= BASE;
        
        // Track variable presence
        if (token >= id_x_start && token < id_x_start + num_vars) {
            int var_idx = token - id_x_start;
            var_mask |= (1 << var_idx);
        }
    }
    
    hashes[b] = (int64_t)hash;
    if (var_presence != nullptr) {
        var_presence[b] = var_mask;
    }
}

void launch_compute_population_hashes(
    const torch::Tensor& population,
    torch::Tensor& hashes,
    torch::Tensor& var_presence,
    int PAD_ID,
    int id_x_start,
    int num_vars
) {
    CHECK_INPUT(population);
    CHECK_INPUT(hashes);
    
    int B = population.size(0);
    int L = population.size(1);
    
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    
    int32_t* var_ptr = (var_presence.defined() && var_presence.numel() > 0) 
        ? var_presence.data_ptr<int32_t>() 
        : nullptr;
    
    compute_population_hashes_kernel<<<blocks, threads>>>(
        population.data_ptr<unsigned char>(),
        hashes.data_ptr<int64_t>(),
        var_ptr,
        B, L, PAD_ID,
        id_x_start, num_vars
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in compute_population_hashes: %s\n", cudaGetErrorString(err));
    }
}

// ============================================================
// Kernel 2: Structural Deduplication (Find Unique Hashes)
// ============================================================
// Uses atomic operations to track first occurrence of each hash
// Returns: duplicate_mask[b] = 1 if individual b is a duplicate
//          original_index[b] = index of the first occurrence (or self)

// Hash table size (must be power of 2 for efficient modulo)
// 2^20 = 1M entries, good for populations up to ~500K
#define HASH_TABLE_SIZE (1 << 20)
#define HASH_TABLE_MASK (HASH_TABLE_SIZE - 1)

__global__ void structural_dedup_kernel(
    const int64_t* __restrict__ hashes,        // [B]
    int64_t* __restrict__ hash_table,          // [HASH_TABLE_SIZE] first occurrence indices
    int32_t* __restrict__ duplicate_mask,      // [B] 1 if duplicate
    int64_t* __restrict__ original_index,      // [B] index of original (or self)
    int B
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    
    int64_t my_hash = hashes[b];
    int slot = (int)(my_hash & HASH_TABLE_MASK);  // Simple modulo via bitmask
    
    // Atomic: try to claim this slot
    // hash_table[slot] = -1 initially (empty)
    // We use atomicCAS to atomically check-and-set
    
    int64_t expected = -1;
    int64_t my_idx = (int64_t)b;
    
    // Try to claim the slot
    int64_t old_val = atomicCAS((unsigned long long*)&hash_table[slot], 
                                 (unsigned long long)expected, 
                                 (unsigned long long)my_idx);
    
    if (old_val == -1) {
        // We claimed the slot -> this is the first occurrence
        duplicate_mask[b] = 0;
        original_index[b] = my_idx;
    } else {
        // Slot already occupied -> check if same hash (collision vs duplicate)
        // Note: Different hashes can map to same slot (collision)
        // We need to check if the hash at old_val equals our hash
        
        // Linear probing to handle collisions
        int attempts = 0;
        int found_original = -1;
        
        while (attempts < 16) {  // Limit probing attempts
            int64_t other_idx = hash_table[slot];
            
            if (other_idx >= 0 && other_idx < B) {
                if (hashes[other_idx] == my_hash) {
                    // Found a match!
                    found_original = (int)other_idx;
                    break;
                }
            }
            
            // Collision with different hash -> probe next slot
            slot = (slot + 1) & HASH_TABLE_MASK;
            
            // Try to claim this new slot
            old_val = atomicCAS((unsigned long long*)&hash_table[slot],
                               (unsigned long long)expected,
                               (unsigned long long)my_idx);
            
            if (old_val == -1) {
                // Claimed new slot -> first occurrence (just had collision)
                duplicate_mask[b] = 0;
                original_index[b] = my_idx;
                return;
            }
            
            attempts++;
        }
        
        if (found_original >= 0) {
            // This is a duplicate
            duplicate_mask[b] = 1;
            original_index[b] = found_original;
        } else {
            // Too many collisions or probe failed -> treat as unique (safe fallback)
            duplicate_mask[b] = 0;
            original_index[b] = my_idx;
        }
    }
}

void launch_structural_dedup(
    const torch::Tensor& hashes,
    torch::Tensor& hash_table,      // Pre-allocated, initialized to -1
    torch::Tensor& duplicate_mask,
    torch::Tensor& original_index
) {
    CHECK_INPUT(hashes);
    CHECK_INPUT(hash_table);
    CHECK_INPUT(duplicate_mask);
    CHECK_INPUT(original_index);
    
    int B = hashes.size(0);
    
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    
    structural_dedup_kernel<<<blocks, threads>>>(
        hashes.data_ptr<int64_t>(),
        hash_table.data_ptr<int64_t>(),
        duplicate_mask.data_ptr<int32_t>(),
        original_index.data_ptr<int64_t>(),
        B
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in structural_dedup: %s\n", cudaGetErrorString(err));
    }
}

// ============================================================
// Kernel 3: Count Unique Hashes (for statistics)
// ============================================================
__global__ void count_unique_kernel(
    const int32_t* __restrict__ duplicate_mask,
    int64_t* __restrict__ unique_count,
    int B
) {
    __shared__ int local_count;
    
    if (threadIdx.x == 0) {
        local_count = 0;
    }
    __syncthreads();
    
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < B && duplicate_mask[b] == 0) {
        atomicAdd(&local_count, 1);
    }
    __syncthreads();
    
    if (threadIdx.x == 0) {
        atomicAdd((unsigned long long*)unique_count, (unsigned long long)local_count);
    }
}

int64_t launch_count_unique(const torch::Tensor& duplicate_mask) {
    CHECK_INPUT(duplicate_mask);
    
    int B = duplicate_mask.size(0);
    auto options = torch::TensorOptions().dtype(torch::kInt64).device(duplicate_mask.device());
    torch::Tensor unique_count = torch::zeros({1}, options);
    
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    
    count_unique_kernel<<<blocks, threads>>>(
        duplicate_mask.data_ptr<int32_t>(),
        unique_count.data_ptr<int64_t>(),
        B
    );
    
    return unique_count.item<int64_t>();
}

// ============================================================
// Kernel 4: Replace Duplicates with Random Individuals
// ============================================================
// This kernel marks positions that need replacement
// Actual generation happens in Python/another kernel

__global__ void mark_replacement_positions_kernel(
    const int32_t* __restrict__ duplicate_mask,
    int64_t* __restrict__ replacement_positions,
    int64_t* __restrict__ n_replacements,
    int B
) {
    __shared__ int local_idx;
    
    if (threadIdx.x == 0) {
        local_idx = 0;
    }
    __syncthreads();
    
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (b < B && duplicate_mask[b] == 1) {
        int my_idx = atomicAdd(&local_idx, 1);
        // We can't directly write to replacement_positions[my_idx] safely
        // Instead, use global atomic to get position
        int global_pos = atomicAdd((unsigned long long*)n_replacements, 1ULL);
        replacement_positions[global_pos] = b;
    }
}

int64_t launch_get_replacement_positions(
    const torch::Tensor& duplicate_mask,
    torch::Tensor& replacement_positions,
    torch::Tensor& n_replacements
) {
    CHECK_INPUT(duplicate_mask);
    CHECK_INPUT(replacement_positions);
    CHECK_INPUT(n_replacements);
    
    int B = duplicate_mask.size(0);
    n_replacements.zero_();
    
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    
    mark_replacement_positions_kernel<<<blocks, threads>>>(
        duplicate_mask.data_ptr<int32_t>(),
        replacement_positions.data_ptr<int64_t>(),
        n_replacements.data_ptr<int64_t>(),
        B
    );
    
    return n_replacements.item<int64_t>();
}

// ============================================================
// Kernel 5: Batch Variable Presence Check
// ============================================================
// Efficiently check which variables each formula uses
// Used for multi-variable optimization

__global__ void compute_var_presence_kernel(
    const unsigned char* __restrict__ population,
    int32_t* __restrict__ var_presence,
    int B, int L, int PAD_ID,
    int id_x_start, int num_vars
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    
    const unsigned char* row = &population[b * L];
    int32_t mask = 0;
    
    for (int i = 0; i < L; ++i) {
        int token = (int)row[i];
        if (token == PAD_ID) break;
        
        if (token >= id_x_start && token < id_x_start + num_vars) {
            int var_idx = token - id_x_start;
            mask |= (1 << var_idx);
        }
    }
    
    var_presence[b] = mask;
}

void launch_compute_var_presence(
    const torch::Tensor& population,
    torch::Tensor& var_presence,
    int PAD_ID,
    int id_x_start,
    int num_vars
) {
    CHECK_INPUT(population);
    CHECK_INPUT(var_presence);
    
    int B = population.size(0);
    int L = population.size(1);
    
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    
    compute_var_presence_kernel<<<blocks, threads>>>(
        population.data_ptr<unsigned char>(),
        var_presence.data_ptr<int32_t>(),
        B, L, PAD_ID,
        id_x_start, num_vars
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error in compute_var_presence: %s\n", cudaGetErrorString(err));
    }
}