
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
