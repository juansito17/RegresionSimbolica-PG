
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
#define STACK_SIZE 10
#define ERROR_VAL 1e300
#define EPSILON 1e-9
#define EPSILON 1e-9

// Device functions for unary/binary ops
__device__ __forceinline__ double safe_div(double a, double b) {
    if (abs(b) < EPSILON) return ERROR_VAL;
    return a / b;
}

__device__ __forceinline__ double safe_mod(double a, double b) {
    if (abs(b) < EPSILON) return ERROR_VAL;
    return remainder(a, b);
}

__device__ __forceinline__ double safe_log(double a) {
    if (a <= EPSILON) return -ERROR_VAL; // Log(0) or neg
    return log(a);
}

__device__ __forceinline__ double safe_exp(double a) {
    if (a > 80.0) return ERROR_VAL;
    if (a < -80.0) return 0.0;
    return exp(a);
}

__device__ __forceinline__ double safe_sqrt(double a) {
    if (a < 0) return sqrt(-a); // sqrt(abs(a)) for safety
    return sqrt(a);
}

__device__ __forceinline__ double safe_pow(double a, double b) {
    double res = pow(a, b);
    if (isnan(res) || isinf(res)) return ERROR_VAL;
    return res;
}

__device__ __forceinline__ double safe_asin(double a) {
    if (a > 1.0) a = 1.0;
    if (a < -1.0) a = -1.0;
    return asin(a);
}

__device__ __forceinline__ double safe_acos(double a) {
    if (a > 1.0) a = 1.0;
    if (a < -1.0) a = -1.0;
    return acos(a);
}

__device__ __forceinline__ double safe_gamma(double a) {
    if (a <= -1.0) return ERROR_VAL; 
    return lgamma(a + 1.0); 
}

extern "C" __global__ void rpn_eval_kernel(
    const int64_t* __restrict__ population,  // [B, L]
    const double* __restrict__ x,         // [Vars, D] (Column Major for Coalescing)
    const double* __restrict__ constants, // [B, K]
    double* __restrict__ out_preds,       // [B, D]
    int* __restrict__ out_sp,             // [B, D]
    unsigned char* __restrict__ out_error, // [B, D]
    int B, int D, int L, int K, int num_vars,
    // ID Mappings passed as scalars
    int PAD_ID, 
    int id_x_start, 
    int id_C, int id_pi, int id_e,
    int id_1, int id_2, int id_3, int id_5,
    // Ops
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_gamma,
    int op_asin, int op_acos, int op_atan,
    // Values
    double pi_val, double e_val
) {
    // Global Index: [0 ... B*D - 1]
    // Uses Implicit Expansion: No need to expand tensors
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D) return;

    // implicit b, d
    int b = idx / D; // Population Index
    int d = idx % D; // Sample Index

    // Registers
    double stack[STACK_SIZE];
    int sp = 0;
    bool error = false;
    int c_idx = 0; // Constants pointer

    const int64_t* my_prog = &population[b * L];
    
    for (int pc = 0; pc < L; ++pc) {
        int64_t token = my_prog[pc];
        
        if (token == PAD_ID) continue;

        double val = 0.0;
        bool is_push = true;

        // --- Operands ---
        if (token >= id_x_start && token < id_x_start + num_vars) {
            int v_idx = token - id_x_start;
            // X is [Vars, D]. x[v, d] = x[v * D + d]
            // Access is coalesced because adjacent threads (d, d+1) read adjacent addresses
            val = x[v_idx * D + d]; 
        }
        else if (token == id_1) val = 1.0;
        else if (token == id_2) val = 2.0;
        else if (token == id_3) val = 3.0;
        else if (token == id_5) val = 5.0;
        else if (token == id_pi) val = pi_val;
        else if (token == id_e) val = e_val;
        else if (token == id_C) {
             if (K > 0) {
                 int r_idx = c_idx;
                 if (r_idx >= K) r_idx = K - 1;
                 val = constants[b * K + r_idx];
                 c_idx++;
             } else {
                 val = 1.0;
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
            double op2 = stack[--sp];
            double op1 = stack[--sp];
            double res = 0.0;
            
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
        double op1 = stack[--sp];
        double res = 0.0;
        
        if (token == op_sin) res = sin(op1);
        else if (token == op_cos) res = cos(op1);
        else if (token == op_tan) res = tan(op1);
        else if (token == op_abs) res = abs(op1);
        else if (token == op_neg) res = -op1;
        else if (token == op_sqrt) res = safe_sqrt(op1);
        else if (token == op_log) res = safe_log(op1);
        else if (token == op_exp) res = safe_exp(op1);
        else if (token == op_floor) res = floor(op1);
        else if (token == op_asin) res = safe_asin(op1);
        else if (token == op_acos) res = safe_acos(op1);
        else if (token == op_atan) res = atan(op1);
        else if (token == op_fact) res = safe_gamma(op1); 
        else if (token == op_gamma) res = safe_gamma(op1); // Same for now
        
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
    int id_1, int id_2, int id_3, int id_5,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_gamma,
    int op_asin, int op_acos, int op_atan,
    double pi_val, double e_val
) {
    CHECK_INPUT(population);
    CHECK_INPUT(x);
    // constants can be empty
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
    
    rpn_eval_kernel<<<grid_size, block_size>>>(
        population.data_ptr<int64_t>(),
        x.data_ptr<double>(),
        constants.data_ptr<double>(),
        out_preds.data_ptr<double>(),
        out_sp.data_ptr<int>(),
        out_error.data_ptr<unsigned char>(),
        B, D, L, K, num_vars,
        PAD_ID, 
        id_x_start, 
        id_C, id_pi, id_e,
        id_1, id_2, id_3, id_5,
        op_add, op_sub, op_mul, op_div, op_pow, op_mod,
        op_sin, op_cos, op_tan,
        op_log, op_exp,
        op_sqrt, op_abs, op_neg,
        op_fact, op_floor, op_gamma,
        op_asin, op_acos, op_atan,
        pi_val, e_val
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}
