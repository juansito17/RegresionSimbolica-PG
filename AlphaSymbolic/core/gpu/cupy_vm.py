
import cupy as cp
import torch
from torch.utils.dlpack import to_dlpack, from_dlpack

# CUDA C++ Kernel
RPN_KERNEL_SOURCE = r'''
extern "C" __global__
void rpn_kernel(
    const long long* population,  // [eff_B, L] - Fixed for Windows (int64)
    const double* x,         // [eff_B] or [eff_B, D]
    const double* constants, // [eff_B, K]
    double* out_preds,       // [eff_B]
    int* out_sp,             // [eff_B]
    unsigned char* out_err,  // [eff_B]
    int B, int D, int L, int K, int x_dim_stride,
    // IDs
    int PAD_ID, int id_x_legacy,
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
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= B * D) return;

    // Registers for stack (size 10)
    double s[10];
    int sp = 0;
    int c_idx = 0;
    bool error = false;
    
    // Constant Limit
    double val_1e30 = 1e30;
    double val_1e9 = 1e-9;

    for (int i = 0; i < L; ++i) {
        // Read token: population[idx * L + i]
        int token = (int)population[idx * L + i];
        
        if (token == PAD_ID) continue;
        
        double val = 0.0;
        bool push = false;
        
        if (token == id_x_legacy) {
            val = x[idx]; 
            push = true;
        } 
        else if (token == id_pi) { val = pi_val; push = true; }
        else if (token == id_e) { val = e_val; push = true; }
        else if (token == id_1) { val = 1.0; push = true; }
        else if (token == id_2) { val = 2.0; push = true; }
        else if (token == id_3) { val = 3.0; push = true; }
        else if (token == id_5) { val = 5.0; push = true; }
        else if (token == id_C) {
             if (K > 0) {
                 int r_idx = c_idx;
                 if (r_idx >= K) r_idx = K - 1;
                 val = constants[idx * K + r_idx];
                 c_idx++;
             } else {
                 val = 1.0;
             }
             push = true;
        }
        
        if (push) {
            if (sp < 10) {
                s[sp] = val;
                sp++;
            }
            continue;
        }
        
        // Ops
        // Binary
        if (sp >= 2) {
             double b = s[sp-1];
             double a = s[sp-2];
             double res = 0.0;
             bool valid = false;
             
             if (token == op_add) { res = a + b; valid = true; }
             else if (token == op_sub) { res = a - b; valid = true; }
             else if (token == op_mul) { res = a * b; valid = true; }
             else if (token == op_div) {
                 if (abs(b) < val_1e9) res = val_1e30;
                 else res = a / b;
                 valid = true;
             }
             else if (token == op_mod) {
                 if (abs(b) < val_1e9) res = val_1e30;
                 else res = remainder(a, b); // fmod vs remainder?
                 valid = true;
             }
             else if (token == op_pow) {
                 res = pow(a, b);
                 if (isnan(res) || isinf(res)) res = val_1e30;
                 valid = true;
             }
             
             if (valid) {
                 s[sp-2] = res;
                 sp--;
                 continue;
             }
        }
        
        // Unary
        if (sp >= 1) {
            double a = s[sp-1];
            double res = 0.0;
            bool valid = false;
            
            if (token == op_sin) { res = sin(a); valid = true; }
            else if (token == op_cos) { res = cos(a); valid = true; }
            else if (token == op_tan) { res = tan(a); valid = true; }
            else if (token == op_abs) { res = abs(a); valid = true; }
            else if (token == op_neg) { res = -a; valid = true; }
            else if (token == op_sqrt) { res = sqrt(abs(a)); valid = true; }
            else if (token == op_log) { 
                if (a <= val_1e9) res = -val_1e30; 
                else res = log(a);
                valid = true;
            }
            else if (token == op_exp) {
                if (a > 80.0) res = val_1e30;
                else if (a < -80.0) res = 0.0;
                else res = exp(a);
                valid = true; 
            }
            else if (token == op_floor) { res = floor(a); valid = true; }
            else if (token == op_asin) {
                double v = a;
                if (v > 1.0) v = 1.0;
                if (v < -1.0) v = -1.0;
                res = asin(v);
                valid = true;
            }
            else if (token == op_acos) {
                double v = a;
                if (v > 1.0) v = 1.0;
                if (v < -1.0) v = -1.0;
                res = acos(v);
                valid = true;
            }
            else if (token == op_atan) { res = atan(a); valid = true; }
            else if (token == op_fact) {
                if (a >= 0 && a <= 170) res = exp(lgamma(a + 1.0));
                else res = val_1e30;
                valid = true;
            }
            else if (token == op_gamma) {
                if (a > -1.0) res = lgamma(a + 1.0);
                else res = val_1e30;
                valid = true;
            }
            
            if (valid) {
                s[sp-1] = res;
                continue;
            }
        }
    }
    
    out_sp[idx] = sp;
    out_err[idx] = error ? 1 : 0;
    if (sp >= 1) out_preds[idx] = s[sp-1];
    else out_preds[idx] = val_1e30;
}
'''

rpn_kernel = cp.RawKernel(RPN_KERNEL_SOURCE, 'rpn_kernel')

def run_vm_cupy(
    population: torch.Tensor, 
    x: torch.Tensor, 
    constants: torch.Tensor,
    # IDs
    PAD_ID, id_x_legacy,
    id_C, id_pi, id_e,
    id_1, id_2, id_3, id_5,
    op_add, op_sub, op_mul, op_div, op_pow, op_mod,
    op_sin, op_cos, op_tan,
    op_log, op_exp,
    op_sqrt, op_abs, op_neg,
    op_fact, op_floor, op_gamma,
    op_asin, op_acos, op_atan,
    pi_val, e_val
):
    B, L = population.shape
    D = x.shape[0]
    eff_B = B * D
    
    # Debug
    # print(f"CupyVM: B={B}, L={L}, D={D}, eff_B={eff_B}")
    # print(f"Pop shape: {population.shape}")
    # print(f"X shape: {x.shape}")
    
    try:
        # Expand Population: [B, L] -> [B, D, L] -> [B*D, L]
        # pop_indices = torch.arange(B, device=population.device).view(B, 1).expand(B, D).reshape(-1) # [0,0,0, 1,1,1...]
        # But standard expand/reshape works:
        pop_exp = population.unsqueeze(1).expand(B, D, L).reshape(eff_B, L).contiguous()

        # Expand X: [D] -> [B, D] -> [B*D] (Sequential: x0..xD, x0..xD)
        # If x is 1D:
        if x.dim() == 1:
            x_exp = x.unsqueeze(0).expand(B, D).reshape(eff_B).contiguous()
        # If x is 2D [D, 1]:
        elif x.dim() == 2 and x.shape[1] == 1:
            x_sq = x.squeeze(1)
            x_exp = x_sq.unsqueeze(0).expand(B, D).reshape(eff_B).contiguous()
        # If x is [D, V] or [V, D]? 
        # Evaluate assumes x shape matches usage.
        # Let's assume flattened for kernel if unknown
        else:
             # Just repeat X, B times?
             # x: [D, ...]
             # We want B copies of x.
             # repeat(B, 1...)
             x_exp = x.repeat(B, *([1]*(x.dim()-1))).reshape(eff_B, -1).contiguous()
             if x_exp.shape[1] == 1:
                 x_exp = x_exp.squeeze(1)
            
        const_exp = constants.unsqueeze(1).expand(B, D, constants.shape[1]).reshape(eff_B, constants.shape[1]).contiguous()
        
    except Exception as e:
        import sys
        print(f"Error in Cupy VM Expansion: {e}")
        print(f"B={B}, L={L}, D={D}")
        print(f"Pop: {population.shape}")
        print(f"X: {x.shape}")
        sys.stdout.flush()
        raise e
    
    out_preds = torch.zeros(eff_B, dtype=torch.float64, device=population.device)
    out_sp = torch.zeros(eff_B, dtype=torch.int32, device=population.device)
    out_err = torch.zeros(eff_B, dtype=torch.uint8, device=population.device)
    
    c_pop = cp.from_dlpack(to_dlpack(pop_exp))
    c_x = cp.from_dlpack(to_dlpack(x_exp))
    c_const = cp.from_dlpack(to_dlpack(const_exp))
    c_out_preds = cp.from_dlpack(to_dlpack(out_preds))
    c_out_sp = cp.from_dlpack(to_dlpack(out_sp))
    c_out_err = cp.from_dlpack(to_dlpack(out_err))
    
    block_size = 128
    grid_size = (eff_B + block_size - 1) // block_size
    
    K = constants.shape[1]
    
    rpn_kernel((grid_size,), (block_size,), (
        c_pop, c_x, c_const, c_out_preds, c_out_sp, c_out_err,
        B*D, D, L, K, 1,
        PAD_ID, id_x_legacy,
        id_C, id_pi, id_e,
        id_1, id_2, id_3, id_5,
        op_add, op_sub, op_mul, op_div, op_pow, op_mod,
        op_sin, op_cos, op_tan,
        op_log, op_exp,
        op_sqrt, op_abs, op_neg,
        op_fact, op_floor, op_gamma,
        op_asin, op_acos, op_atan,
        float(pi_val), float(e_val)
    ))
    
    return out_preds, out_sp, out_err.bool()
