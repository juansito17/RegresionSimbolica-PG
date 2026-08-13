#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>

#define STACK_SIZE 32

// ============================================================================
// Phase 6: Native L-BFGS-B Jacobians via Custom CUDA Autograd
// Computes partial derivatives dL/dC for the constants in the RPN tree.
// ============================================================================

template <typename scalar_t>
__global__ void rpn_eval_backward_kernel(
    const unsigned char* __restrict__ population, // [B, L]
    const scalar_t* __restrict__ x,               // [Vars, D]
    const scalar_t* __restrict__ constants,       // [B, K]
    const scalar_t* __restrict__ grad_output,     // [B, D] (dL/dY from MSE/RMSE)
    scalar_t* __restrict__ grad_constants,        // [B, K]
    int B, int D, int L, int K, int num_vars,
    int PAD_ID, int id_x_start, 
    int id_C, int id_pi, int id_e,
    int id_0, int id_1, int id_2, int id_3, int id_4, int id_5, int id_6, int id_10,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan,
    double pi_val, double e_val
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * D) return;

    int b_idx = idx / D; // Population Index
    int d_idx = idx % D; // Sample Index

    // RPN Forward Execution structures
    scalar_t val_stack[STACK_SIZE];
    int sp = 0;
    
    // =========================================================================
    // Reverse-Mode Auto-Differentiation (Adjoint Method) for RPN
    // -------------------------------------------------------------------------
    // An RPN execution builds a tree. To compute exact derivatives w.r.t
    // the constants `C_i`, we do:
    // 1. FORWARD PASS: Execute RPN, but record the values at every instruction in a Tape.
    //    We also need to know the 'parent' of each operation to backpropagate.
    //    Because RPN naturally forms a tree, if we save the `stack` at each step,
    //    we can reconstruct the topology. However, saving the full stack (32*L) 
    //    is too expensive for GPU registers.
    //
    //    Instead, we record a single `tape[L]` containing the scalar output 
    //    of the operator at `pc`.
    // 
    // 2. REVERSE PASS: Start from pc = L-1 down to 0.
    //    Maintain an `adjoint` array: `adj[pc]` = dLoss / d(node_at_pc).
    //    Initialize `adj[root_pc] = grad_output` (from MSE/RMSE).
    //    For each operation at `pc`:
    //       - Pop its inputs' PCs from an `arg_stack`.
    //       - Compute local gradients: d(op) / d(left), d(op) / d(right).
    //       - Add (local_grad * adj[pc]) to the arguments' adjoints.
    //       - If the argument is a Constant (`id_C`), accumulate into `grad_constants`.
    // =========================================================================

    // Tapes
    scalar_t val_tape[64];  // Output value of the instruction at pc
    scalar_t adj_tape[64];  // Adjoint (gradient) of the instruction at pc
    
    // To track tree topology without pointers, we use an index stack.
    // When an instruction executes, it pops indices of its arguments.
    int idx_stack[STACK_SIZE]; 
    int args_left[64];      // PC of the left argument for instruction at pc
    int args_right[64];     // PC of the right argument (or -1 if unary/terminal)
    
    // Constants mapping
    int const_index[64];    // Which K index does a token at pc map to?
    int c_idx = 0;          // C counter during forward pass
    
    const unsigned char* my_prog = &population[b_idx * L];
    
    // --- FORWARD PASS ---
    int root_pc = -1;
    for (int pc = 0; pc < L; ++pc) {
        int64_t token = (int64_t)my_prog[pc];
        if (token == PAD_ID) break;
        
        args_left[pc] = -1;
        args_right[pc] = -1;
        const_index[pc] = -1;
        adj_tape[pc] = (scalar_t)0.0;
        
        scalar_t val = (scalar_t)0.0;
        bool is_push = true;
        
        // Terminals
        if (token >= id_x_start && token < id_x_start + num_vars) {
            int v_idx = token - id_x_start;
            val = x[v_idx * D + d_idx]; 
        }
        else if (token == id_0) val = (scalar_t)0.0;
        else if (token == id_1) val = (scalar_t)1.0;
        else if (token == id_2) val = (scalar_t)2.0;
        else if (token == id_3) val = (scalar_t)3.0;
        else if (token == id_4) val = (scalar_t)4.0;
        else if (token == id_5) val = (scalar_t)5.0;
        else if (token == id_6) val = (scalar_t)6.0;
        else if (token == id_10) val = (scalar_t)10.0;
        else if (token == id_pi) val = (scalar_t)pi_val;
        else if (token == id_e) val = (scalar_t)e_val;
        else if (token == id_C) {
             if (K > 0) {
                 int r_idx = (c_idx < K) ? c_idx : (K - 1);
                 val = constants[b_idx * K + r_idx];
                 const_index[pc] = r_idx;
                 c_idx++;
             } else {
                 val = (scalar_t)1.0;
             }
        }
        else {
            is_push = false;
        }
        
        if (is_push) {
            val_tape[pc] = val;
            if (sp < STACK_SIZE) {
                idx_stack[sp++] = pc;
            }
            root_pc = pc;
            continue;
        }
        
        // Operations
        // Binary
        if (token == op_add || token == op_sub || token == op_mul || token == op_div || token == op_pow || token == op_mod) {
            if (sp < 2) break;
            int right_pc = idx_stack[--sp];
            int left_pc  = idx_stack[--sp];
            scalar_t op2 = val_tape[right_pc];
            scalar_t op1 = val_tape[left_pc];
            
            args_left[pc] = left_pc;
            args_right[pc] = right_pc;
            
            if (token == op_add) val = op1 + op2;
            else if (token == op_sub) val = op1 - op2;
            else if (token == op_mul) val = op1 * op2;
            else if (token == op_div) {
                if (abs(op2) < (scalar_t)1e-9) val = op1; else val = op1 / op2;
            }
            else if (token == op_pow) {
                // Simplified safe pow
                if (op1 < 0) {
                    scalar_t ib = round(op2);
                    if (abs(op2 - ib) <= (scalar_t)1e-3) {
                        op2 = ib;
                    } else {
                        op1 = abs(op1);
                    }
                }
                val = pow(op1, op2);
            }
            else if (token == op_mod) {
                if (abs(op2) < (scalar_t)1e-9) val = 0; else val = fmod(op1, op2);
            }
        }
        // Unary
        else {
            if (sp < 1) break;
            int left_pc = idx_stack[--sp];
            scalar_t op1 = val_tape[left_pc];
            
            args_left[pc] = left_pc;
            
            if (token == op_sin) val = sin(op1);
            else if (token == op_cos) val = cos(op1);
            else if (token == op_tan) val = tan(op1);
            else if (token == op_log) val = log(abs(op1) + (scalar_t)1e-9);
            else if (token == op_exp) {
                scalar_t cl = op1; 
                if(cl < -80.0) cl = -80.0; if(cl > 80.0) cl = 80.0;
                val = exp(cl);
            }
            else if (token == op_sqrt) val = sqrt(abs(op1));
            else if (token == op_abs) val = abs(op1);
            else if (token == op_neg) val = -op1;
            // Gamma excluded for simplicity in gradients or handled basically
        }
        
        val_tape[pc] = val;
        idx_stack[sp++] = pc;
        root_pc = pc;
    }
    
    if (root_pc == -1 || sp != 1) return; // Invalid formula
    
    // --- REVERSE PASS ---
    // Start by seeding the root node with the external gradient (dL/dY)
    adj_tape[root_pc] = grad_output[b_idx * D + d_idx];
    
    for (int pc = root_pc; pc >= 0; --pc) {
        scalar_t adj = adj_tape[pc];
        if (adj == (scalar_t)0.0) continue;
        
        int64_t token = (int64_t)my_prog[pc];
        int l_pc = args_left[pc];
        int r_pc = args_right[pc];
        
        // If it's a Constant, commit gradient directly and safely
        if (const_index[pc] != -1) {
            int c_id = const_index[pc];
            // Atomic Add is needed because multiple threads (D samples)
            // will update the same Constant gradient!
            atomicAdd(&grad_constants[b_idx * K + c_id], adj);
            continue;
        }
        
        if (l_pc == -1) continue; // Was Terminal, nothing to backprop
        
        scalar_t l_val = val_tape[l_pc];
        
        if (r_pc != -1) {
            // Binary Ops Backward
            scalar_t r_val = val_tape[r_pc];
            
            if (token == op_add) {
                adj_tape[l_pc] += adj;
                adj_tape[r_pc] += adj;
            }
            else if (token == op_sub) {
                adj_tape[l_pc] += adj;
                adj_tape[r_pc] -= adj;
            }
            else if (token == op_mul) {
                adj_tape[l_pc] += r_val * adj;
                adj_tape[r_pc] += l_val * adj;
            }
            else if (token == op_div) {
                if (abs(r_val) >= (scalar_t)1e-9) {
                    adj_tape[l_pc] += (1.0 / r_val) * adj;
                    adj_tape[r_pc] -= (l_val / (r_val * r_val)) * adj;
                }
            }
            else if (token == op_pow) {
                // d(a^b)/da = b * a^(b-1)
                // d(a^b)/db = a^b * ln(a)
                scalar_t out_val = val_tape[pc];
                if (l_val > 0) {
                    adj_tape[l_pc] += (r_val * pow(l_val, r_val - 1.0)) * adj;
                    adj_tape[r_pc] += (out_val * log(l_val)) * adj;
                }
            }
        } else {
            // Unary Ops Backward
            if (token == op_sin) {
                adj_tape[l_pc] += cos(l_val) * adj;
            }
            else if (token == op_cos) {
                adj_tape[l_pc] += -sin(l_val) * adj;
            }
            else if (token == op_log) {
                if (abs(l_val) > (scalar_t)1e-9) {
                    adj_tape[l_pc] += (1.0 / l_val) * adj;
                }
            }
            else if (token == op_exp) {
                adj_tape[l_pc] += val_tape[pc] * adj; // e^x * adj
            }
            else if (token == op_sqrt) {
                if (l_val > (scalar_t)1e-9) {
                    adj_tape[l_pc] += (0.5 / sqrt(l_val)) * adj;
                }
            }
            else if (token == op_neg) {
                adj_tape[l_pc] += -adj;
            }
        }
    }
}

// ============================================================================
// Host Launcher
// ============================================================================
void launch_rpn_backward(
    const torch::Tensor& population,
    const torch::Tensor& x,
    const torch::Tensor& constants,
    const torch::Tensor& grad_output,
    torch::Tensor& grad_constants,
    int PAD_ID, int id_x_start, 
    int id_C, int id_pi, int id_e,
    int id_0, int id_1, int id_2, int id_3, int id_4, int id_5, int id_6, int id_10,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan,
    double pi_val, double e_val
) {
    int B = population.size(0);
    int L = population.size(1);
    int Vars = x.size(0);
    int D = x.size(1);
    int K = constants.size(1);
    
    int total_threads = B * D;
    int threads_per_block = 256;
    int num_blocks = (total_threads + threads_per_block - 1) / threads_per_block;
    
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "rpn_eval_backward_kernel", ([&] {
        rpn_eval_backward_kernel<scalar_t><<<num_blocks, threads_per_block>>>(
            population.data_ptr<unsigned char>(),
            x.data_ptr<scalar_t>(),
            constants.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            grad_constants.data_ptr<scalar_t>(),
            B, D, L, K, Vars,
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
            pi_val, e_val
        );
    }));
}
