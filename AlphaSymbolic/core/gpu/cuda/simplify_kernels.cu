/*
 * GPU-Native Symbolic Simplification Kernels
 * 
 * Replaces ~900 PyTorch kernel launches + ~300 CPU-GPU syncs per simplify_batch call
 * with a single fused CUDA kernel. Each thread processes one formula independently.
 *
 * Rules implemented (full parity with gpu_simplifier.py):
 *   1. Identity:      x+0, x-0, x*1, x/1, x^1 -> x; 0+x, 1*x -> x; x^0 -> 1; 1^x -> 1
 *   2. Zero:          x*0, 0*x -> 0; 0/x -> 0; 0-x -> neg(x)
 *   3. Chain:         neg(neg(x)) -> x; exp(log(x)) -> x; log(exp(x)) -> x
 *   4. Trig inverse:  acos(cos(x)), asin(sin(x)), atan(tan(x)) -> x
 *   5. Constant args: sin(0)=0, cos(0)=1, exp(0)=1, log(1)=0, etc.
 *   6. Self-cancel:   x-x -> 0, x/x -> 1; x+neg(x) -> 0
 *   7. Term consol:   x+x -> 2*x, x*x -> x^2; a*x+b*x -> (a+b)*x
 *   8. Modulo:        x%x -> 0
 *   9. Commutative normalization (scoring: const<var<complex, tiebreak by ID)
 *  10. Constant folding: sin(c), c1+c2, etc. -> nearest literal
 *  11. Associative:   x+(x+y) -> 2*x+y (single-token x)
 *  12. Compact: shift non-PAD tokens left
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

// Maximum formula length (tokens)
#define MAX_FORMULA_LEN 64
// Max number of literals for constant folding mapping
#define MAX_LITERALS 64

// PAD_ID is always 0 in this project
#define PAD_ID_CONST 0

/*
 * Compute subtree start index for position `end_idx` in a formula.
 * Uses backward scan with arity-based balance tracking.
 * Fully thread-local, no shared memory needed.
 */
__device__ __forceinline__ int subtree_start(
    const int64_t* formula, const int* arities, int max_id, int end_idx, int L
) {
    if (end_idx < 0 || end_idx >= L) return -1;
    int64_t tok = formula[end_idx];
    if (tok == PAD_ID_CONST || tok < 0 || tok >= max_id) return end_idx;
    
    int balance = 1;
    for (int k = end_idx; k >= 0; k--) {
        int64_t t = formula[k];
        if (t == PAD_ID_CONST || t < 0 || t >= max_id) {
            balance -= 1;
        } else {
            int ar = arities[min((int)t, max_id - 1)];
            balance += ar - 1;  // terminal: -1+1=0 consume, unary: 0, binary: +1
        }
        if (balance == 0) return k;
    }
    return 0;  // fallback
}

/*
 * Compact a formula in-place: move all non-PAD tokens to the left.
 * Operates in local registers (formula array).
 */
__device__ void compact_formula(int64_t* f, int L) {
    int write = 0;
    for (int read = 0; read < L; read++) {
        if (f[read] != PAD_ID_CONST) {
            f[write++] = f[read];
        }
    }
    for (int i = write; i < L; i++) {
        f[i] = PAD_ID_CONST;
    }
}

/*
 * Main simplification kernel.
 * One thread per formula. Formula loaded into registers, simplified in-place,
 * then written back. Multiple passes internally.
 *
 * Parameters:
 *   pop:      [B, L] int64 population tensor (modified in-place)
 *   arities:  [VocabSize] int32 arity table  
 *   val_table:[VocabSize] float32 — numerical value per token (NaN = not a literal)
 *   literal_ids: [n_literals] int64 — token IDs of all literal constants
 *   literal_vals: [n_literals] float32 — numerical values of those literals
 *   n_literals: number of literals for constant folding mapping
 *   max_id:   vocabulary size (for bounds checking)
 *   B, L:     population dimensions
 *   max_passes: number of simplification passes
 *   
 *   Op-code IDs (passed as kernel params to avoid hardcoding):
 */
__global__ void simplify_batch_kernel(
    int64_t* __restrict__ pop,
    const int* __restrict__ arities,
    const float* __restrict__ val_table,
    const int64_t* __restrict__ literal_ids,
    const float* __restrict__ literal_vals,
    int n_literals,
    int max_id,
    int B, int L,
    int max_passes,
    // Op-codes
    int op_plus, int op_minus, int op_mult, int op_div,
    int op_neg, int op_mod, int op_pow,
    int op_sin, int op_cos, int op_tan,
    int op_asin, int op_acos, int op_atan,
    int op_log, int op_exp, int op_sqrt, int op_abs,
    int op_gamma, int op_lgamma,
    int op_floor, int op_ceil, int op_sign,
    int id_0, int id_1, int id_2
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    
    // Load formula into registers
    int64_t f[MAX_FORMULA_LEN];
    int64_t* row = pop + (int64_t)b * L;
    int actual_L = min(L, MAX_FORMULA_LEN);
    
    for (int i = 0; i < actual_L; i++) f[i] = row[i];
    for (int i = actual_L; i < MAX_FORMULA_LEN; i++) f[i] = PAD_ID_CONST;
    
    // Cache literal values in local memory for constant folding
    int64_t loc_lit_ids[MAX_LITERALS];
    float loc_lit_vals[MAX_LITERALS];
    int n_lits = min(n_literals, MAX_LITERALS);
    for (int i = 0; i < n_lits; i++) {
        loc_lit_ids[i] = literal_ids[i];
        loc_lit_vals[i] = literal_vals[i];
    }
    
    for (int pass = 0; pass < max_passes; pass++) {
        bool changed = false;
        
        // === PHASE 1: Chain reductions, advanced unary, & constant folding ===
        for (int j = 1; j < actual_L; j++) {
            int64_t tok = f[j];
            if (tok == PAD_ID_CONST) continue;
            
            int64_t prev = f[j-1];
            
            // --- neg(neg(x)) -> x ---
            if (tok == op_neg && prev == op_neg) {
                f[j] = PAD_ID_CONST; f[j-1] = PAD_ID_CONST;
                changed = true; continue;
            }
            
            // --- exp(log(x)) -> x ---
            if (tok == op_exp && prev == op_log) {
                f[j] = PAD_ID_CONST; f[j-1] = PAD_ID_CONST;
                changed = true; continue;
            }
            
            // --- log(exp(x)) -> x ---
            if (tok == op_log && prev == op_exp) {
                f[j] = PAD_ID_CONST; f[j-1] = PAD_ID_CONST;
                changed = true; continue;
            }
            
            // --- acos(cos(x)) -> x ---
            if (tok == op_acos && prev == op_cos) {
                f[j] = PAD_ID_CONST; f[j-1] = PAD_ID_CONST;
                changed = true; continue;
            }
            
            // --- asin(sin(x)) -> x ---
            if (tok == op_asin && prev == op_sin) {
                f[j] = PAD_ID_CONST; f[j-1] = PAD_ID_CONST;
                changed = true; continue;
            }
            
            // --- atan(tan(x)) -> x ---
            if (tok == op_atan && prev == op_tan) {
                f[j] = PAD_ID_CONST; f[j-1] = PAD_ID_CONST;
                changed = true; continue;
            }
            
            // --- sqrt(x^2) -> abs(x) ---
            if (j >= 2 && tok == op_sqrt && prev == op_pow && f[j-2] == id_2 && op_abs != -1) {
                f[j] = op_abs; f[j-1] = PAD_ID_CONST; f[j-2] = PAD_ID_CONST;
                changed = true; continue;
            }
            
            // --- Constant-argument unary identities (known results) ---
            if (tok > 0 && tok < max_id && arities[tok] == 1 && prev != PAD_ID_CONST) {
                bool is_zero_arg = (prev == id_0);
                bool is_one_arg = (prev == id_1);
                bool is_two_arg = (prev == id_2);
                
                // -> 0: sin(0), tan(0), abs(0), log(1), lgamma(1), lgamma(2)
                bool fold_to_zero = false;
                if (is_zero_arg && (tok == op_sin || tok == op_tan || tok == op_abs)) fold_to_zero = true;
                if (is_one_arg && (tok == op_log || tok == op_lgamma)) fold_to_zero = true;
                if (is_two_arg && tok == op_lgamma) fold_to_zero = true;
                
                if (fold_to_zero && id_0 != -1) {
                    f[j-1] = id_0; f[j] = PAD_ID_CONST;
                    changed = true; continue;
                }
                
                // -> 1: cos(0), exp(0), gamma(1), gamma(2)
                bool fold_to_one = false;
                if (is_zero_arg && (tok == op_cos || tok == op_exp)) fold_to_one = true;
                if ((is_one_arg || is_two_arg) && tok == op_gamma) fold_to_one = true;
                
                if (fold_to_one && id_1 != -1) {
                    f[j-1] = id_1; f[j] = PAD_ID_CONST;
                    changed = true; continue;
                }
                
                // --- General Unary Constant Folding ---
                // If arg is a literal, compute f(arg) and map to nearest literal
                if (n_lits > 0 && prev > 0 && prev < max_id) {
                    float arg_val = val_table[prev];
                    if (!isnan(arg_val)) {
                        float res = nanf("");
                        if (tok == op_sin)    res = sinf(arg_val);
                        else if (tok == op_cos)    res = cosf(arg_val);
                        else if (tok == op_tan)    res = tanf(arg_val);
                        else if (tok == op_log)    res = (arg_val > 0) ? logf(arg_val) : nanf("");
                        else if (tok == op_exp)    res = (arg_val < 80.0f) ? expf(arg_val) : nanf("");
                        else if (tok == op_sqrt)   res = (arg_val >= 0) ? sqrtf(arg_val) : nanf("");
                        else if (tok == op_abs)    res = fabsf(arg_val);
                        else if (tok == op_neg)    res = -arg_val;
                        else if (tok == op_floor)  res = floorf(arg_val);
                        else if (tok == op_ceil)   res = ceilf(arg_val);
                        else if (tok == op_sign)   res = (arg_val > 0) ? 1.0f : ((arg_val < 0) ? -1.0f : 0.0f);
                        else if (tok == op_asin)   res = (fabsf(arg_val) <= 1.0f) ? asinf(arg_val) : nanf("");
                        else if (tok == op_acos)   res = (fabsf(arg_val) <= 1.0f) ? acosf(arg_val) : nanf("");
                        else if (tok == op_atan)   res = atanf(arg_val);
                        else if (tok == op_gamma)  res = (arg_val > 0 && arg_val < 20.0f) ? expf(lgammaf(arg_val)) : nanf("");
                        else if (tok == op_lgamma) res = (arg_val > 0) ? lgammaf(arg_val) : nanf("");
                        
                        if (!isnan(res) && !isinf(res)) {
                            // Find nearest literal
                            float best_dist = 1e30f;
                            int best_idx = -1;
                            for (int li = 0; li < n_lits; li++) {
                                float d = fabsf(res - loc_lit_vals[li]);
                                if (d < best_dist) { best_dist = d; best_idx = li; }
                            }
                            if (best_idx >= 0 && best_dist < 1e-5f) {
                                f[j-1] = loc_lit_ids[best_idx];
                                f[j] = PAD_ID_CONST;
                                changed = true; continue;
                            }
                        }
                    }
                }
            }
            
            // --- x + neg(x) -> 0 ---
            // Pattern at position j: f[j] = +, f[j-1] = neg, subtree of neg = subtree before it
            if (j >= 3 && tok == op_plus && f[j-1] == op_neg) {
                // neg's argument is the subtree ending at j-2
                int s_neg_arg = subtree_start(f, arities, max_id, j-2, actual_L);
                // The first operand of + starts before neg's subtree
                int e_first = s_neg_arg - 1;
                if (e_first >= 0 && s_neg_arg >= 0) {
                    int s_first = subtree_start(f, arities, max_id, e_first, actual_L);
                    if (s_first >= 0) {
                        // Check if first operand (s_first..e_first) equals neg's arg (s_neg_arg..j-2)
                        int len1 = e_first - s_first + 1;
                        int len2 = (j-2) - s_neg_arg + 1;
                        if (len1 == len2 && len1 > 0) {
                            bool match = true;
                            for (int k = 0; k < len1; k++) {
                                if (f[s_first + k] != f[s_neg_arg + k]) { match = false; break; }
                            }
                            if (match && id_0 != -1) {
                                f[s_first] = id_0;
                                for (int k = s_first + 1; k <= j; k++) f[k] = PAD_ID_CONST;
                                changed = true; continue;
                            }
                        }
                    }
                }
            }
        }
        
        // Compact after chain reductions
        compact_formula(f, actual_L);
        
        // === PHASE 2: Binary identity/zero/consolidation rules ===
        for (int j = 2; j < actual_L; j++) {
            int64_t op = f[j];
            if (op == PAD_ID_CONST) continue;
            if (op < 0 || op >= max_id) continue;
            int ar = arities[op];
            if (ar != 2) continue;
            
            // For binary ops, find subtree boundaries
            int s2 = subtree_start(f, arities, max_id, j-1, actual_L);
            int s1 = subtree_start(f, arities, max_id, s2-1, actual_L);
            if (s1 < 0 || s2 < 0) continue;
            
            int len1 = s2 - s1;    // length of arg1
            int len2 = j - s2;     // length of arg2
            int64_t arg2_single = (s2 == j-1) ? f[j-1] : -999;  // arg2 is single token?
            int64_t arg1_single = (s1 == s2-1) ? f[s2-1] : -999;  // arg1 is single token?
            
            bool arg2_is_zero = (arg2_single == id_0);
            bool arg2_is_one = (arg2_single == id_1);
            bool arg1_is_zero = (arg1_single == id_0);
            bool arg1_is_one = (arg1_single == id_1);
            bool is_pow = (op == op_pow);
            
            // --- x + 0, x - 0 -> x (remove arg2 + op) ---
            if ((op == op_plus || op == op_minus) && arg2_is_zero) {
                f[j-1] = PAD_ID_CONST; f[j] = PAD_ID_CONST;
                changed = true; continue;
            }
            
            // --- x * 1, x / 1, x ^ 1 -> x ---
            if ((op == op_mult || op == op_div || is_pow) && arg2_is_one) {
                f[j-1] = PAD_ID_CONST; f[j] = PAD_ID_CONST;
                changed = true; continue;
            }
            
            // --- x ^ 0 -> 1 (entire subtree replaced by 1) ---
            if (is_pow && arg2_is_zero && id_1 != -1) {
                for (int k = s1; k <= j; k++) f[k] = PAD_ID_CONST;
                f[s1] = id_1;
                changed = true; continue;
            }
            
            // --- 1 ^ x -> 1 (entire subtree replaced by 1) ---
            if (is_pow && arg1_is_one && id_1 != -1) {
                for (int k = s1; k <= j; k++) f[k] = PAD_ID_CONST;
                f[s1] = id_1;
                changed = true; continue;
            }
            
            // --- 0 + x -> x (remove arg1 + op, keep arg2) ---
            if (op == op_plus && arg1_is_zero) {
                f[s2-1] = PAD_ID_CONST; f[j] = PAD_ID_CONST;
                changed = true; continue;
            }
            
            // --- 1 * x -> x ---
            if (op == op_mult && arg1_is_one) {
                f[s2-1] = PAD_ID_CONST; f[j] = PAD_ID_CONST;
                changed = true; continue;
            }
            
            // --- x * 0, 0 * x -> 0 ---
            if (op == op_mult && (arg2_is_zero || arg1_is_zero) && id_0 != -1) {
                for (int k = s1; k <= j; k++) f[k] = PAD_ID_CONST;
                f[s1] = id_0;
                changed = true; continue;
            }
            
            // --- 0 / x -> 0 ---
            if (op == op_div && arg1_is_zero && id_0 != -1) {
                for (int k = s1; k <= j; k++) f[k] = PAD_ID_CONST;
                f[s1] = id_0;
                changed = true; continue;
            }
            
            // --- 0 - x -> neg(x) ---
            if (op == op_minus && arg1_is_zero && op_neg != -1) {
                f[s2-1] = PAD_ID_CONST;  // remove the 0
                f[j] = op_neg;           // change - to neg
                changed = true; continue;
            }
            
            // --- Self-cancellation (subtree comparison) ---
            if ((op == op_minus || op == op_div) && len1 == len2 && len1 > 0) {
                bool match = true;
                for (int k = 0; k < len1; k++) {
                    if (f[s1 + k] != f[s2 + k]) { match = false; break; }
                }
                if (match) {
                    if (op == op_minus && id_0 != -1) {
                        f[s1] = id_0;
                        for (int k = s1+1; k <= j; k++) f[k] = PAD_ID_CONST;
                        changed = true; continue;
                    }
                    if (op == op_div && id_1 != -1) {
                        f[s1] = id_1;
                        for (int k = s1+1; k <= j; k++) f[k] = PAD_ID_CONST;
                        changed = true; continue;
                    }
                }
            }
            
            // --- Term consolidation: x + x -> 2*x, x * x -> x^2 (subtree comparison) ---
            if ((op == op_plus || op == op_mult) && len1 == len2 && len1 > 0 && len1 <= 3) {
                bool match = true;
                for (int k = 0; k < len1; k++) {
                    if (f[s1 + k] != f[s2 + k]) { match = false; break; }
                }
                if (match) {
                    if (op == op_plus && id_2 != -1 && len1 + 2 <= len1 + len2 + 1) {
                        // subtree + subtree -> [2, subtree, *]
                        // We need: id_2, then the subtree, then op_mult
                        // Space available: s1 to j (= len1 + len2 + 1 positions)
                        // Need: 1 + len1 + 1 = len1 + 2 positions
                        int new_len = 1 + len1 + 1;
                        int total_space = j - s1 + 1;
                        if (new_len <= total_space) {
                            int w = s1;
                            f[w++] = id_2;
                            for (int k = 0; k < len1; k++) f[w++] = f[s2 + k]; // copy subtree (from arg2, still intact)
                            f[w++] = op_mult;
                            for (int k = w; k <= j; k++) f[k] = PAD_ID_CONST;
                            changed = true; continue;
                        }
                    }
                    if (op == op_mult && op_pow != -1 && id_2 != -1 && len1 + 2 <= len1 + len2 + 1) {
                        // subtree * subtree -> [subtree, 2, ^]
                        int new_len = len1 + 1 + 1;
                        int total_space = j - s1 + 1;
                        if (new_len <= total_space) {
                            // arg1 is already in place at s1..s2-1
                            int w = s1 + len1;
                            f[w++] = id_2;
                            f[w++] = op_pow;
                            for (int k = w; k <= j; k++) f[k] = PAD_ID_CONST;
                            changed = true; continue;
                        }
                    }
                }
            }
            
            // --- Modulo: x % x -> 0 (subtree comparison) ---
            if (op == op_mod && len1 == len2 && len1 > 0) {
                bool match = true;
                for (int k = 0; k < len1; k++) {
                    if (f[s1 + k] != f[s2 + k]) { match = false; break; }
                }
                if (match && id_0 != -1) {
                    f[s1] = id_0;
                    for (int k = s1+1; k <= j; k++) f[k] = PAD_ID_CONST;
                    changed = true; continue;
                }
            }
            
            // --- Commutative normalization with scoring: const(0) < var(1) < complex(2) ---
            if (op == op_plus || op == op_mult) {
                bool is_leaf1 = (len1 == 1);
                bool is_leaf2 = (len2 == 1);

                // Determine if single-token args are constants (exist in val_table as non-NaN)
                bool is_const1 = false, is_const2 = false;
                if (is_leaf1 && arg1_single > 0 && arg1_single < max_id) {
                    is_const1 = !isnan(val_table[arg1_single]);
                }
                if (is_leaf2 && arg2_single > 0 && arg2_single < max_id) {
                    is_const2 = !isnan(val_table[arg2_single]);
                }
                
                int score1 = is_const1 ? 0 : (is_leaf1 ? 1 : 2);
                int score2 = is_const2 ? 0 : (is_leaf2 ? 1 : 2);
                
                bool should_swap = false;
                if (score2 < score1) should_swap = true;
                // Tiebreaker for leaves: lower ID first
                if (score1 == score2 && score1 < 2 && arg2_single != -999 && arg1_single != -999 && arg2_single < arg1_single) should_swap = true;
                // Tiebreaker for complex: shorter first
                if (score1 == 2 && score2 == 2 && len2 < len1) should_swap = true;
                
                if (should_swap) {
                    if (is_leaf1 && is_leaf2) {
                        // Simple swap of single tokens
                        f[s2-1] = arg2_single;
                        f[j-1] = arg1_single;
                    } else {
                        // Variable-length swap using temp buffer
                        int64_t tmp[MAX_FORMULA_LEN];
                        // Copy arg1
                        for (int k = 0; k < len1; k++) tmp[k] = f[s1 + k];
                        // Copy arg2 to arg1 position
                        for (int k = 0; k < len2; k++) f[s1 + k] = f[s2 + k];
                        // Copy arg1 from tmp to arg2 position
                        for (int k = 0; k < len1; k++) f[s1 + len2 + k] = tmp[k];
                    }
                    // Don't set changed=true for normalization
                }
            }
            
            // --- Factoring: [a, x, *, b, x, *, +] -> [a, b, +, x, *] ---
            if (op == op_plus && j >= 6) {
                int64_t v0 = f[j], v1 = f[j-1], v2 = f[j-2], v3 = f[j-3];
                int64_t v4 = f[j-4], v5 = f[j-5], v6 = f[j-6];
                
                if (v0 == op_plus && v1 == op_mult && v4 == op_mult) {
                    // Case 1: [a, x, *, b, x, *, +] where x=v5=v2
                    if (v5 == v2 && v5 != PAD_ID_CONST &&
                        v5 >= 0 && v5 < max_id && arities[v5] == 0 &&
                        v6 >= 0 && v6 < max_id && arities[v6] == 0 &&
                        v3 >= 0 && v3 < max_id && arities[v3] == 0) {
                        f[j-6] = v6;         // a
                        f[j-5] = v3;         // b
                        f[j-4] = op_plus;    // +
                        f[j-3] = v5;         // x
                        f[j-2] = op_mult;    // *
                        f[j-1] = PAD_ID_CONST;
                        f[j]   = PAD_ID_CONST;
                        changed = true; continue;
                    }
                    // Case 2: [x, a, *, x, b, *, +] where x=v6=v3
                    if (v6 == v3 && v6 != PAD_ID_CONST &&
                        v6 >= 0 && v6 < max_id && arities[v6] == 0 &&
                        v5 >= 0 && v5 < max_id && arities[v5] == 0 &&
                        v2 >= 0 && v2 < max_id && arities[v2] == 0) {
                        f[j-6] = v5;         // a
                        f[j-5] = v2;         // b
                        f[j-4] = op_plus;    // +
                        f[j-3] = v6;         // x
                        f[j-2] = op_mult;    // *
                        f[j-1] = PAD_ID_CONST;
                        f[j]   = PAD_ID_CONST;
                        changed = true; continue;
                    }
                }
            }
            
            // --- Associative: x + (x + y) -> 2*x + y (single-token x only) ---
            if (op == op_plus && id_2 != -1 && j >= 4) {
                // Pattern: [x, x, y, +, +] where arg2 ends at j-1 and is (x+y)
                // arg2 = subtree(j-1), must root at op_plus at j-1
                if (f[j-1] == op_plus) {
                    // Inner + at j-1: inner right = subtree(j-2), inner left = subtree before that
                    int s_inner_r = subtree_start(f, arities, max_id, j-2, actual_L);
                    if (s_inner_r >= 1) {
                        int e_inner_l = s_inner_r - 1;
                        int s_inner_l = subtree_start(f, arities, max_id, e_inner_l, actual_L);
                        if (s_inner_l >= 1) {
                            // arg1 of outer + is subtree ending at s_inner_l - 1
                            int e_outer1 = s_inner_l - 1;
                            int s_outer1 = subtree_start(f, arities, max_id, e_outer1, actual_L);
                            if (s_outer1 >= 0) {
                                // Check x == inner_left (both single token)
                                bool outer1_single = (s_outer1 == e_outer1);
                                bool inner_l_single = (s_inner_l == e_inner_l);
                                if (outer1_single && inner_l_single && f[s_outer1] == f[s_inner_l] && f[s_outer1] != PAD_ID_CONST) {
                                    // x + (x + y) -> 2*x + y = [2, x, *, y_subtree, +]
                                    int y_len = (j-2) - s_inner_r + 1;
                                    int new_len = 3 + y_len + 1; // 2, x, *, y..., +
                                    int total_space = j - s_outer1 + 1;
                                    if (new_len <= total_space) {
                                        int64_t x_tok = f[s_outer1];
                                        // Save y subtree
                                        int64_t y_buf[MAX_FORMULA_LEN];
                                        for (int k = 0; k < y_len; k++) y_buf[k] = f[s_inner_r + k];
                                        
                                        int w = s_outer1;
                                        f[w++] = id_2;
                                        f[w++] = x_tok;
                                        f[w++] = op_mult;
                                        for (int k = 0; k < y_len; k++) f[w++] = y_buf[k];
                                        f[w++] = op_plus;
                                        for (int k = w; k <= j; k++) f[k] = PAD_ID_CONST;
                                        changed = true; continue;
                                    }
                                }
                                // Check y == outer_x: x + (y + x) -> 2*x + y
                                int inner_r_len = (j-2) - s_inner_r + 1;
                                bool outer1_is_single = (s_outer1 == e_outer1);
                                bool inner_r_single = (inner_r_len == 1);
                                if (outer1_is_single && inner_r_single && f[s_outer1] == f[s_inner_r] && f[s_outer1] != PAD_ID_CONST) {
                                    int y_len = e_inner_l - s_inner_l + 1;
                                    int new_len = 3 + y_len + 1;
                                    int total_space = j - s_outer1 + 1;
                                    if (new_len <= total_space) {
                                        int64_t x_tok = f[s_outer1];
                                        int64_t y_buf[MAX_FORMULA_LEN];
                                        for (int k = 0; k < y_len; k++) y_buf[k] = f[s_inner_l + k];
                                        
                                        int w = s_outer1;
                                        f[w++] = id_2;
                                        f[w++] = x_tok;
                                        f[w++] = op_mult;
                                        for (int k = 0; k < y_len; k++) f[w++] = y_buf[k];
                                        f[w++] = op_plus;
                                        for (int k = w; k <= j; k++) f[k] = PAD_ID_CONST;
                                        changed = true; continue;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // --- Binary Constant Folding ---
            if (n_lits > 0 && arg1_single != -999 && arg2_single != -999 &&
                arg1_single > 0 && arg1_single < max_id && arg2_single > 0 && arg2_single < max_id) {
                float v1 = val_table[arg1_single];
                float v2 = val_table[arg2_single];
                if (!isnan(v1) && !isnan(v2)) {
                    float res = nanf("");
                    if (op == op_plus) res = v1 + v2;
                    else if (op == op_minus) res = v1 - v2;
                    else if (op == op_mult) res = v1 * v2;
                    else if (op == op_div && fabsf(v2) > 1e-9f) res = v1 / v2;
                    else if (op == op_pow) res = powf(v1, v2);
                    else if (op == op_mod && fabsf(v2) > 1e-9f) res = fmodf(v1, v2);
                    
                    if (!isnan(res) && !isinf(res)) {
                        float best_dist = 1e30f;
                        int best_idx = -1;
                        for (int li = 0; li < n_lits; li++) {
                            float d = fabsf(res - loc_lit_vals[li]);
                            if (d < best_dist) { best_dist = d; best_idx = li; }
                        }
                        if (best_idx >= 0 && best_dist < 1e-5f) {
                            f[s1] = loc_lit_ids[best_idx];
                            for (int k = s1+1; k <= j; k++) f[k] = PAD_ID_CONST;
                            changed = true; continue;
                        }
                    }
                }
            }
        }
        
        // Compact after all rules
        compact_formula(f, actual_L);
        
        if (!changed) break;
    }
    
    // Write back to global memory
    for (int i = 0; i < actual_L; i++) row[i] = f[i];
}


/*
 * Precompute subtree starts for every position in every formula.
 * One thread per formula. Computes subtree_len via left-to-right DP,
 * then starts = position - subtree_len + 1.
 * 
 * Output: out_starts[b][j] = start index of subtree ending at position j.
 *         PAD positions get -1.
 */
__global__ void precompute_subtree_starts_kernel(
    const int64_t* __restrict__ pop,     // [B, L]
    const int* __restrict__ arities,     // [VocabSize]
    int64_t* __restrict__ out_starts,    // [B, L]
    int max_id, int B, int L
) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= B) return;
    
    const int64_t* row = pop + (int64_t)b * L;
    int64_t* out_row = out_starts + (int64_t)b * L;
    
    int actual_L = min(L, MAX_FORMULA_LEN);
    int sub_len[MAX_FORMULA_LEN];
    
    // Forward DP: compute subtree length for each position
    for (int j = 0; j < actual_L; j++) {
        int64_t tok = row[j];
        if (tok == PAD_ID_CONST || tok < 0 || tok >= max_id) {
            sub_len[j] = 0;
            out_row[j] = -1;
            continue;
        }
        
        int ar = arities[min((int)tok, max_id - 1)];
        
        if (ar == 0) {
            // Terminal
            sub_len[j] = 1;
        } else if (ar == 1 && j >= 1) {
            // Unary: child ends at j-1
            sub_len[j] = 1 + sub_len[j-1];
        } else if (ar == 2 && j >= 2) {
            // Binary: right child ends at j-1, left child ends at j-1-sub_len[j-1]
            int right_len = sub_len[j-1];
            int left_idx = j - 1 - right_len;
            int left_len = (left_idx >= 0) ? sub_len[left_idx] : 0;
            sub_len[j] = 1 + right_len + left_len;
        } else {
            sub_len[j] = 1;  // fallback
        }
        
        out_row[j] = j - sub_len[j] + 1;
    }
    
    // Fill remaining
    for (int j = actual_L; j < L; j++) {
        out_row[j] = -1;
    }
}


// ======================== C++ Launch Wrappers ========================

void launch_simplify_batch(
    torch::Tensor& population,
    const torch::Tensor& arities,
    const torch::Tensor& val_table,
    const torch::Tensor& literal_ids,
    const torch::Tensor& literal_vals,
    int max_passes,
    int op_plus, int op_minus, int op_mult, int op_div,
    int op_neg, int op_mod, int op_pow,
    int op_sin, int op_cos, int op_tan,
    int op_asin, int op_acos, int op_atan,
    int op_log, int op_exp, int op_sqrt, int op_abs,
    int op_gamma, int op_lgamma,
    int op_floor, int op_ceil, int op_sign,
    int id_0, int id_1, int id_2
) {
    CHECK_INPUT(population);
    CHECK_INPUT(arities);
    CHECK_INPUT(val_table);
    CHECK_INPUT(literal_ids);
    CHECK_INPUT(literal_vals);
    
    int B = population.size(0);
    int L = population.size(1);
    int max_id = arities.size(0);
    int n_literals = literal_ids.size(0);
    
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    
    simplify_batch_kernel<<<blocks, threads>>>(
        population.data_ptr<int64_t>(),
        arities.data_ptr<int>(),
        val_table.data_ptr<float>(),
        literal_ids.data_ptr<int64_t>(),
        literal_vals.data_ptr<float>(),
        n_literals,
        max_id, B, L, max_passes,
        op_plus, op_minus, op_mult, op_div,
        op_neg, op_mod, op_pow,
        op_sin, op_cos, op_tan,
        op_asin, op_acos, op_atan,
        op_log, op_exp, op_sqrt, op_abs,
        op_gamma, op_lgamma,
        op_floor, op_ceil, op_sign,
        id_0, id_1, id_2
    );
}

void launch_precompute_subtree_starts(
    const torch::Tensor& population,
    const torch::Tensor& arities,
    torch::Tensor& out_starts
) {
    CHECK_INPUT(population);
    CHECK_INPUT(arities);
    CHECK_INPUT(out_starts);
    
    int B = population.size(0);
    int L = population.size(1);
    int max_id = arities.size(0);
    
    int threads = 256;
    int blocks = (B + threads - 1) / threads;
    
    precompute_subtree_starts_kernel<<<blocks, threads>>>(
        population.data_ptr<int64_t>(),
        arities.data_ptr<int>(),
        out_starts.data_ptr<int64_t>(),
        max_id, B, L
    );
}
