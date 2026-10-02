/*
 * Levenberg-Marquardt constant optimisation.
 *
 * One warp per individual. The program is decoded once (constant tokens keep
 * their slot), the Jacobian of the residuals with respect to the constants is
 * computed in reverse mode on a per-sample value tape (one forward and one
 * backward sweep give the whole row; programs longer than LM_TAPE use one
 * forward-mode dual pass per constant). Derivatives are exact for the same
 * strict/protected semantics used by the fused evaluator. The damped
 * normal equations
 *
 *     (J^T J + lambda * diag(J^T J)) delta = -J^T r
 *
 * are solved by one lane with a double-precision Cholesky factorisation.
 *
 * Each sample's Jacobian row is written to shared memory; the 32 rows of a
 * round are then reduced into J^T J and J^T r by the lanes, each lane owning a
 * few entries of the (symmetric) system, so no lane keeps a full matrix in
 * registers.
 *
 * With SCALED the residual is a + b*f - y and (a, b) are optimised together
 * with the constants. After every accepted step (a, b) are reset to the least
 * squares fit of the current program, so the reported error is exactly the
 * linear-scaling objective of the fused evaluator.
 *
 * Compared with the fused PSO (30 particles x 40 steps = 1200 evaluations per
 * individual), an iteration costs one forward and one backward sweep plus one
 * or two trial evaluations.
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <type_traits>

#include "eval_core.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

#define LM_MAX_L 256
#define LM_MAX_K 32                 // constant slots per individual (storage)
#define LM_MAX_C 10                 // constant slots that can be optimised
#define LM_MAX_P (LM_MAX_C + 2)     // + (a, b) with linear scaling
#define LM_N_ENTRIES (LM_MAX_P * (LM_MAX_P + 1) / 2 + LM_MAX_P)
#define LM_ENTRIES_PER_LANE ((LM_N_ENTRIES + 31) / 32)
#define LM_WARPS_PER_BLOCK 4
#define LM_MAX_TRIALS 6
// Programs up to this length use the reverse-mode tape (two passes per sample
// for the whole Jacobian row); longer ones fall back to one forward-mode dual
// pass per constant.
#define LM_TAPE 128

// Residual sum of squares at explicit (constants, a, b). Without SCALED the
// residual is f - y and a/b are ignored.
template <typename T, bool STRICT, bool SCALED>
__device__ __forceinline__ bool lm_explicit_sse(
    const unsigned char* __restrict__ code, const unsigned char* __restrict__ aux,
    const T* __restrict__ imm, int len,
    const T* __restrict__ x, int D, const T* __restrict__ y,
    const T* __restrict__ consts, T a, T b, int lane, T &sse
) {
    T sq = (T)0;
    bool bad = false;
    for (int d0 = 0; d0 < D; d0 += 32) {
        const int d = d0 + lane;
        if (d < D) {
            T pred;
            bool ok = rpn_run_program<T, STRICT>(code, aux, imm, len, x, D, d, consts, pred);
            if (!ok || !isfinite(pred)) {
                bad = true;
            } else {
                const T r = SCALED ? (a + b * pred - y[d]) : (pred - y[d]);
                sq += r * r;
            }
        }
        if (__any_sync(RPN_FULL_MASK, bad)) { bad = true; break; }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor_sync(RPN_FULL_MASK, sq, off);
    sq = __shfl_sync(RPN_FULL_MASK, sq, 0);
    sse = sq;
    return !bad && isfinite(sq);
}

// (row, col) of upper-triangle entry e of a P x P symmetric matrix.
__device__ __forceinline__ void lm_unpack(int e, int P, int &i, int &k) {
    i = 0;
    while (e >= P - i) { e -= P - i; ++i; }
    k = i + e;
}

template <typename scalar_t, bool STRICT, bool SCALED>
__global__ void __launch_bounds__(LM_WARPS_PER_BLOCK * 32)
lm_optimize_kernel(
    const unsigned char* __restrict__ population,  // [B, L]
    const scalar_t* __restrict__ init_consts,      // [B, K]
    const scalar_t* __restrict__ x,                // [Vars, D]
    const scalar_t* __restrict__ y_target,         // [D]
    scalar_t* __restrict__ out_consts,             // [B, K]
    scalar_t* __restrict__ out_rmse,               // [B]
    int B, int L, int K, int D, int max_iter,
    float const_min, float const_max,
    RpnOpIds ids
) {
    extern __shared__ __align__(16) unsigned char lm_smem[];
    __shared__ scalar_t s_J[LM_WARPS_PER_BLOCK][32][LM_MAX_P + 1];  // last column: residual
    __shared__ scalar_t s_sys[LM_WARPS_PER_BLOCK][LM_N_ENTRIES];
    __shared__ scalar_t s_c[LM_WARPS_PER_BLOCK][2][LM_MAX_K];       // current / trial constants
    __shared__ scalar_t s_ab[LM_WARPS_PER_BLOCK][2];                 // trial (a, b)
    __shared__ int s_ok[LM_WARPS_PER_BLOCK];

    const scalar_t BIG = (scalar_t)1e30;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t b = (int64_t)blockIdx.x * LM_WARPS_PER_BLOCK + warp;
    if (b >= B) return;  // whole warp

    scalar_t* imm = reinterpret_cast<scalar_t*>(lm_smem) + (size_t)warp * L;
    unsigned char* code = lm_smem + (size_t)LM_WARPS_PER_BLOCK * L * sizeof(scalar_t) + (size_t)warp * 4 * L;
    unsigned char* aux = code + L;
    unsigned char* lc = aux + L;   // child tape positions (reverse mode)
    unsigned char* rc = lc + L;
    DecodedProgram<scalar_t> prog{code, aux, imm};

    scalar_t* cur = s_c[warp][0];
    scalar_t* trial = s_c[warp][1];
    const scalar_t* c0 = init_consts + b * (int64_t)K;
    scalar_t* outc = out_consts + b * (int64_t)K;
    for (int k = lane; k < K; k += 32) {
        cur[k] = c0[k];
        trial[k] = c0[k];
        outc[k] = c0[k];
    }

    int n_const = 0;
    const int len = rpn_decode_program_warp<scalar_t, false>(
        population + b * (int64_t)L, L, ids, nullptr, K, prog, lane, &n_const);
    __syncwarp();
    if (len == 0) {
        if (lane == 0) out_rmse[b] = BIG;
        return;
    }
    const int nc = n_const < K ? n_const : K;

    scalar_t sse, ca, cb;
    if (!rpn_warp_sse<scalar_t, STRICT, SCALED>(code, aux, imm, len, x, D, y_target, cur, lane, sse, ca, cb)) {
        if (lane == 0) out_rmse[b] = BIG;
        return;
    }
    if (nc == 0 || nc > LM_MAX_C) {
        if (lane == 0) out_rmse[b] = sqrt(sse / (scalar_t)D);
        return;
    }

    const int P = SCALED ? nc + 2 : nc;
    const int nA = P * (P + 1) / 2;
    const int nE = nA + P;
    double lambda = 1e-3;
    const bool use_tape = len <= LM_TAPE;
    if (use_tape && lane == 0) rpn_program_children(code, len, lc, rc);
    __syncwarp();
    scalar_t tape[LM_TAPE];
    scalar_t adj[LM_TAPE];
    scalar_t grad[LM_MAX_C];

    for (int it = 0; it < max_iter; ++it) {
        // ---------------- Jacobian pass ----------------
        scalar_t acc[LM_ENTRIES_PER_LANE];
#pragma unroll
        for (int q = 0; q < LM_ENTRIES_PER_LANE; ++q) acc[q] = (scalar_t)0;
        bool bad = false;
        for (int d0 = 0; d0 < D; d0 += 32) {
            const int d = d0 + lane;
            scalar_t* row = s_J[warp][lane];
            if (d < D) {
                scalar_t f = (scalar_t)0;
                if (use_tape) {
                    if (rpn_tape_forward<scalar_t, STRICT>(code, aux, imm, lc, rc, len, x, D, d, cur, tape)) {
                        f = tape[len - 1];
                        rpn_tape_backward<scalar_t, STRICT>(code, aux, lc, rc, len, tape, adj, grad, nc);
                        for (int j = 0; j < nc; ++j) row[j] = SCALED ? cb * grad[j] : grad[j];
                    } else {
                        bad = true;
                    }
                } else {
                    for (int j = 0; j < nc; ++j) {
                        scalar_t v, dv;
                        if (!rpn_run_program_dual<scalar_t, STRICT>(code, aux, imm, len, x, D, d, cur, j, v, dv)) {
                            bad = true;
                            break;
                        }
                        f = v;
                        row[j] = SCALED ? cb * dv : dv;
                    }
                }
                if (!isfinite(f)) bad = true;
                if (SCALED) {
                    row[nc] = (scalar_t)1;
                    row[nc + 1] = f;
                    row[P] = ca + cb * f - y_target[d];
                } else {
                    row[P] = f - y_target[d];
                }
            } else {
                for (int j = 0; j <= P; ++j) row[j] = (scalar_t)0;
            }
            if (__any_sync(RPN_FULL_MASK, bad)) { bad = true; break; }
            __syncwarp();
#pragma unroll
            for (int q = 0; q < LM_ENTRIES_PER_LANE; ++q) {
                const int e = lane + 32 * q;
                if (e < nE) {
                    int ri, ck;
                    if (e < nA) lm_unpack(e, P, ri, ck);
                    else { ri = e - nA; ck = P; }
                    scalar_t s = (scalar_t)0;
                    for (int p = 0; p < 32; ++p) s += s_J[warp][p][ri] * s_J[warp][p][ck];
                    acc[q] += s;
                }
            }
            __syncwarp();
        }
        if (bad) break;  // keep the last accepted constants
#pragma unroll
        for (int q = 0; q < LM_ENTRIES_PER_LANE; ++q) {
            const int e = lane + 32 * q;
            if (e < nE) s_sys[warp][e] = acc[q];
        }
        __syncwarp();

        // ---------------- Damped solves ----------------
        bool accepted = false;
        scalar_t new_sse = sse;
        for (int t = 0; t < LM_MAX_TRIALS; ++t) {
            if (lane == 0) {
                double M[LM_MAX_P][LM_MAX_P];
                double g[LM_MAX_P];
                double trace = 0.0;
                for (int i = 0; i < P; ++i) {
                    for (int k = i; k < P; ++k) {
                        const int e = i * P - (i * (i - 1)) / 2 + (k - i);
                        M[i][k] = (double)s_sys[warp][e];
                        M[k][i] = M[i][k];
                    }
                    g[i] = (double)s_sys[warp][nA + i];
                    trace += M[i][i];
                }
                const double floor_d = 1e-9 * (trace / P) + 1e-30;
                for (int i = 0; i < P; ++i) M[i][i] += lambda * (M[i][i] + floor_d) + 1e-30;
                // In-place Cholesky (lower triangle). A Jacobi-scaled float32
                // solve was measured to be no faster and converged worse.
                bool ok = isfinite(trace);
                for (int j = 0; j < P && ok; ++j) {
                    double s = M[j][j];
                    for (int k = 0; k < j; ++k) s -= M[j][k] * M[j][k];
                    if (!(s > 0.0)) { ok = false; break; }
                    const double ljj = sqrt(s);
                    M[j][j] = ljj;
                    for (int i = j + 1; i < P; ++i) {
                        double v = M[i][j];
                        for (int k = 0; k < j; ++k) v -= M[i][k] * M[j][k];
                        M[i][j] = v / ljj;
                    }
                }
                double delta[LM_MAX_P];
                if (ok) {
                    for (int i = 0; i < P; ++i) {          // L z = -g
                        double v = -g[i];
                        for (int k = 0; k < i; ++k) v -= M[i][k] * delta[k];
                        delta[i] = v / M[i][i];
                    }
                    for (int i = P - 1; i >= 0; --i) {     // L^T delta = z
                        double v = delta[i];
                        for (int k = i + 1; k < P; ++k) v -= M[k][i] * delta[k];
                        delta[i] = v / M[i][i];
                    }
                    for (int i = 0; i < P; ++i) if (!isfinite(delta[i])) ok = false;
                }
                if (ok) {
                    for (int j = 0; j < nc; ++j) {
                        scalar_t v = (scalar_t)((double)cur[j] + delta[j]);
                        if (v < (scalar_t)const_min) v = (scalar_t)const_min;
                        if (v > (scalar_t)const_max) v = (scalar_t)const_max;
                        trial[j] = v;
                    }
                    if (SCALED) {
                        s_ab[warp][0] = (scalar_t)((double)ca + delta[nc]);
                        s_ab[warp][1] = (scalar_t)((double)cb + delta[nc + 1]);
                    }
                }
                s_ok[warp] = ok ? 1 : 0;
            }
            __syncwarp();
            if (!s_ok[warp]) { lambda *= 10.0; __syncwarp(); continue; }
            scalar_t ta = SCALED ? s_ab[warp][0] : (scalar_t)0;
            scalar_t tb = SCALED ? s_ab[warp][1] : (scalar_t)1;
            scalar_t tsse;
            const bool tok = lm_explicit_sse<scalar_t, STRICT, SCALED>(
                code, aux, imm, len, x, D, y_target, trial, ta, tb, lane, tsse);
            if (tok && tsse < sse) {
                for (int j = lane; j < nc; j += 32) cur[j] = trial[j];
                __syncwarp();
                new_sse = tsse;
                ca = ta;
                cb = tb;
                lambda = fmax(lambda * 0.3, 1e-9);
                accepted = true;
                break;
            }
            lambda *= 4.0;
            __syncwarp();
        }
        if (!accepted) break;

        if (SCALED) {
            // Variable projection: (a, b) become the exact least squares fit of
            // the current program, which can only lower the error.
            scalar_t psse, pa, pb;
            if (rpn_warp_sse<scalar_t, STRICT, SCALED>(code, aux, imm, len, x, D, y_target, cur, lane, psse, pa, pb)
                    && psse <= new_sse) {
                new_sse = psse;
                ca = pa;
                cb = pb;
            }
        }
        const scalar_t prev = sse;
        sse = new_sse;
        if (prev - sse <= (scalar_t)1e-7 * prev) break;  // relative progress stalled
    }

    for (int k = lane; k < K; k += 32) outc[k] = cur[k];
    if (lane == 0) {
        scalar_t rmse = sqrt(sse / (scalar_t)D);
        out_rmse[b] = isfinite(rmse) ? rmse : BIG;
    }
}

void launch_lm_optimize(
    const torch::Tensor& population,   // [B, L] uint8
    const torch::Tensor& init_consts,  // [B, K]
    const torch::Tensor& x,            // [Vars, D]
    const torch::Tensor& y_target,     // [D]
    torch::Tensor& out_consts,         // [B, K]
    torch::Tensor& out_rmse,           // [B]
    int max_iter,
    float const_min, float const_max,
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
    double pi_val, double e_val,
    int strict_mode,
    int scaled
) {
    CHECK_INPUT(population);
    CHECK_INPUT(init_consts);
    CHECK_INPUT(x);
    CHECK_INPUT(y_target);
    CHECK_INPUT(out_consts);
    CHECK_INPUT(out_rmse);

    const int B = population.size(0);
    const int L = population.size(1);
    const int K = init_consts.size(1);
    const int num_vars = x.size(0);
    const int D = x.size(1);

    TORCH_CHECK(population.scalar_type() == torch::kUInt8, "population must use uint8 tokens");
    TORCH_CHECK(L <= LM_MAX_L, "lm_optimize: program length exceeds ", LM_MAX_L);
    TORCH_CHECK(K <= LM_MAX_K, "lm_optimize: at most ", LM_MAX_K, " constant slots");
    TORCH_CHECK(D > 0, "lm_optimize needs at least one sample");
    TORCH_CHECK(init_consts.size(0) == B, "init_consts must have one row per individual");
    TORCH_CHECK(y_target.numel() == D, "y_target must have one value per sample");
    TORCH_CHECK(out_consts.sizes() == init_consts.sizes(), "out_consts must match init_consts");
    TORCH_CHECK(out_rmse.numel() == B, "out_rmse must have one entry per individual");
    TORCH_CHECK(init_consts.scalar_type() == x.scalar_type() && y_target.scalar_type() == x.scalar_type() &&
                out_consts.scalar_type() == x.scalar_type() && out_rmse.scalar_type() == x.scalar_type(),
                "lm_optimize tensors must share the dtype of x");
    if (B == 0) return;

    RpnOpIds ids;
    ids.pad = PAD_ID; ids.x_start = id_x_start; ids.num_vars = num_vars;
    ids.c = id_C; ids.pi = id_pi; ids.e = id_e;
    ids.l0 = id_0; ids.l1 = id_1; ids.l2 = id_2; ids.l3 = id_3;
    ids.l4 = id_4; ids.l5 = id_5; ids.l6 = id_6; ids.l10 = id_10;
    ids.add = op_add; ids.sub = op_sub; ids.mul = op_mul; ids.div = op_div;
    ids.pow = op_pow; ids.mod = op_mod;
    ids.sin = op_sin; ids.cos = op_cos; ids.tan = op_tan; ids.log = op_log; ids.exp = op_exp;
    ids.sqrt = op_sqrt; ids.abs = op_abs; ids.neg = op_neg;
    ids.fact = op_fact; ids.floor = op_floor; ids.ceil = op_ceil; ids.sign = op_sign;
    ids.gamma = op_gamma; ids.lgamma = op_lgamma;
    ids.asin = op_asin; ids.acos = op_acos; ids.atan = op_atan;

    const int threads = LM_WARPS_PER_BLOCK * 32;
    const int64_t blocks = ((int64_t)B + LM_WARPS_PER_BLOCK - 1) / LM_WARPS_PER_BLOCK;
    TORCH_CHECK(blocks <= 2147483647LL, "lm_optimize: population too large for one launch");

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "lm_optimize_kernel", ([&] {
        const size_t smem = (size_t)LM_WARPS_PER_BLOCK * L * (sizeof(scalar_t) + 4);
        auto launch = [&](auto strict_tag, auto scaled_tag) {
            constexpr bool strict = decltype(strict_tag)::value;
            constexpr bool sc = decltype(scaled_tag)::value;
            lm_optimize_kernel<scalar_t, strict, sc><<<(unsigned int)blocks, threads, smem>>>(
                population.data_ptr<unsigned char>(),
                init_consts.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(),
                y_target.data_ptr<scalar_t>(),
                out_consts.data_ptr<scalar_t>(),
                out_rmse.data_ptr<scalar_t>(),
                B, L, K, D, max_iter, const_min, const_max, ids);
        };
        if (strict_mode) {
            if (scaled) launch(std::true_type{}, std::true_type{});
            else launch(std::true_type{}, std::false_type{});
        } else {
            if (scaled) launch(std::false_type{}, std::true_type{});
            else launch(std::false_type{}, std::false_type{});
        }
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error in lm_optimize: ", cudaGetErrorString(err));
}
