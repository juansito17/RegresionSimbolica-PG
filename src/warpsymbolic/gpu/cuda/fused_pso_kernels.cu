/*
 * Fused PSO for constant optimisation.
 *
 * One block per individual and one warp per particle. The formula is decoded
 * once per block (same decoder and operator semantics as the fused RMSE
 * evaluator); every warp then evaluates its particle on all samples with the
 * lanes striding over the data, so the work is parallel over samples instead of
 * serial per thread. Personal/global bests, velocities and positions live in
 * shared memory, and randomness comes from a counter-based Philox stream.
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

#define PSO_MAX_L 256
#define PSO_MAX_K 16
#define PSO_MAX_PARTICLES 32

template <typename scalar_t>
__device__ __forceinline__ scalar_t pso_normal(uint4 r) {
    float u1 = rpn_u01(r.x);
    float u2 = rpn_u01(r.y);
    return (scalar_t)(sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2));
}

template <typename scalar_t, bool STRICT>
__global__ void __launch_bounds__(1024)
fused_pso_kernel(
    const unsigned char* __restrict__ population,  // [B, L]
    const scalar_t* __restrict__ init_consts,      // [B, K]
    const scalar_t* __restrict__ x,                // [Vars, D]
    const scalar_t* __restrict__ y_target,         // [D]
    scalar_t* __restrict__ out_gbest_pos,          // [B, K]
    scalar_t* __restrict__ out_gbest_err,          // [B]
    int B, int L, int K, int D,
    int P, int num_steps,
    float w, float c1, float c2,
    float const_min, float const_max,
    uint64_t rng_seed,
    RpnOpIds ids
) {
    extern __shared__ __align__(16) unsigned char pso_smem[];
    const scalar_t BIG = (scalar_t)1e30;
    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int p_warp = tid >> 5;
    const int nthreads = blockDim.x;

    scalar_t* imm = reinterpret_cast<scalar_t*>(pso_smem);
    scalar_t* pos = imm + L;
    scalar_t* vel = pos + P * K;
    scalar_t* pbest = vel + P * K;
    scalar_t* pbest_err = pbest + P * K;
    scalar_t* cur_err = pbest_err + P;
    scalar_t* gbest = cur_err + P;
    unsigned char* code = reinterpret_cast<unsigned char*>(gbest + K);
    unsigned char* aux = code + L;

    __shared__ int s_len;
    __shared__ int s_nconst;
    __shared__ int s_best_p;
    __shared__ scalar_t s_gbest_err;

    const scalar_t* init = init_consts + (int64_t)b * K;

    if (p_warp == 0) {
        int n_const = 0;
        DecodedProgram<scalar_t> prog{code, aux, imm};
        int l = rpn_decode_program_warp<scalar_t, false>(
            population + (int64_t)b * L, L, ids, nullptr, K, prog, lane, &n_const);
        if (lane == 0) {
            s_len = l;
            s_nconst = n_const;
            s_gbest_err = BIG;
        }
    }
    __syncthreads();
    const int len = s_len;
    const int n_active = s_nconst < K ? s_nconst : K;

    if (len == 0) {
        for (int k = tid; k < K; k += nthreads) out_gbest_pos[(int64_t)b * K + k] = init[k];
        if (tid == 0) out_gbest_err[b] = BIG;
        return;
    }

    const scalar_t range = (scalar_t)(const_max - const_min);
    const scalar_t jitter_sigma = range * (scalar_t)0.15;
    const scalar_t vel_sigma = range * (scalar_t)0.02;
    for (int idx = tid; idx < P * K; idx += nthreads) {
        const int p = idx / K;
        const int k = idx - p * K;
        scalar_t v0 = init[k];
        scalar_t vv = (scalar_t)0.0;
        if (k < n_active) {
            uint4 r = rpn_random4(rng_seed, (uint64_t)b, (uint64_t)idx, 0xFFFFFFFFULL);
            if (p > 0) {
                v0 += pso_normal<scalar_t>(r) * jitter_sigma;
                if (v0 < (scalar_t)const_min) v0 = (scalar_t)const_min;
                if (v0 > (scalar_t)const_max) v0 = (scalar_t)const_max;
            }
            uint4 r2 = make_uint4(r.z, r.w, r.x ^ 0xA5A5A5A5u, r.y);
            vv = pso_normal<scalar_t>(r2) * vel_sigma;
        }
        pos[idx] = v0;
        vel[idx] = vv;
        pbest[idx] = v0;
    }
    for (int p = tid; p < P; p += nthreads) pbest_err[p] = BIG;
    for (int k = tid; k < K; k += nthreads) gbest[k] = init[k];
    __syncthreads();

    // Formulas without constants: a single evaluation is all there is to do.
    const int steps = (n_active == 0) ? 1 : num_steps;
    const int active_particles = (n_active == 0) ? 1 : P;

    for (int step = 0; step < steps; ++step) {
        // --- 1. Evaluate: warp p evaluates particle p over all samples ---
        if (p_warp < active_particles) {
            const scalar_t* my_consts = pos + p_warp * K;
            scalar_t sq = (scalar_t)0.0;
            bool bad = false;
            for (int d0 = 0; d0 < D; d0 += 32) {
                const int d = d0 + lane;
                if (d < D) {
                    scalar_t pred;
                    bool ok = rpn_run_program<scalar_t, STRICT>(code, aux, imm, len, x, D, d, my_consts, pred);
                    if (!ok || isnan(pred) || isinf(pred)) {
                        bad = true;
                    } else {
                        scalar_t diff = pred - y_target[d];
                        scalar_t s2 = diff * diff;
                        if (isnan(s2) || isinf(s2)) bad = true;
                        else sq += s2;
                    }
                }
                if (__any_sync(RPN_FULL_MASK, bad)) { bad = true; break; }
            }
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor_sync(RPN_FULL_MASK, sq, off);
            if (lane == 0) {
                scalar_t rmse = bad ? BIG : sqrt(sq / (scalar_t)D);
                if (isnan(rmse) || isinf(rmse)) rmse = BIG;
                cur_err[p_warp] = rmse;
            }
        }
        __syncthreads();

        // --- 2. Personal bests (copy first, then update the error) ---
        for (int idx = tid; idx < active_particles * K; idx += nthreads) {
            const int p = idx / K;
            if (cur_err[p] < pbest_err[p]) pbest[idx] = pos[idx];
        }
        __syncthreads();
        for (int p = tid; p < active_particles; p += nthreads) {
            if (cur_err[p] < pbest_err[p]) pbest_err[p] = cur_err[p];
        }
        __syncthreads();

        // --- 3. Global best (argmin over particles by warp 0) ---
        if (p_warp == 0) {
            scalar_t v = (lane < active_particles) ? pbest_err[lane] : BIG;
            int bi = (lane < active_particles) ? lane : -1;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                scalar_t ov = __shfl_xor_sync(RPN_FULL_MASK, v, off);
                int oi = __shfl_xor_sync(RPN_FULL_MASK, bi, off);
                if (ov < v || (ov == v && oi >= 0 && (bi < 0 || oi < bi))) { v = ov; bi = oi; }
            }
            if (lane == 0) {
                if (bi >= 0 && v < s_gbest_err) { s_gbest_err = v; s_best_p = bi; }
                else s_best_p = -1;
            }
        }
        __syncthreads();
        if (s_best_p >= 0) {
            for (int k = tid; k < K; k += nthreads) gbest[k] = pbest[s_best_p * K + k];
        }
        __syncthreads();

        if (step + 1 >= steps) break;

        // --- 4. Velocity / position update (linear inertia decay w -> 0.4) ---
        const scalar_t w_curr = (scalar_t)w - ((scalar_t)w - (scalar_t)0.4) *
            (scalar_t)step / (scalar_t)(steps > 1 ? steps - 1 : 1);
        for (int idx = tid; idx < P * K; idx += nthreads) {
            const int p = idx / K;
            const int k = idx - p * K;
            if (k >= n_active) continue;
            uint4 r = rpn_random4(rng_seed, (uint64_t)b, (uint64_t)idx, (uint64_t)step);
            const scalar_t r1 = (scalar_t)rpn_u01(r.x);
            const scalar_t r2 = (scalar_t)rpn_u01(r.y);
            scalar_t vnew = w_curr * vel[idx]
                + (scalar_t)c1 * r1 * (pbest[idx] - pos[idx])
                + (scalar_t)c2 * r2 * (gbest[k] - pos[idx]);
            scalar_t pnew = pos[idx] + vnew;
            if (pnew < (scalar_t)const_min) pnew = (scalar_t)const_min;
            if (pnew > (scalar_t)const_max) pnew = (scalar_t)const_max;
            vel[idx] = vnew;
            pos[idx] = pnew;
        }
        __syncthreads();
    }

    for (int k = tid; k < K; k += nthreads) out_gbest_pos[(int64_t)b * K + k] = gbest[k];
    if (tid == 0) out_gbest_err[b] = s_gbest_err;
}


// ===================== C++ Wrapper =====================
void launch_fused_pso(
    const torch::Tensor& population,    // [B, L]
    const torch::Tensor& init_consts,   // [B, K]
    const torch::Tensor& x,             // [Vars, D]
    const torch::Tensor& y_target,      // [D]
    torch::Tensor& out_gbest_pos,       // [B, K]
    torch::Tensor& out_gbest_err,       // [B]
    int num_particles, int num_steps,
    float w, float c1, float c2,
    float const_min, float const_max,
    // OpCode IDs
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
    uint64_t rng_seed,
    int strict_mode
) {
    CHECK_INPUT(population);
    CHECK_INPUT(init_consts);
    CHECK_INPUT(x);
    CHECK_INPUT(y_target);
    CHECK_INPUT(out_gbest_pos);
    CHECK_INPUT(out_gbest_err);

    int B = population.size(0);
    int L = population.size(1);
    int K = init_consts.size(1);
    int num_vars = x.size(0);
    int D = x.size(1);

    TORCH_CHECK(L <= PSO_MAX_L, "Formula length exceeds PSO_MAX_L");
    TORCH_CHECK(K <= PSO_MAX_K, "Constants exceed PSO_MAX_K");
    TORCH_CHECK(D > 0, "fused_pso needs at least one sample");
    TORCH_CHECK(num_particles >= 1 && num_particles <= PSO_MAX_PARTICLES,
                "num_particles must be in 1..", PSO_MAX_PARTICLES);
    TORCH_CHECK(init_consts.scalar_type() == x.scalar_type(), "init_consts dtype must match x");
    TORCH_CHECK(y_target.scalar_type() == x.scalar_type(), "y_target dtype must match x");
    TORCH_CHECK(out_gbest_pos.scalar_type() == x.scalar_type(), "out_gbest_pos dtype must match x");
    TORCH_CHECK(out_gbest_err.scalar_type() == x.scalar_type(), "out_gbest_err dtype must match x");
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

    const int threads = 32 * num_particles;
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "fused_pso_kernel", ([&] {
        size_t smem = (size_t)L * sizeof(scalar_t)
                    + (size_t)3 * num_particles * K * sizeof(scalar_t)
                    + (size_t)2 * num_particles * sizeof(scalar_t)
                    + (size_t)K * sizeof(scalar_t)
                    + (size_t)2 * L;
        auto launch = [&](auto strict_tag) {
            constexpr bool strict = decltype(strict_tag)::value;
            fused_pso_kernel<scalar_t, strict><<<B, threads, smem>>>(
                population.data_ptr<unsigned char>(),
                init_consts.data_ptr<scalar_t>(),
                x.data_ptr<scalar_t>(),
                y_target.data_ptr<scalar_t>(),
                out_gbest_pos.data_ptr<scalar_t>(),
                out_gbest_err.data_ptr<scalar_t>(),
                B, L, K, D,
                num_particles, num_steps,
                w, c1, c2,
                const_min, const_max,
                rng_seed, ids);
        };
        if (strict_mode) launch(std::true_type{});
        else launch(std::false_type{});
    }));

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "CUDA Error in fused_pso: ", cudaGetErrorString(err));
}
