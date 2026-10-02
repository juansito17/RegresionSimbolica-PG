// Shared RPN semantics and the decoded interpreter used by the fused kernels.
//
// Every evaluator (fused RMSE, classic per-sample, fused PSO) uses the same
// protected/strict operator definitions from this header, so constant
// optimisation and fitness evaluation agree on the meaning of a formula.
//
// The fused kernels decode a program once per individual: token -> opcode,
// variable index, constant slot or literal value, while validating the stack
// discipline with a warp-wide prefix scan. Sample evaluation then runs a dense
// switch over pre-decoded opcodes with the top of stack cached in a register.
#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <cstdint>
#include <climits>

// Maximum stack depth accepted by every evaluator. A valid RPN program of
// length L has depth <= (L + 1) / 2, so 64 covers every program up to 127
// tokens; deeper programs are reported invalid instead of being truncated.
#define RPN_EVAL_STACK 64
#define RPN_FULL_MASK 0xFFFFFFFFu

// ----------------------------------------------------------------------------
// Operator semantics
// ----------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T safe_div(T a, T b, bool &error) {
    if (fabs(b) < (T)1e-9) return a;  // protected division
    return a / b;
}

template <typename T>
__device__ __forceinline__ T safe_mod(T a, T b, bool &error) {
    if (fabs(b) < (T)1e-9) return (T)0.0;  // protected modulo
    T r = fmod(a, b);
    if ((b > 0 && r < 0) || (b < 0 && r > 0)) r += b;
    return r;
}

template <typename T>
__device__ __forceinline__ T safe_log(T a, bool &error) {
    return log(fabs(a) + (T)1e-9);
}

template <typename T>
__device__ __forceinline__ T safe_exp(T a, bool &error) {
    T x = a;
    if (x < (T)-80.0) x = (T)-80.0;
    if (x > (T)80.0) x = (T)80.0;
    return exp(x);
}

template <typename T>
__device__ __forceinline__ T safe_sqrt(T a, bool &error) {
    return sqrt(fabs(a));
}

template <typename T>
__device__ __forceinline__ T safe_pow(T a, T b, bool &error) {
    if (a != a || b != b) { error = true; return (T)0.0; }
    if (fabs(a) < (T)1e-10 && fabs(b) < (T)1e-10) return (T)1.0;
    if (a < (T)0.0) {
        T ib = round(b);
        if (fabs(b - ib) > (T)1e-3) a = fabs(a);
        else b = ib;
    }
    if (fabs(a) > (T)1.0 && b > (T)80.0) b = (T)80.0;
    if (fabs(a) > (T)100.0 && b > (T)10.0) b = (T)10.0;
    T res = pow(a, b);
    if (res != res || isinf(res)) { error = true; return (T)0.0; }
    return res;
}

template <typename T>
__device__ __forceinline__ T safe_asin(T a, bool &error) {
    if (a < (T)-1.0) a = (T)-1.0;
    if (a > (T)1.0) a = (T)1.0;
    return asin(a);
}

template <typename T>
__device__ __forceinline__ T safe_acos(T a, bool &error) {
    if (a < (T)-1.0) a = (T)-1.0;
    if (a > (T)1.0) a = (T)1.0;
    return acos(a);
}

template <typename T>
__device__ __forceinline__ T safe_tgamma(T a, bool &error) {
    if (a <= (T)0.0 && floor(a) == a) return (T)0.0;
    T res = tgamma(a);
    if (res != res || isinf(res)) { error = true; return (T)0.0; }
    return res;
}

template <typename T>
__device__ __forceinline__ T safe_lgamma(T a, bool &error) {
    if (a <= (T)0.0 && floor(a) == a) return (T)0.0;
    T res = lgamma(a);
    if (res != res || isinf(res)) { error = true; return (T)0.0; }
    return res;
}

template <typename T>
__device__ __forceinline__ T strict_div(T a, T b, bool &error) {
    if (fabs(b) < (T)1e-9) { error = true; return (T)0.0; }
    return a / b;
}

template <typename T>
__device__ __forceinline__ T strict_mod(T a, T b, bool &error) {
    if (fabs(b) < (T)1e-9) { error = true; return (T)0.0; }
    T r = fmod(a, b);
    if ((b > 0 && r < 0) || (b < 0 && r > 0)) r += b;
    return r;
}

template <typename T>
__device__ __forceinline__ T strict_log(T a, bool &error) {
    if (a <= (T)0.0) { error = true; return (T)0.0; }
    return log(a);
}

template <typename T>
__device__ __forceinline__ T strict_exp(T a, bool &error) {
    T res = exp(a);
    if (isinf(res)) { error = true; return (T)0.0; }
    return res;
}

template <typename T>
__device__ __forceinline__ T strict_sqrt(T a, bool &error) {
    if (a < (T)0.0) { error = true; return (T)0.0; }
    return sqrt(a);
}

template <typename T>
__device__ __forceinline__ T strict_pow(T a, T b, bool &error) {
    if (a != a || b != b) { error = true; return (T)0.0; }
    if (fabs(a) < (T)1e-10 && fabs(b) < (T)1e-10) return (T)1.0;
    if (a < (T)0.0) {
        T ib = round(b);
        if (fabs(b - ib) > (T)1e-3) { error = true; return (T)0.0; }
        b = ib;
    }
    T res = pow(a, b);
    if (res != res || isinf(res)) { error = true; return (T)0.0; }
    return res;
}

template <typename T>
__device__ __forceinline__ T strict_asin(T a, bool &error) {
    if (a < (T)-1.0 || a > (T)1.0) { error = true; return (T)0.0; }
    return asin(a);
}

template <typename T>
__device__ __forceinline__ T strict_acos(T a, bool &error) {
    if (a < (T)-1.0 || a > (T)1.0) { error = true; return (T)0.0; }
    return acos(a);
}

template <typename T>
__device__ __forceinline__ T strict_tgamma(T a, bool &error) {
    if (a <= (T)0.0 && floor(a) == a) { error = true; return (T)0.0; }
    T res = tgamma(a);
    if (res != res || isinf(res)) { error = true; return (T)0.0; }
    return res;
}

template <typename T>
__device__ __forceinline__ T strict_lgamma(T a, bool &error) {
    if (a <= (T)0.0 && floor(a) == a) { error = true; return (T)0.0; }
    T res = lgamma(a);
    if (res != res || isinf(res)) { error = true; return (T)0.0; }
    return res;
}

// ----------------------------------------------------------------------------
// Counter-based RNG shared by the native operators
// ----------------------------------------------------------------------------

__device__ __forceinline__ uint4 rpn_philox4x32_10(uint4 counter, uint2 key) {
    constexpr uint32_t M0 = 0xD2511F53U;
    constexpr uint32_t M1 = 0xCD9E8D57U;
    constexpr uint32_t W0 = 0x9E3779B9U;
    constexpr uint32_t W1 = 0xBB67AE85U;
#pragma unroll
    for (int round = 0; round < 10; ++round) {
        uint32_t hi0 = __umulhi(M0, counter.x);
        uint32_t lo0 = M0 * counter.x;
        uint32_t hi1 = __umulhi(M1, counter.z);
        uint32_t lo1 = M1 * counter.z;
        counter = make_uint4(hi1 ^ counter.y ^ key.x, lo1,
                             hi0 ^ counter.w ^ key.y, lo0);
        key.x += W0;
        key.y += W1;
    }
    return counter;
}

__device__ __forceinline__ uint4 rpn_random4(uint64_t seed, uint64_t a, uint64_t b, uint64_t c) {
    uint4 counter = make_uint4((uint32_t)a, (uint32_t)(a >> 32), (uint32_t)b,
                               (uint32_t)c ^ ((uint32_t)(b >> 32) * 0x9E3779B9U));
    uint2 key = make_uint2((uint32_t)seed, (uint32_t)(seed >> 32));
    return rpn_philox4x32_10(counter, key);
}

__device__ __forceinline__ float rpn_u01(uint32_t bits) {
    // (0, 1]: never returns 0, safe for log() in Box-Muller.
    return ((float)(bits >> 8) + 1.0f) * (1.0f / 16777216.0f);
}

// ----------------------------------------------------------------------------
// Opcode decoding
// ----------------------------------------------------------------------------

struct RpnOpIds {
    int pad, x_start, num_vars, c, pi, e;
    int l0, l1, l2, l3, l4, l5, l6, l10;
    int add, sub, mul, div, pow, mod;
    int sin, cos, tan, log, exp, sqrt, abs, neg;
    int fact, floor, ceil, sign, gamma, lgamma, asin, acos, atan;
};

enum RpnCode : unsigned char {
    RC_INVALID = 0,
    RC_VAR, RC_CONST, RC_IMM,
    RC_ADD, RC_SUB, RC_MUL, RC_DIV, RC_POW, RC_MOD,
    RC_SIN, RC_COS, RC_TAN, RC_ASIN, RC_ACOS, RC_ATAN,
    RC_LOG, RC_EXP, RC_SQRT, RC_ABS, RC_NEG,
    RC_FLOOR, RC_CEIL, RC_SIGN, RC_FACT, RC_GAMMA, RC_LGAMMA,
};

// Decode one (non-PAD) token. Returns the opcode and writes the stack delta
// and, for literals, the immediate value.
__device__ __forceinline__ unsigned char rpn_decode(const RpnOpIds &ids, int t, int &delta, double &imm) {
    delta = 1;
    imm = 0.0;
    if (t >= ids.x_start && t < ids.x_start + ids.num_vars) return RC_VAR;
    if (t == ids.c) return RC_CONST;
    if (t == ids.l0) { imm = 0.0; return RC_IMM; }
    if (t == ids.l1) { imm = 1.0; return RC_IMM; }
    if (t == ids.l2) { imm = 2.0; return RC_IMM; }
    if (t == ids.l3) { imm = 3.0; return RC_IMM; }
    if (t == ids.l4) { imm = 4.0; return RC_IMM; }
    if (t == ids.l5) { imm = 5.0; return RC_IMM; }
    if (t == ids.l6) { imm = 6.0; return RC_IMM; }
    if (t == ids.l10) { imm = 10.0; return RC_IMM; }
    if (t == ids.pi) { imm = 3.141592653589793; return RC_IMM; }
    if (t == ids.e) { imm = 2.718281828459045; return RC_IMM; }
    delta = -1;
    if (t == ids.add) return RC_ADD;
    if (t == ids.sub) return RC_SUB;
    if (t == ids.mul) return RC_MUL;
    if (t == ids.div) return RC_DIV;
    if (t == ids.pow) return RC_POW;
    if (t == ids.mod) return RC_MOD;
    delta = 0;
    if (t == ids.sin) return RC_SIN;
    if (t == ids.cos) return RC_COS;
    if (t == ids.tan) return RC_TAN;
    if (t == ids.asin) return RC_ASIN;
    if (t == ids.acos) return RC_ACOS;
    if (t == ids.atan) return RC_ATAN;
    if (t == ids.log) return RC_LOG;
    if (t == ids.exp) return RC_EXP;
    if (t == ids.sqrt) return RC_SQRT;
    if (t == ids.abs) return RC_ABS;
    if (t == ids.neg) return RC_NEG;
    if (t == ids.floor) return RC_FLOOR;
    if (t == ids.ceil) return RC_CEIL;
    if (t == ids.sign) return RC_SIGN;
    if (t == ids.fact) return RC_FACT;
    if (t == ids.gamma) return RC_GAMMA;
    if (t == ids.lgamma) return RC_LGAMMA;
    return RC_INVALID;
}

// Decoded program stored in shared memory.
template <typename T>
struct DecodedProgram {
    unsigned char* code;  // [L]
    unsigned char* aux;   // [L] variable index or constant slot
    T* imm;               // [L] literal (or resolved constant) value
};

// Decode the program of one individual with a single warp.
// Returns the program length (> 0) when the program is a valid RPN expression
// whose stack depth stays within RPN_EVAL_STACK, or 0 otherwise.
// When RESOLVE_CONST is true, constant tokens become immediates holding
// constants[slot]; otherwise they stay RC_CONST with aux = slot.
template <typename T, bool RESOLVE_CONST>
__device__ __forceinline__ int rpn_decode_program_warp(
    const unsigned char* __restrict__ row, int L,
    const RpnOpIds &ids,
    const T* __restrict__ consts_row, int K,
    DecodedProgram<T> prog, int lane, int* n_const_out = nullptr
) {
    int depth_base = 0;
    int const_base = 0;
    int len = L;
    bool ok = true;
    for (int base = 0; base < L; base += 32) {
        int i = base + lane;
        int t = (i < L) ? (int)row[i] : ids.pad;
        unsigned int pad_mask = __ballot_sync(RPN_FULL_MASK, t == ids.pad);
        int chunk_len = 32;
        if (pad_mask) chunk_len = __ffs(pad_mask) - 1;
        bool active = lane < chunk_len;

        int delta = 0;
        double imm = 0.0;
        unsigned char c = RC_INVALID;
        if (active) c = rpn_decode(ids, t, delta, imm);
        if (active && c == RC_INVALID) ok = false;

        unsigned int c_mask = __ballot_sync(RPN_FULL_MASK, active && c == RC_CONST);
        int slot = const_base + __popc(c_mask & ((1u << lane) - 1u));
        const_base += __popc(c_mask);

        // Inclusive prefix sum of stack deltas.
        int v = active ? delta : 0;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            int n = __shfl_up_sync(RPN_FULL_MASK, v, off);
            if (lane >= off) v += n;
        }
        int depth = depth_base + v;
        if (active && (depth < 1 || depth > RPN_EVAL_STACK)) ok = false;
        depth_base += __shfl_sync(RPN_FULL_MASK, v, 31);

        if (active) {
            unsigned char a = 0;
            T value = (T)imm;
            if (c == RC_VAR) {
                a = (unsigned char)(t - ids.x_start);
            } else if (c == RC_CONST) {
                int s = slot;
                if (K > 0 && s >= K) s = K - 1;
                if (s > 255) s = 255;
                a = (unsigned char)s;
                if (RESOLVE_CONST) {
                    value = (K > 0) ? consts_row[s] : (T)1.0;
                    c = RC_IMM;
                }
            }
            prog.code[i] = c;
            prog.aux[i] = a;
            prog.imm[i] = value;
        }
        if (pad_mask) {
            len = base + chunk_len;
            break;
        }
    }
    ok = __all_sync(RPN_FULL_MASK, ok);
    if (n_const_out != nullptr) *n_const_out = const_base;
    if (!ok || len <= 0 || depth_base != 1) return 0;
    return len;
}

// Evaluate a decoded program on sample d. The top of stack lives in a register.
// Returns false on a math error; `out` receives the prediction otherwise.
template <typename T, bool STRICT>
__device__ __forceinline__ bool rpn_run_program(
    const unsigned char* __restrict__ code,
    const unsigned char* __restrict__ aux,
    const T* __restrict__ imm,
    int len,
    const T* __restrict__ x, long long D, long long d,
    const T* __restrict__ particle_consts,
    T &out
) {
    T stack[RPN_EVAL_STACK];
    int sp = 0;
    T top = (T)0.0;
    bool err = false;
    for (int pc = 0; pc < len; ++pc) {
        switch (code[pc]) {
            case RC_IMM: stack[sp++] = top; top = imm[pc]; break;
            case RC_VAR: stack[sp++] = top; top = x[(long long)aux[pc] * D + d]; break;
            case RC_CONST: stack[sp++] = top; top = particle_consts[aux[pc]]; break;
            case RC_ADD: top = stack[--sp] + top; break;
            case RC_SUB: top = stack[--sp] - top; break;
            case RC_MUL: top = stack[--sp] * top; break;
            case RC_DIV: { T a = stack[--sp]; top = STRICT ? strict_div(a, top, err) : safe_div(a, top, err); break; }
            case RC_POW: { T a = stack[--sp]; top = STRICT ? strict_pow(a, top, err) : safe_pow(a, top, err); break; }
            case RC_MOD: { T a = stack[--sp]; top = STRICT ? strict_mod(a, top, err) : safe_mod(a, top, err); break; }
            case RC_SIN: top = sin(top); break;
            case RC_COS: top = cos(top); break;
            case RC_TAN: top = tan(top); break;
            case RC_ASIN: top = STRICT ? strict_asin(top, err) : safe_asin(top, err); break;
            case RC_ACOS: top = STRICT ? strict_acos(top, err) : safe_acos(top, err); break;
            case RC_ATAN: top = atan(top); break;
            case RC_LOG: top = STRICT ? strict_log(top, err) : safe_log(top, err); break;
            case RC_EXP: top = STRICT ? strict_exp(top, err) : safe_exp(top, err); break;
            case RC_SQRT: top = STRICT ? strict_sqrt(top, err) : safe_sqrt(top, err); break;
            case RC_ABS: top = fabs(top); break;
            case RC_NEG: top = -top; break;
            case RC_FLOOR: top = floor(top); break;
            case RC_CEIL: top = ceil(top); break;
            case RC_SIGN: top = (top > (T)0.0) ? (T)1.0 : ((top < (T)0.0) ? (T)-1.0 : (T)0.0); break;
            case RC_FACT: top = STRICT ? strict_tgamma(top + (T)1.0, err) : safe_tgamma(top + (T)1.0, err); break;
            case RC_GAMMA: top = STRICT ? strict_tgamma(top, err) : safe_tgamma(top, err); break;
            case RC_LGAMMA: top = STRICT ? strict_lgamma(top, err) : safe_lgamma(top, err); break;
            default: err = true; break;
        }
        if (err) return false;
    }
    out = top;
    return true;
}

__device__ __forceinline__ unsigned long long rpn_shfl_xor_u64(unsigned long long v, int off) {
    return __shfl_xor_sync(RPN_FULL_MASK, v, off);
}

// Bitwise equality: constants are compared exactly (NaN == NaN, -0 != +0), so
// a reused fitness always belongs to the very same program and constants.
__device__ __forceinline__ bool rpn_bits_equal(float a, float b) {
    return __float_as_uint(a) == __float_as_uint(b);
}
__device__ __forceinline__ bool rpn_bits_equal(double a, double b) {
    return __double_as_longlong(a) == __double_as_longlong(b);
}

// ----------------------------------------------------------------------------
// Linear scaling statistics
// ----------------------------------------------------------------------------
//
// With linear scaling the fitness of a program f is the RMSE of the least
// squares fit a + b*f, with b = cov(f, y) / var(f) and a = mean(y) - b*mean(f).
// Means, second moments and the co-moment are accumulated with Welford's
// update and merged with Chan's formula, which stays accurate in float32.
// The residual itself is summed in a second pass: the one-pass identity
// SSE = M2y - Cfy^2 / M2f cancels catastrophically for near-exact fits.

template <typename T>
struct RpnScaleStats {
    T n, mf, my, m2f, m2y, cfy;
};

template <typename T>
__device__ __forceinline__ RpnScaleStats<T> rpn_scale_empty() {
    RpnScaleStats<T> s;
    s.n = (T)0; s.mf = (T)0; s.my = (T)0; s.m2f = (T)0; s.m2y = (T)0; s.cfy = (T)0;
    return s;
}

template <typename T>
__device__ __forceinline__ void rpn_scale_push(RpnScaleStats<T> &s, T f, T y) {
    s.n += (T)1;
    const T inv = (T)1 / s.n;
    const T df = f - s.mf;
    const T dy = y - s.my;
    s.mf += df * inv;
    s.my += dy * inv;
    const T dy2 = y - s.my;
    s.m2f += df * (f - s.mf);
    s.m2y += dy * dy2;
    s.cfy += df * dy2;
}

template <typename T>
__device__ __forceinline__ RpnScaleStats<T> rpn_scale_merge(const RpnScaleStats<T> &a, const RpnScaleStats<T> &b) {
    if (b.n <= (T)0) return a;
    if (a.n <= (T)0) return b;
    RpnScaleStats<T> r;
    r.n = a.n + b.n;
    const T df = b.mf - a.mf;
    const T dy = b.my - a.my;
    const T wb = b.n / r.n;
    const T w = a.n * wb;
    r.mf = a.mf + df * wb;
    r.my = a.my + dy * wb;
    r.m2f = a.m2f + b.m2f + df * df * w;
    r.m2y = a.m2y + b.m2y + dy * dy * w;
    r.cfy = a.cfy + b.cfy + df * dy * w;
    return r;
}

// Merge the statistics of the 32 lanes; every lane receives lane 0's result,
// so all lanes use bit-identical coefficients.
template <typename T>
__device__ __forceinline__ RpnScaleStats<T> rpn_scale_warp_reduce(RpnScaleStats<T> s) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        RpnScaleStats<T> o;
        o.n = __shfl_xor_sync(RPN_FULL_MASK, s.n, off);
        o.mf = __shfl_xor_sync(RPN_FULL_MASK, s.mf, off);
        o.my = __shfl_xor_sync(RPN_FULL_MASK, s.my, off);
        o.m2f = __shfl_xor_sync(RPN_FULL_MASK, s.m2f, off);
        o.m2y = __shfl_xor_sync(RPN_FULL_MASK, s.m2y, off);
        o.cfy = __shfl_xor_sync(RPN_FULL_MASK, s.cfy, off);
        s = rpn_scale_merge(s, o);
    }
    s.n = __shfl_sync(RPN_FULL_MASK, s.n, 0);
    s.mf = __shfl_sync(RPN_FULL_MASK, s.mf, 0);
    s.my = __shfl_sync(RPN_FULL_MASK, s.my, 0);
    s.m2f = __shfl_sync(RPN_FULL_MASK, s.m2f, 0);
    s.m2y = __shfl_sync(RPN_FULL_MASK, s.m2y, 0);
    s.cfy = __shfl_sync(RPN_FULL_MASK, s.cfy, 0);
    return s;
}

// Optimal slope; a (numerically) constant prediction gets b = 0, i.e. the
// fit falls back to mean(y). The variance floor is relative to mean(f)^2
// (1e-10 sits well above float32 rounding noise, ~1e-14), so predictions of
// any magnitude keep their shape while constant noise is never amplified.
template <typename T>
__device__ __forceinline__ T rpn_scale_slope(const RpnScaleStats<T> &s) {
    const T floor_var = s.n * ((T)1e-10 * s.mf * s.mf + (T)1e-30);
    if (!(s.m2f > floor_var)) return (T)0;
    const T b = s.cfy / s.m2f;
    return isfinite(b) ? b : (T)0;
}

template <typename T>
__device__ __forceinline__ bool rpn_scale_finite(const RpnScaleStats<T> &s) {
    return isfinite(s.mf) && isfinite(s.my) && isfinite(s.m2f) && isfinite(s.m2y) && isfinite(s.cfy);
}

// Residual sum of squares of one program over all samples, computed by one
// warp (lanes stride over the samples). With SCALED the residual is the one of
// the least squares fit a + b*f (two passes); otherwise it is f - y.
// Returns false when a sample is invalid or the error overflows. Every lane
// receives the same sse/a/b.
template <typename T, bool STRICT, bool SCALED>
__device__ __forceinline__ bool rpn_warp_sse(
    const unsigned char* __restrict__ code,
    const unsigned char* __restrict__ aux,
    const T* __restrict__ imm,
    int len,
    const T* __restrict__ x, int D,
    const T* __restrict__ y,
    const T* __restrict__ consts,
    int lane,
    T &sse, T &a_out, T &b_out
) {
    bool bad = false;
    T sq = (T)0;
    RpnScaleStats<T> st = rpn_scale_empty<T>();
    for (int d0 = 0; d0 < D; d0 += 32) {
        const int d = d0 + lane;
        if (d < D) {
            T pred;
            bool ok = rpn_run_program<T, STRICT>(code, aux, imm, len, x, D, d, consts, pred);
            if (!ok || !isfinite(pred)) {
                bad = true;
            } else if (SCALED) {
                rpn_scale_push(st, pred, y[d]);
            } else {
                const T diff = pred - y[d];
                const T s2 = diff * diff;
                if (!isfinite(s2)) bad = true;
                else sq += s2;
            }
        }
        if (__any_sync(RPN_FULL_MASK, bad)) { bad = true; break; }
    }
    a_out = (T)0;
    b_out = (T)1;
    if (bad) { sse = (T)0; return false; }
    if (SCALED) {
        st = rpn_scale_warp_reduce(st);
        if (!rpn_scale_finite(st)) { sse = (T)0; return false; }
        const T b = rpn_scale_slope(st);
        for (int d0 = 0; d0 < D; d0 += 32) {
            const int d = d0 + lane;
            if (d < D) {
                T pred;
                rpn_run_program<T, STRICT>(code, aux, imm, len, x, D, d, consts, pred);
                const T r = (y[d] - st.my) - b * (pred - st.mf);
                sq += r * r;
            }
        }
        a_out = st.my - b * st.mf;
        b_out = b;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sq += __shfl_xor_sync(RPN_FULL_MASK, sq, off);
    sq = __shfl_sync(RPN_FULL_MASK, sq, 0);
    sse = sq;
    return isfinite(sq);
}

// ----------------------------------------------------------------------------
// Forward-mode derivatives (dual numbers)
// ----------------------------------------------------------------------------
//
// rpn_run_program_dual evaluates a program (decoded with RESOLVE_CONST=false,
// so constant tokens keep their slot) together with the derivative of the
// result with respect to constant slot `dir`. Values follow exactly the same
// strict/protected semantics as rpn_run_program; derivatives are those of the
// branch actually taken (clamped or protected branches have zero slope).

template <typename T>
__device__ __forceinline__ T rpn_digamma(T x) {
    T acc = (T)0;
    if (x < (T)0.5) {
        // Reflection: psi(1 - x) - psi(x) = pi * cot(pi * x)
        const T pix = (T)3.141592653589793 * x;
        acc = -(T)3.141592653589793 / tan(pix);
        x = (T)1 - x;
    }
    while (x < (T)6) { acc -= (T)1 / x; x += (T)1; }
    const T f = (T)1 / (x * x);
    acc += log(x) - (T)0.5 / x
         - f * ((T)(1.0 / 12) - f * ((T)(1.0 / 120) - f * ((T)(1.0 / 252) - f * ((T)(1.0 / 240) - f * (T)(1.0 / 132)))));
    return acc;
}

template <typename T>
__device__ __forceinline__ T rpn_sign(T v) {
    return (v > (T)0) ? (T)1 : ((v < (T)0) ? (T)-1 : (T)0);
}

// a^b with derivative; `b`/`db` hold the exponent on entry and the result on exit.
template <typename T, bool STRICT>
__device__ __forceinline__ void rpn_dual_pow(T a, T da, T &b, T &db, bool &err) {
    const T res = STRICT ? strict_pow(a, b, err) : safe_pow(a, b, err);
    if (err) return;
    T d = (T)0;
    if (!(fabs(a) < (T)1e-10 && fabs(b) < (T)1e-10)) {
        T ea = a, eda = da, eb = b, edb = db;
        if (ea < (T)0) {
            const T ib = round(eb);
            if (fabs(eb - ib) > (T)1e-3) { ea = -ea; eda = -eda; }  // protected: |a|
            else { eb = ib; edb = (T)0; }
        }
        if (!STRICT) {
            if (fabs(ea) > (T)1.0 && eb > (T)80.0) { eb = (T)80.0; edb = (T)0; }
            if (fabs(ea) > (T)100.0 && eb > (T)10.0) { eb = (T)10.0; edb = (T)0; }
        }
        T t = (T)0;
        if (eda != (T)0) t += eb * pow(ea, eb - (T)1) * eda;
        if (edb != (T)0 && ea > (T)0) t += res * log(ea) * edb;
        d = isfinite(t) ? t : (T)0;
    }
    b = res;
    db = d;
}

template <typename T, bool STRICT>
__device__ __forceinline__ bool rpn_run_program_dual(
    const unsigned char* __restrict__ code,
    const unsigned char* __restrict__ aux,
    const T* __restrict__ imm,
    int len,
    const T* __restrict__ x, long long D, long long d,
    const T* __restrict__ consts,
    int dir,
    T &out, T &dout
) {
    T sv[RPN_EVAL_STACK];
    T sd[RPN_EVAL_STACK];
    int sp = 0;
    T v = (T)0, dv = (T)0;
    bool err = false;
    for (int pc = 0; pc < len; ++pc) {
        switch (code[pc]) {
            case RC_IMM: sv[sp] = v; sd[sp] = dv; ++sp; v = imm[pc]; dv = (T)0; break;
            case RC_VAR: sv[sp] = v; sd[sp] = dv; ++sp; v = x[(long long)aux[pc] * D + d]; dv = (T)0; break;
            case RC_CONST:
                sv[sp] = v; sd[sp] = dv; ++sp;
                v = consts[aux[pc]];
                dv = ((int)aux[pc] == dir) ? (T)1 : (T)0;
                break;
            case RC_ADD: --sp; v = sv[sp] + v; dv = sd[sp] + dv; break;
            case RC_SUB: --sp; v = sv[sp] - v; dv = sd[sp] - dv; break;
            case RC_MUL: { --sp; const T a = sv[sp], da = sd[sp]; dv = da * v + a * dv; v = a * v; break; }
            case RC_DIV: {
                --sp;
                const T a = sv[sp], da = sd[sp];
                if (fabs(v) < (T)1e-9) {
                    if (STRICT) err = true;
                    else { v = a; dv = da; }
                } else {
                    const T r = a / v;
                    dv = (da - r * dv) / v;
                    v = r;
                }
                break;
            }
            case RC_POW: { --sp; rpn_dual_pow<T, STRICT>(sv[sp], sd[sp], v, dv, err); break; }
            case RC_MOD: {
                --sp;
                const T a = sv[sp], da = sd[sp];
                const T r = STRICT ? strict_mod(a, v, err) : safe_mod(a, v, err);
                if (fabs(v) < (T)1e-9) dv = (T)0;
                else dv = da - round((a - r) / v) * dv;
                v = r;
                break;
            }
            case RC_SIN: dv = cos(v) * dv; v = sin(v); break;
            case RC_COS: dv = -sin(v) * dv; v = cos(v); break;
            case RC_TAN: { const T t = tan(v); dv = ((T)1 + t * t) * dv; v = t; break; }
            case RC_ASIN:
            case RC_ACOS: {
                const bool outside = (v < (T)-1.0 || v > (T)1.0);
                const T r = (code[pc] == RC_ASIN)
                    ? (STRICT ? strict_asin(v, err) : safe_asin(v, err))
                    : (STRICT ? strict_acos(v, err) : safe_acos(v, err));
                const T den = sqrt(fmax((T)0, (T)1 - v * v));
                T g = (outside || den <= (T)0) ? (T)0 : dv / den;
                dv = (code[pc] == RC_ASIN) ? g : -g;
                v = r;
                break;
            }
            case RC_ATAN: dv = dv / ((T)1 + v * v); v = atan(v); break;
            case RC_LOG: {
                if (STRICT) {
                    const T r = strict_log(v, err);
                    dv = dv / v;
                    v = r;
                } else {
                    const T av = fabs(v) + (T)1e-9;
                    dv = rpn_sign(v) * dv / av;
                    v = log(av);
                }
                break;
            }
            case RC_EXP: {
                if (STRICT) {
                    const T r = strict_exp(v, err);
                    dv = r * dv;
                    v = r;
                } else {
                    const bool clamped = (v < (T)-80.0 || v > (T)80.0);
                    const T r = safe_exp(v, err);
                    dv = clamped ? (T)0 : r * dv;
                    v = r;
                }
                break;
            }
            case RC_SQRT: {
                const T r = STRICT ? strict_sqrt(v, err) : safe_sqrt(v, err);
                const T s = STRICT ? (T)1 : rpn_sign(v);
                dv = (r > (T)0) ? s * dv / ((T)2 * r) : (T)0;
                v = r;
                break;
            }
            case RC_ABS: dv = rpn_sign(v) * dv; v = fabs(v); break;
            case RC_NEG: v = -v; dv = -dv; break;
            case RC_FLOOR: v = floor(v); dv = (T)0; break;
            case RC_CEIL: v = ceil(v); dv = (T)0; break;
            case RC_SIGN: v = rpn_sign(v); dv = (T)0; break;
            case RC_FACT:
            case RC_GAMMA: {
                const T arg = (code[pc] == RC_FACT) ? v + (T)1.0 : v;
                const T r = STRICT ? strict_tgamma(arg, err) : safe_tgamma(arg, err);
                dv = (r != (T)0) ? r * rpn_digamma(arg) * dv : (T)0;
                v = r;
                break;
            }
            case RC_LGAMMA: {
                const T r = STRICT ? strict_lgamma(v, err) : safe_lgamma(v, err);
                const bool pole = (v <= (T)0.0 && floor(v) == v);
                dv = pole ? (T)0 : rpn_digamma(v) * dv;
                v = r;
                break;
            }
            default: err = true; break;
        }
        if (err) return false;
    }
    if (!isfinite(dv)) dv = (T)0;
    out = v;
    dout = dv;
    return true;
}

// ----------------------------------------------------------------------------
// Reverse mode (adjoints) on a value tape
// ----------------------------------------------------------------------------
//
// Every node of a decoded program writes its value to tape[pc]; operands are
// read through precomputed child positions instead of a stack. A backward
// sweep then yields the derivative of the result with respect to every
// constant slot at once, so a Jacobian row costs two passes whatever the
// number of constants (forward mode needs one pass per constant).

#define RPN_NO_CHILD 255

__device__ __forceinline__ int rpn_code_arity(unsigned char c) {
    if (c == RC_IMM || c == RC_VAR || c == RC_CONST) return 0;
    if (c >= RC_ADD && c <= RC_MOD) return 2;
    return 1;
}

// Child tape positions of every node of a valid program (one thread).
// Positions fit in uint8, so programs longer than 255 tokens are not supported.
__device__ __forceinline__ void rpn_program_children(
    const unsigned char* __restrict__ code, int len,
    unsigned char* __restrict__ lc, unsigned char* __restrict__ rc
) {
    unsigned char st[RPN_EVAL_STACK];
    int sp = 0;
    for (int pc = 0; pc < len; ++pc) {
        const int ar = rpn_code_arity(code[pc]);
        if (ar == 2) {
            rc[pc] = st[--sp];
            lc[pc] = st[--sp];
        } else if (ar == 1) {
            lc[pc] = st[--sp];
            rc[pc] = RPN_NO_CHILD;
        } else {
            lc[pc] = RPN_NO_CHILD;
            rc[pc] = RPN_NO_CHILD;
        }
        st[sp++] = (unsigned char)pc;
    }
}

// Forward sweep: same values and error semantics as rpn_run_program.
template <typename T, bool STRICT>
__device__ __forceinline__ bool rpn_tape_forward(
    const unsigned char* __restrict__ code,
    const unsigned char* __restrict__ aux,
    const T* __restrict__ imm,
    const unsigned char* __restrict__ lc,
    const unsigned char* __restrict__ rc,
    int len,
    const T* __restrict__ x, long long D, long long d,
    const T* __restrict__ consts,
    T* __restrict__ tape
) {
    bool err = false;
    for (int pc = 0; pc < len; ++pc) {
        T v;
        switch (code[pc]) {
            case RC_IMM: v = imm[pc]; break;
            case RC_VAR: v = x[(long long)aux[pc] * D + d]; break;
            case RC_CONST: v = consts[aux[pc]]; break;
            case RC_ADD: v = tape[lc[pc]] + tape[rc[pc]]; break;
            case RC_SUB: v = tape[lc[pc]] - tape[rc[pc]]; break;
            case RC_MUL: v = tape[lc[pc]] * tape[rc[pc]]; break;
            case RC_DIV: v = STRICT ? strict_div(tape[lc[pc]], tape[rc[pc]], err) : safe_div(tape[lc[pc]], tape[rc[pc]], err); break;
            case RC_POW: v = STRICT ? strict_pow(tape[lc[pc]], tape[rc[pc]], err) : safe_pow(tape[lc[pc]], tape[rc[pc]], err); break;
            case RC_MOD: v = STRICT ? strict_mod(tape[lc[pc]], tape[rc[pc]], err) : safe_mod(tape[lc[pc]], tape[rc[pc]], err); break;
            case RC_SIN: v = sin(tape[lc[pc]]); break;
            case RC_COS: v = cos(tape[lc[pc]]); break;
            case RC_TAN: v = tan(tape[lc[pc]]); break;
            case RC_ASIN: v = STRICT ? strict_asin(tape[lc[pc]], err) : safe_asin(tape[lc[pc]], err); break;
            case RC_ACOS: v = STRICT ? strict_acos(tape[lc[pc]], err) : safe_acos(tape[lc[pc]], err); break;
            case RC_ATAN: v = atan(tape[lc[pc]]); break;
            case RC_LOG: v = STRICT ? strict_log(tape[lc[pc]], err) : safe_log(tape[lc[pc]], err); break;
            case RC_EXP: v = STRICT ? strict_exp(tape[lc[pc]], err) : safe_exp(tape[lc[pc]], err); break;
            case RC_SQRT: v = STRICT ? strict_sqrt(tape[lc[pc]], err) : safe_sqrt(tape[lc[pc]], err); break;
            case RC_ABS: v = fabs(tape[lc[pc]]); break;
            case RC_NEG: v = -tape[lc[pc]]; break;
            case RC_FLOOR: v = floor(tape[lc[pc]]); break;
            case RC_CEIL: v = ceil(tape[lc[pc]]); break;
            case RC_SIGN: v = rpn_sign(tape[lc[pc]]); break;
            case RC_FACT: v = STRICT ? strict_tgamma(tape[lc[pc]] + (T)1.0, err) : safe_tgamma(tape[lc[pc]] + (T)1.0, err); break;
            case RC_GAMMA: v = STRICT ? strict_tgamma(tape[lc[pc]], err) : safe_tgamma(tape[lc[pc]], err); break;
            case RC_LGAMMA: v = STRICT ? strict_lgamma(tape[lc[pc]], err) : safe_lgamma(tape[lc[pc]], err); break;
            default: err = true; v = (T)0; break;
        }
        if (err) return false;
        tape[pc] = v;
    }
    return true;
}

// Partial derivatives of a^b for the branch rpn's pow semantics take.
template <typename T, bool STRICT>
__device__ __forceinline__ void rpn_pow_partials(T a, T b, T res, T &da, T &db) {
    da = (T)0;
    db = (T)0;
    if (fabs(a) < (T)1e-10 && fabs(b) < (T)1e-10) return;
    T ea = a, sa = (T)1, eb = b;
    bool b_free = true;
    if (ea < (T)0) {
        const T ib = round(eb);
        if (fabs(eb - ib) > (T)1e-3) { ea = -ea; sa = (T)-1; }  // protected: |a|
        else { eb = ib; b_free = false; }
    }
    if (!STRICT) {
        if (fabs(ea) > (T)1.0 && eb > (T)80.0) { eb = (T)80.0; b_free = false; }
        if (fabs(ea) > (T)100.0 && eb > (T)10.0) { eb = (T)10.0; b_free = false; }
    }
    const T pa = sa * eb * pow(ea, eb - (T)1);
    da = isfinite(pa) ? pa : (T)0;
    if (b_free && ea > (T)0) {
        const T pb = res * log(ea);
        db = isfinite(pb) ? pb : (T)0;
    }
}

// Backward sweep over a tape filled by rpn_tape_forward. grad[s] receives the
// derivative of the result with respect to constant slot s (s < n_grad).
template <typename T, bool STRICT>
__device__ __forceinline__ void rpn_tape_backward(
    const unsigned char* __restrict__ code,
    const unsigned char* __restrict__ aux,
    const unsigned char* __restrict__ lc,
    const unsigned char* __restrict__ rc,
    int len,
    const T* __restrict__ tape,
    T* __restrict__ adj,
    T* __restrict__ grad, int n_grad
) {
    for (int i = 0; i < len; ++i) adj[i] = (T)0;
    for (int j = 0; j < n_grad; ++j) grad[j] = (T)0;
    adj[len - 1] = (T)1;
    for (int pc = len - 1; pc >= 0; --pc) {
        const T g = adj[pc];
        if (g == (T)0) continue;
        const unsigned char c = code[pc];
        if (c == RC_CONST) {
            const int s = aux[pc];
            if (s < n_grad) grad[s] += g;
            continue;
        }
        if (c == RC_IMM || c == RC_VAR) continue;
        const int l = lc[pc];
        const T in = tape[l];
        const T out = tape[pc];
        T pl = (T)0, pr = (T)0;
        switch (c) {
            case RC_ADD: pl = (T)1; pr = (T)1; break;
            case RC_SUB: pl = (T)1; pr = (T)-1; break;
            case RC_MUL: pl = tape[rc[pc]]; pr = in; break;
            case RC_DIV: {
                const T bv = tape[rc[pc]];
                if (fabs(bv) < (T)1e-9) { pl = (T)1; }      // protected branch returns a
                else { pl = (T)1 / bv; pr = -out / bv; }
                break;
            }
            case RC_POW: rpn_pow_partials<T, STRICT>(in, tape[rc[pc]], out, pl, pr); break;
            case RC_MOD: {
                const T bv = tape[rc[pc]];
                if (fabs(bv) >= (T)1e-9) { pl = (T)1; pr = -round((in - out) / bv); }
                break;
            }
            case RC_SIN: pl = cos(in); break;
            case RC_COS: pl = -sin(in); break;
            case RC_TAN: pl = (T)1 + out * out; break;
            case RC_ASIN:
            case RC_ACOS: {
                const T den = sqrt(fmax((T)0, (T)1 - in * in));
                const bool outside = (in < (T)-1.0 || in > (T)1.0);
                const T p = (outside || den <= (T)0) ? (T)0 : (T)1 / den;
                pl = (c == RC_ASIN) ? p : -p;
                break;
            }
            case RC_ATAN: pl = (T)1 / ((T)1 + in * in); break;
            case RC_LOG: pl = STRICT ? (T)1 / in : rpn_sign(in) / (fabs(in) + (T)1e-9); break;
            case RC_EXP:
                pl = (!STRICT && (in < (T)-80.0 || in > (T)80.0)) ? (T)0 : out;
                break;
            case RC_SQRT:
                pl = (out > (T)0) ? (STRICT ? (T)1 : rpn_sign(in)) / ((T)2 * out) : (T)0;
                break;
            case RC_ABS: pl = rpn_sign(in); break;
            case RC_NEG: pl = (T)-1; break;
            case RC_FACT: pl = (out != (T)0) ? out * rpn_digamma(in + (T)1) : (T)0; break;
            case RC_GAMMA: pl = (out != (T)0) ? out * rpn_digamma(in) : (T)0; break;
            case RC_LGAMMA: pl = (in <= (T)0 && floor(in) == in) ? (T)0 : rpn_digamma(in); break;
            default: break;  // floor, ceil, sign: zero slope
        }
        const T cl = g * pl;
        if (isfinite(cl)) adj[l] += cl;
        if (rpn_code_arity(c) == 2) {
            const T cr = g * pr;
            if (isfinite(cr)) adj[rc[pc]] += cr;
        }
    }
    for (int j = 0; j < n_grad; ++j) if (!isfinite(grad[j])) grad[j] = (T)0;
}
