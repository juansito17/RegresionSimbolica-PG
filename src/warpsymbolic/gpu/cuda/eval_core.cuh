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
