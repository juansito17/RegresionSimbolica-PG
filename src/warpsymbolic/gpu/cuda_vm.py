
import math
import os
import sys
import torch
from .cuda_loader import load_rpn_cuda_native

_CUDA_DIR = os.path.join(os.path.dirname(__file__), 'cuda')
if _CUDA_DIR not in sys.path:
    sys.path.insert(0, _CUDA_DIR)

try:
    rpn_cuda = load_rpn_cuda_native()
except ImportError:
    rpn_cuda = None
    print("[CUDA VM] Warning: 'rpn_cuda_native' extension not found. Please compile it.")

class CudaRPNVM:
    # Hard limits compiled into rpn_eval_fused_kernel. The decoded evaluator
    # reads variables straight from x and lets each thread stride over the
    # samples, so the only structural limit is the decoded program length
    # (and the uint8 vocabulary, which bounds the number of variables).
    FUSED_MAX_VARS = 255
    FUSED_MAX_L = 256
    FUSED_MAX_D = None

    def __init__(self, grammar, device):
        self.grammar = grammar
        self.device = device
        self._cache_ids()
        self._output_cache = {}  # P1-2: Pre-allocated output buffers keyed by (B, D, dtype)
        self._empty_constants_cache = {}
        self._eval_mode_cache = {}
        self.last_eval_mode = "block"
        
    def _cache_ids(self):
        # Cache IDs standard
        g = self.grammar.token_to_id
        self.PAD_ID = g.get('<PAD>', -999)
        self.id_C = g.get('C', -100)
        self.id_pi = g.get('pi', -100)
        self.id_e = g.get('e', -100)
        
        # Operators (using standard names from warpsymbolic.gpu.grammar)
        self.op_add = g.get('+', -100)
        self.op_sub = g.get('-', -100)
        self.op_mul = g.get('*', -100)
        self.op_div = g.get('/', -100)
        self.op_pow = g.get('pow', -100)
        self.op_mod = g.get('%', -100)
        
        self.op_sin = g.get('sin', -100)
        self.op_cos = g.get('cos', -100)
        self.op_tan = g.get('tan', -100)
        self.op_asin = g.get('asin', -100)
        self.op_acos = g.get('acos', -100)
        self.op_atan = g.get('atan', -100)
        self.op_exp = g.get('exp', -100)
        self.op_log = g.get('log', -100)
        self.op_sqrt = g.get('sqrt', -100)
        self.op_abs = g.get('abs', -100)
        self.op_neg = g.get('neg', -100)
        
        self.op_fact = g.get('fact', -100)
        self.op_floor = g.get('floor', -100)
        self.op_ceil = g.get('ceil', -100)
        self.op_sign = g.get('sign', -100)
        self.op_gamma = g.get('gamma', -100)
        self.op_lgamma = g.get('lgamma', -100)
        
        self.id_C = self.grammar.token_to_id.get('C', -1)
        self.id_0 = self.grammar.token_to_id.get('0', -1)
        self.id_1 = self.grammar.token_to_id.get('1', -1)
        self.id_2 = self.grammar.token_to_id.get('2', -1)
        self.id_3 = self.grammar.token_to_id.get('3', -1)
        self.id_4 = self.grammar.token_to_id.get('4', -1)
        self.id_5 = self.grammar.token_to_id.get('5', -1)
        self.id_6 = self.grammar.token_to_id.get('6', -1)
        self.id_10 = self.grammar.token_to_id.get('10', -1)

        # Variables
        first_var = self.grammar.active_variables[0]
        self.id_x_start = g.get(first_var, -999)
        self.num_vars = len(self.grammar.active_variables)
        
    def eval(self, population: torch.Tensor, x: torch.Tensor, constants: torch.Tensor, strict_mode: int = 0) -> tuple:
        """
        Evaluates population against x.
        population: [B, L]
        x: [Vars, Samples] (Optimized Layout)
        constants: [B, K] or None
        
        Returns: (preds [B, Samples], sp [B, Samples], error [B, Samples])
        """
        if rpn_cuda is None:
            raise RuntimeError("rpn_cuda module not loaded.")

        B, _ = population.shape
        num_vars, D = x.shape
        
        # Validation
        if num_vars != self.num_vars:
            # Maybe implicit single variable?
            pass
            
        # Ensure Inputs are contiguous
        if not population.is_contiguous(): population = population.contiguous()
        if not x.is_contiguous(): x = x.contiguous()
        
        # Infer dtype from input
        dtype = x.dtype
        
        if constants is None:
            empty_key = (B, dtype)
            if empty_key in self._empty_constants_cache:
                constants = self._empty_constants_cache[empty_key]
            else:
                constants = torch.empty((B, 0), device=self.device, dtype=dtype)
                self._empty_constants_cache[empty_key] = constants
        else:
            if not constants.is_contiguous(): constants = constants.contiguous()
            if constants.dtype != dtype: constants = constants.to(dtype)
            
        # Prepare Outputs — P1-2: Reuse pre-allocated buffers when sizes match
        cache_key = (B, D, dtype)
        if cache_key in self._output_cache:
            out_preds, out_sp, out_error = self._output_cache[cache_key]
        else:
            out_preds = torch.empty((B, D), dtype=dtype, device=self.device)
            out_sp = torch.empty((B, D), dtype=torch.int32, device=self.device)
            out_error = torch.empty((B, D), dtype=torch.uint8, device=self.device)
            self._output_cache[cache_key] = (out_preds, out_sp, out_error)
        
        # Call Kernel
        rpn_cuda.eval_rpn(
            population,
            x,
            constants,
            out_preds, out_sp, out_error,
            self.PAD_ID, self.id_x_start,
            self.id_C, self.id_pi, self.id_e,
            self.id_0, self.id_1, self.id_2, self.id_3, self.id_4, self.id_5, self.id_6, self.id_10,
            self.op_add, self.op_sub, self.op_mul, self.op_div, self.op_pow, self.op_mod,
            self.op_sin, self.op_cos, self.op_tan,
            self.op_log, self.op_exp,
            self.op_sqrt, self.op_abs, self.op_neg,
            self.op_fact, self.op_floor, self.op_ceil, self.op_sign,
            self.op_gamma, self.op_lgamma,
            self.op_asin, self.op_acos, self.op_atan,
            math.pi, math.e,
            strict_mode
        )
        
        return out_preds, out_sp, out_error

    def op_id_args(self):
        """Token ids in the positional order shared by the native kernels."""
        return (
            self.PAD_ID, self.id_x_start,
            self.id_C, self.id_pi, self.id_e,
            self.id_0, self.id_1, self.id_2, self.id_3, self.id_4, self.id_5, self.id_6, self.id_10,
            self.op_add, self.op_sub, self.op_mul, self.op_div, self.op_pow, self.op_mod,
            self.op_sin, self.op_cos, self.op_tan, self.op_log, self.op_exp,
            self.op_sqrt, self.op_abs, self.op_neg,
            self.op_fact, self.op_floor, self.op_ceil, self.op_sign,
            self.op_gamma, self.op_lgamma,
            self.op_asin, self.op_acos, self.op_atan,
            math.pi, math.e,
        )

    def _launch_fused(self, population, x, constants, y_target, out_rmse, strict_mode, launch_mode,
                      scaled=False, out_ab=None, reuse=None):
        """Launch one native evaluator variant. launch_mode: 0=block, 1=warp."""
        if x.ndim != 2 or int(x.shape[0]) != self.num_vars:
            raise ValueError(
                f"x must have shape [{self.num_vars}, D] for this grammar"
            )
        extra = {}
        if scaled:
            extra['scaled'] = 1
        if out_ab is not None:
            extra['out_ab'] = out_ab
        if reuse is not None:
            parent, old_pop, old_consts, old_fit = reuse
            extra.update(reuse_parent=parent, reuse_pop=old_pop,
                         reuse_consts=old_consts, reuse_fit=old_fit)
        rpn_cuda.eval_rpn_fused(
            population, x, constants, y_target, out_rmse,
            *self.op_id_args(),
            strict_mode, launch_mode, **extra
        )

    def supports_fused_shape(self, population: torch.Tensor, x: torch.Tensor) -> bool:
        """Return whether the compiled fused evaluator can represent the workload."""
        if population.ndim != 2 or x.ndim != 2:
            return False
        return (
            int(population.shape[0]) > 0
            and int(population.shape[1]) > 0
            and self.num_vars > 0
            and self.num_vars <= self.FUSED_MAX_VARS
            and int(x.shape[0]) == self.num_vars
            and int(x.shape[1]) > 0
            and int(population.shape[1]) <= self.FUSED_MAX_L
        )

    def _select_eval_mode(self, population, x, constants, y_target, out_rmse, strict_mode, scaled=False):
        """Autotune once per representative workload and cache the fastest safe variant."""
        from .config import GpuGlobals

        requested = str(getattr(GpuGlobals, 'CUDA_EVAL_MODE', 'auto')).lower()
        D = int(x.shape[1])
        if requested == 'block':
            return 0
        if requested == 'warp':
            return 1
        if not bool(getattr(GpuGlobals, 'CUDA_AUTOTUNE', True)):
            return 1

        B, L = population.shape
        K = constants.shape[1] if constants.dim() > 1 else 0
        # Launch behavior changes at broad population scales, but exact B values
        # should not create an unbounded cache during partial evaluations.
        b_bucket = 1 << max(0, int(B - 1).bit_length())
        key = (population.device.index, str(x.dtype), b_bucket, int(D), int(L), int(K),
               int(strict_mode), bool(scaled))
        cached = self._eval_mode_cache.get(key)
        if cached is not None:
            return cached

        # Tiny batches are latency-bound and not worth a synchronous tuning pass.
        # One warp per individual keeps every SM busy unless B is tiny and D huge.
        if B < 4096:
            mode = 0 if (B < 256 and D > 256) else 1
            self._eval_mode_cache[key] = mode
            return mode

        timings = {}
        reference = None
        candidate = None
        for mode in (0, 1):
            self._launch_fused(population, x, constants, y_target, out_rmse, strict_mode, mode, scaled)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(3):
                self._launch_fused(population, x, constants, y_target, out_rmse, strict_mode, mode, scaled)
            end.record()
            end.synchronize()
            timings[mode] = start.elapsed_time(end)
            if mode == 0:
                reference = out_rmse.clone()
            else:
                candidate = out_rmse.clone()

        # A launch variant must preserve both numeric results and invalid/overflow
        # classification. Any disagreement selects the conservative block path.
        same_class = torch.equal(reference >= 1e14, candidate >= 1e14)
        numerically_equal = torch.allclose(reference, candidate, rtol=2e-5, atol=2e-5)
        selected = 1 if same_class and numerically_equal and timings[1] < timings[0] else 0
        self._eval_mode_cache[key] = selected
        return selected

    def eval_fused(self, population: torch.Tensor, x: torch.Tensor, constants: torch.Tensor,
                   y_target: torch.Tensor, strict_mode: int = 0, scaled: bool = False,
                   return_ab: bool = False, reuse=None):
        """
        Fused eval — returns [B] RMSE directly.

        - Each program is decoded once per individual (not once per sample)
        - 0 warp divergence (all threads of a warp run the same program)
        - Any number of variables and samples (threads stride over samples)
        - RMSE computed by warp shuffle inside kernel (no B*D intermediate buffer)

        population: [B, L]
        x:          [Vars, D]
        constants:  [B, K]
        y_target:   [D]
        scaled:     RMSE of the least squares fit a + b*f (linear scaling)
        return_ab:  also return the [B, 2] (a, b) coefficients (1, 0 when not scaled)
        reuse:      optional (parent_idx [B] int64, parent_pop [P, L], parent_consts [P, K],
                    parent_fitness [P]); a row bit-identical to its parent's row copies
                    the parent's fitness instead of being evaluated
        Returns:    [B] RMSE (and [B, 2] coefficients when return_ab)
        """
        if rpn_cuda is None or not hasattr(rpn_cuda, 'eval_rpn_fused'):
            raise RuntimeError("eval_rpn_fused not available — recompile CUDA extension.")

        if not self.supports_fused_shape(population, x):
            raise ValueError(
                "eval_rpn_fused only supports programs of length "
                f"<= {self.FUSED_MAX_L} with x shaped [{self.num_vars}, D]; "
                "use eval() for the classic safe path."
            )

        B = population.shape[0]
        dtype = x.dtype

        if not population.is_contiguous():  population = population.contiguous()
        if not x.is_contiguous():           x = x.contiguous()
        if not y_target.is_contiguous():    y_target = y_target.contiguous()

        if constants is None:
            key = (B, dtype)
            if key not in self._empty_constants_cache:
                self._empty_constants_cache[key] = torch.empty((B, 0), device=self.device, dtype=dtype)
            constants = self._empty_constants_cache[key]
        else:
            if not constants.is_contiguous(): constants = constants.contiguous()
            if constants.dtype != dtype:      constants = constants.to(dtype)

        if reuse is not None:
            if return_ab:
                raise ValueError("return_ab cannot be combined with fitness reuse")
            parent, old_pop, old_consts, old_fit = reuse
            parent = parent.to(device=population.device, dtype=torch.long).contiguous()
            old_pop = old_pop.contiguous()
            old_fit = old_fit.to(dtype).contiguous()
            if constants.shape[1] > 0:
                old_consts = old_consts.to(dtype).contiguous()
            else:
                old_consts = constants
            if (parent.numel() != B or old_pop.shape[1] != population.shape[1]
                    or old_fit.numel() != old_pop.shape[0]
                    or (constants.shape[1] > 0 and tuple(old_consts.shape) != (old_pop.shape[0], constants.shape[1]))):
                reuse = None
            else:
                reuse = (parent, old_pop, old_consts, old_fit)

        # A fresh output per call: callers keep fitness tensors across later
        # evaluations, so a shared cached buffer would be silently overwritten.
        # The caching allocator makes this allocation essentially free.
        out_rmse = torch.empty(B, dtype=dtype, device=self.device)
        out_ab = torch.empty((B, 2), dtype=dtype, device=self.device) if return_ab else None

        launch_mode = self._select_eval_mode(
            population, x, constants, y_target, out_rmse, strict_mode, scaled)
        self.last_eval_mode = 'warp' if launch_mode == 1 else 'block'
        self._launch_fused(population, x, constants, y_target, out_rmse, strict_mode, launch_mode,
                           scaled=scaled, out_ab=out_ab, reuse=reuse)
        if return_ab:
            return out_rmse, out_ab
        return out_rmse

    def lm_optimize(self, population: torch.Tensor, constants: torch.Tensor, x: torch.Tensor,
                    y_target: torch.Tensor, max_iter: int, const_min: float, const_max: float,
                    strict_mode: int = 1, scaled: bool = False):
        """Levenberg-Marquardt refinement of the constants of every row.

        Returns (constants [B, K], rmse [B]); invalid programs keep their constants
        and report 1e30.
        """
        if rpn_cuda is None or not hasattr(rpn_cuda, 'lm_optimize'):
            raise RuntimeError("lm_optimize not available — recompile CUDA extension.")
        dtype = x.dtype
        population = population.contiguous()
        x = x.contiguous()
        y_target = y_target.reshape(-1).to(dtype).contiguous()
        constants = constants.to(dtype).contiguous()
        out_consts = torch.empty_like(constants)
        out_rmse = torch.empty(population.shape[0], dtype=dtype, device=self.device)
        rpn_cuda.lm_optimize(
            population, constants, x, y_target, out_consts, out_rmse,
            int(max_iter), float(const_min), float(const_max),
            *self.op_id_args(),
            int(strict_mode), int(bool(scaled)))
        return out_consts, out_rmse
