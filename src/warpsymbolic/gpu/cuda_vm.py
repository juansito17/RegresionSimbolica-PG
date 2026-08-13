
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
    # Hard limits compiled into rpn_eval_fused_kernel. Keep these next to the
    # Python dispatch so unsupported shapes never reach a kernel that would
    # otherwise truncate the program or silently replace x4+ with zero.
    FUSED_MAX_VARS = 4
    FUSED_MAX_L = 256
    FUSED_MAX_D = 1024

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
            3.14159265359, 2.718281828,
            strict_mode
        )
        
        return out_preds, out_sp, out_error

    def _launch_fused(self, population, x, constants, y_target, out_rmse, strict_mode, launch_mode):
        """Launch one native evaluator variant. launch_mode: 0=block, 1=warp."""
        if x.ndim != 2 or int(x.shape[0]) != self.num_vars:
            raise ValueError(
                f"x must have shape [{self.num_vars}, D] for this grammar"
            )
        rpn_cuda.eval_rpn_fused(
            population, x, constants, y_target, out_rmse,
            self.PAD_ID, self.id_x_start,
            self.id_C, self.id_pi, self.id_e,
            self.id_0, self.id_1, self.id_2, self.id_3, self.id_4, self.id_5, self.id_6, self.id_10,
            self.op_add, self.op_sub, self.op_mul, self.op_div, self.op_pow, self.op_mod,
            self.op_sin, self.op_cos, self.op_tan, self.op_log, self.op_exp,
            self.op_sqrt, self.op_abs, self.op_neg,
            self.op_fact, self.op_floor, self.op_ceil, self.op_sign,
            self.op_gamma, self.op_lgamma,
            self.op_asin, self.op_acos, self.op_atan,
            3.14159265359, 2.718281828,
            strict_mode, launch_mode
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
            and int(x.shape[1]) <= self.FUSED_MAX_D
        )

    def _select_eval_mode(self, population, x, constants, y_target, out_rmse, strict_mode):
        """Autotune once per representative workload and cache the fastest safe variant."""
        from .config import GpuGlobals

        requested = str(getattr(GpuGlobals, 'CUDA_EVAL_MODE', 'auto')).lower()
        D = int(x.shape[1])
        if requested == 'block' or D > 32:
            return 0
        if requested == 'warp':
            return 1
        if not bool(getattr(GpuGlobals, 'CUDA_AUTOTUNE', True)):
            return 0

        B, L = population.shape
        K = constants.shape[1] if constants.dim() > 1 else 0
        # Launch behavior changes at broad population scales, but exact B values
        # should not create an unbounded cache during partial evaluations.
        b_bucket = 1 << max(0, int(B - 1).bit_length())
        key = (population.device.index, str(x.dtype), b_bucket, int(D), int(L), int(K), int(strict_mode))
        cached = self._eval_mode_cache.get(key)
        if cached is not None:
            return cached

        # Tiny batches are latency-bound and not worth a synchronous tuning pass.
        if B < 4096:
            self._eval_mode_cache[key] = 0
            return 0

        timings = {}
        reference = None
        candidate = None
        for mode in (0, 1):
            self._launch_fused(population, x, constants, y_target, out_rmse, strict_mode, mode)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(3):
                self._launch_fused(population, x, constants, y_target, out_rmse, strict_mode, mode)
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
                   y_target: torch.Tensor, strict_mode: int = 0) -> torch.Tensor:
        """
        Fused eval: block-per-individual kernel — returns [B] RMSE directly.

        - 0 warp divergence (all threads in a block run the same program)
        - Program cached in shared memory (17× less global reads)
        - RMSE computed by warp shuffle inside kernel (no B*D intermediate buffer)

        population: [B, L]
        x:          [Vars, D]
        constants:  [B, K]
        y_target:   [D]
        Returns:    [B] RMSE float32
        """
        if rpn_cuda is None or not hasattr(rpn_cuda, 'eval_rpn_fused'):
            raise RuntimeError("eval_rpn_fused not available — recompile CUDA extension.")

        if not self.supports_fused_shape(population, x):
            raise ValueError(
                "eval_rpn_fused only supports at most "
                f"{self.FUSED_MAX_VARS} variables, programs of length "
                f"{self.FUSED_MAX_L}, and {self.FUSED_MAX_D} samples; "
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

        # Pre-allocate output (reuse across calls)
        rmse_key = ('fused', B, dtype)
        if rmse_key not in self._output_cache:
            self._output_cache[rmse_key] = torch.empty(B, dtype=dtype, device=self.device)
        out_rmse = self._output_cache[rmse_key]

        launch_mode = self._select_eval_mode(
            population, x, constants, y_target, out_rmse, strict_mode)
        self.last_eval_mode = 'warp' if launch_mode == 1 else 'block'
        self._launch_fused(population, x, constants, y_target, out_rmse, strict_mode, launch_mode)
        return out_rmse
