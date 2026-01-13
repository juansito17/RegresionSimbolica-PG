
import torch
import torch.utils.cpp_extension

# Try to import the compiled extension
try:
    from . import rpn_cuda
except ImportError:
    try:
        import rpn_cuda
    except ImportError:
        rpn_cuda = None
        print("[CUDA VM] Warning: 'rpn_cuda' extension not found. Please compile it.")

class CudaRPNVM:
    def __init__(self, grammar, device):
        self.grammar = grammar
        self.device = device
        self._cache_ids()
        
    def _cache_ids(self):
        # Cache IDs standard
        g = self.grammar.token_to_id
        self.PAD_ID = g.get('<PAD>', -999)
        self.id_C = g.get('C', -100)
        self.id_pi = g.get('pi', -100)
        self.id_e = g.get('e', -100)
        
        self.op_add = g.get('+', -100)
        self.op_sub = g.get('-', -100)
        self.op_mul = g.get('*', -100)
        self.op_div = g.get('/', -100)
        self.op_pow = g.get('pow', -100)
        self.op_mod = g.get('%', -100)
        
        self.op_sin = g.get('sin', -100)
        self.op_cos = g.get('cos', -100)
        self.op_tan = g.get('tan', -100)
        self.op_asin = g.get('S', -100)
        self.op_acos = g.get('acos', -100)
        self.op_atan = g.get('T', -100)
        self.op_exp = g.get('exp', -100) # Ensure 'exp' vs 'e' usage
        self.op_log = g.get('log', -100)
        self.op_sqrt = g.get('sqrt', -100)
        self.op_abs = g.get('abs', -100)
        self.op_neg = g.get('neg', -100)
        
        self.op_fact = g.get('!', -100)
        self.op_floor = g.get('_', -100)
        self.op_gamma = g.get('g', -100)
        
        self.id_1 = g.get('1', -999)
        self.id_2 = g.get('2', -999)
        self.id_3 = g.get('3', -999)
        self.id_5 = g.get('5', -999)

        # Variables
        first_var = self.grammar.active_variables[0]
        self.id_x_start = g.get(first_var, -999)
        self.num_vars = len(self.grammar.active_variables)
        
    def eval(self, population: torch.Tensor, x: torch.Tensor, constants: torch.Tensor) -> tuple:
        """
        Evaluates population against x.
        population: [B, L]
        x: [Vars, Samples] (Optimized Layout)
        constants: [B, K] or None
        
        Returns: (preds [B, Samples], sp [B, Samples], error [B, Samples])
        """
        if rpn_cuda is None:
            raise RuntimeError("rpn_cuda module not loaded.")

        B, L = population.shape
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
            # Pass empty
            K = 0
            constants = torch.empty((B, 0), device=self.device, dtype=dtype)
        else:
            if not constants.is_contiguous(): constants = constants.contiguous()
            if constants.dtype != dtype: constants = constants.to(dtype)
            K = constants.shape[1]
            
        # Prepare Outputs
        # [B, D] - Implicitly expanded
        out_preds = torch.empty((B, D), dtype=dtype, device=self.device)
        out_sp = torch.empty((B, D), dtype=torch.int32, device=self.device)
        out_error = torch.empty((B, D), dtype=torch.uint8, device=self.device) # unsigned char in C++ map to bool/byte
        
        # Call Kernel
        rpn_cuda.eval_rpn(
            population, x, constants,
            out_preds, out_sp, out_error,
            self.PAD_ID, self.id_x_start,
            self.id_C, self.id_pi, self.id_e,
            self.id_1, self.id_2, self.id_3, self.id_5,
            self.op_add, self.op_sub, self.op_mul, self.op_div, self.op_pow, self.op_mod,
            self.op_sin, self.op_cos, self.op_tan,
            self.op_log, self.op_exp,
            self.op_sqrt, self.op_abs, self.op_neg,
            self.op_fact, self.op_floor, self.op_gamma,
            self.op_asin, self.op_acos, self.op_atan,
            3.14159265359, 2.718281828
        )
        
        return out_preds, out_sp, out_error.bool()
