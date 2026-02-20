
import torch
import math
from typing import Tuple
from .config import GpuGlobals
from .evaluation import GPUEvaluator
from .operators import GPUOperators

class RPNBackwardFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, population, constants, x, y_target, vm):
        # Forward pass: Evaluate formulas
        B = population.shape[0]
        K = constants.shape[1]
        D = x.shape[1]
        
        preds = torch.empty((B, D), device=constants.device, dtype=constants.dtype)
        sp_out = torch.empty((B, D), device=constants.device, dtype=torch.int32)
        error_out = torch.zeros((B, D), device=constants.device, dtype=torch.uint8)
        
        x_cuda = x
        if x.ndim == 2 and x.shape[0] == D and x.shape[1] != D:
            x_cuda = x.T.contiguous()
        elif not x.is_contiguous():
            x_cuda = x.contiguous()
            
        import rpn_cuda_native
        
        rpn_cuda_native.eval_rpn(
            population.contiguous(),
            x_cuda,
            constants.contiguous(),
            preds,
            sp_out,
            error_out,
            vm.PAD_ID, vm.id_x_start,
            vm.id_C, vm.id_pi, vm.id_e,
            vm.id_0, vm.id_1, vm.id_2, vm.id_3, vm.id_4, vm.id_5, vm.id_6, vm.id_10,
            vm.op_add, vm.op_sub, vm.op_mul, vm.op_div, vm.op_pow, vm.op_mod,
            vm.op_sin, vm.op_cos, vm.op_tan,
            vm.op_log, vm.op_exp,
            vm.op_sqrt, vm.op_abs, vm.op_neg,
            vm.op_fact, vm.op_floor, vm.op_ceil, vm.op_sign,
            vm.op_gamma, vm.op_lgamma,
            vm.op_asin, vm.op_acos, vm.op_atan,
            math.pi, math.e,
            0 # strict mode false
        )
        
        ctx.save_for_backward(population, constants, x_cuda)
        ctx.vm = vm
        return preds

    @staticmethod
    def backward(ctx, grad_output):
        population, constants, x_cuda = ctx.saved_tensors
        vm = ctx.vm
        
        grad_constants = torch.zeros_like(constants)
        import rpn_cuda_native
        
        rpn_cuda_native.eval_rpn_backward(
            population.contiguous(),
            x_cuda,
            constants.contiguous(),
            grad_output.contiguous(),
            grad_constants,
            vm.PAD_ID, vm.id_x_start,
            vm.id_C, vm.id_pi, vm.id_e,
            vm.id_0, vm.id_1, vm.id_2, vm.id_3, vm.id_4, vm.id_5, vm.id_6, vm.id_10,
            vm.op_add, vm.op_sub, vm.op_mul, vm.op_div, vm.op_pow, vm.op_mod,
            vm.op_sin, vm.op_cos, vm.op_tan,
            vm.op_log, vm.op_exp,
            vm.op_sqrt, vm.op_abs, vm.op_neg,
            vm.op_fact, vm.op_floor, vm.op_ceil, vm.op_sign,
            vm.op_gamma, vm.op_lgamma,
            vm.op_asin, vm.op_acos, vm.op_atan,
            math.pi, math.e
        )
        return None, grad_constants, None, None, None

def evaluate_with_gradients(population, constants, x, y_target, vm):
    """Encapsulates the PyTorch function application."""
    preds = RPNBackwardFunction.apply(population, constants, x, y_target, vm)
    diff = preds - y_target.unsqueeze(0)
    mse = (diff ** 2).mean(dim=1)
    loss = mse.sum()
    return loss, torch.sqrt(mse.detach())

class GPUOptimizer:
    def __init__(self, evaluator: GPUEvaluator, operators: GPUOperators, device, dtype=torch.float64):
        self.evaluator = evaluator
        self.operators = operators
        self.device = device
        self.dtype = dtype
        try:
            import rpn_cuda_native
            self._has_fused_pso = hasattr(rpn_cuda_native, 'fused_pso')
            self._rpn_cuda = rpn_cuda_native if self._has_fused_pso else None
        except ImportError:
            self._has_fused_pso = False
            self._rpn_cuda = None



    def optimize_constants(self, population: torch.Tensor, constants: torch.Tensor, x: torch.Tensor, y_target: torch.Tensor, steps=10, lr=0.1):
        """
        Refine constants using Gradient Descent (Adam).
        Returns: (best_constants, best_rmse)
        """
        optimized_consts = constants.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([optimized_consts], lr=lr)
        
        best_mse = torch.full((population.shape[0],), float('inf'), device=self.device, dtype=self.dtype)
        best_consts = constants.clone().detach() 
        
        for _ in range(steps):
            optimizer.zero_grad()
            
            # Use differentiable evaluation (returns (Loss, Preds))
            loss_per_ind, _ = self.evaluator.evaluate_differentiable(population, optimized_consts, x, y_target)
            
            # Track best (unconditional update — no GPU sync)
            current_mse = loss_per_ind.detach()
            valid = ~torch.isnan(current_mse)
            improved = (current_mse < best_mse) & valid
            best_mse = torch.where(improved, current_mse, best_mse)
            best_consts = torch.where(improved.unsqueeze(1), optimized_consts.detach(), best_consts)
                
            loss = loss_per_ind[valid].sum()
            
            if not loss.requires_grad: 
                break
                
            loss.backward()
            optimizer.step()
            
        return best_consts, torch.sqrt(best_mse)

    def nano_pso(self, population: torch.Tensor, constants: torch.Tensor, x: torch.Tensor, y: torch.Tensor, 
                steps: int = 20, num_particles: int = 20, w: float = 0.5, c1: float = 1.5, c2: float = 1.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Particle Swarm Optimization (Gradient-Free) for constants.
        Uses fused CUDA kernel when available (single launch for entire PSO loop),
        falls back to multi-kernel approach otherwise.
        """
        # Try fused kernel first (single CUDA launch, ~5-10x faster for PSO portion).
        # The fused kernel supports both float32 and float64 via AT_DISPATCH_FLOATING_TYPES.
        if self._has_fused_pso:
            return self._fused_nano_pso(population, constants, x, y, steps, num_particles, w, c1, c2)
        
        return self._multi_kernel_nano_pso(population, constants, x, y, steps, num_particles, w, c1, c2)

    def lbfgs_optimize_top_k(self, population: torch.Tensor, constants: torch.Tensor,
                              x: torch.Tensor, y_target: torch.Tensor,
                              top_k: int = 50, max_iter: int = 30) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimización SOTA de constantes vía L-BFGS-B (Native CUDA Autograd).
        
        Reemplaza las Diferencias Finitas emuladas: ahora provee derivadas parciales EXACTAS 
        y de alta velocidad desde C++. Utiliza el optimizador real de PyTorch L-BFGS
        para encontrar los coeficientes sub-escala.
        """
        B, K = constants.shape
        if K == 0 or top_k <= 0 or B == 0 or not self._has_fused_pso:
            return constants, torch.full((B,), float('inf'), device=self.device, dtype=self.dtype)

        actual_k = min(top_k, B)
        pop_k = population[:actual_k].contiguous()
        c = constants[:actual_k].clone().to(self.dtype)
        
        best_errs_k = self.evaluator.evaluate_batch(pop_k, x, y_target, c)
        best_c_k = c.clone()
        
        # PyTorch L-BFGS requires require_grad=True
        c.requires_grad_(True)
        
        # High precision Newton Optimizer
        optimizer = torch.optim.LBFGS(
            [c],
            lr=1.0,  # Line search handles the true step
            max_iter=max_iter,
            max_eval=max_iter * 2,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            history_size=10,
            line_search_fn="strong_wolfe"
        )
        
        vm = self.evaluator.vm
        current_errs_k = None
        
        def closure():
            nonlocal current_errs_k
            optimizer.zero_grad()
            
            # Predict & Accumulate Gradients via Native C++ Reverse AD
            loss, rmses = evaluate_with_gradients(pop_k, c, x, y_target, vm)
            
            if loss.requires_grad:
                loss.backward()
            
            # Clamp gradients for stability if needed
            if c.grad is not None:
                c.grad.data.clamp_(-1e4, 1e4)
                
            current_errs_k = rmses
            return loss

        # Run optimization
        try:
            optimizer.step(closure)
        except Exception as e:
            # Fallback if line-search or domain restrictions blow up
            pass
            
        c_final = c.detach().clamp(GpuGlobals.CONSTANT_MIN_VALUE, GpuGlobals.CONSTANT_MAX_VALUE)
        final_errs_k = self.evaluator.evaluate_batch(pop_k, x, y_target, c_final)
        
        improved = torch.isfinite(final_errs_k) & (final_errs_k < best_errs_k)
        best_c_k[improved] = c_final[improved]
        best_errs_k[improved] = final_errs_k[improved]
        
        result_consts = constants.clone()
        result_consts[:actual_k] = best_c_k
        full_errs = torch.full((B,), float('inf'), device=self.device, dtype=self.dtype)
        full_errs[:actual_k] = best_errs_k
        
        return result_consts, full_errs

    def nano_pso_adaptive(self, population: torch.Tensor, constants: torch.Tensor, x: torch.Tensor, y: torch.Tensor, 
                steps: int = 20, num_particles: int = 20, w_max: float = 0.9, w_min: float = 0.4,
                c1: float = 1.5, c2: float = 1.5) -> tuple:
        """
        PSO con inercia adaptativa (IPSO): w decrece de w_max a w_min.
        Favorece exploración al inicio y explotación al final.
        Usa el mismo kernel fused pero con w calculado por step.
        """
        # Ejecutar en pasos individuales con w variable
        # Para el fused kernel: lo llamamos con steps=1 en loop (menos eficiente)
        # pero permite inercia adaptativa real.
        # Alternativa más eficiente: llamar fused con w promedio (aproximación)
        # El fused kernel ahora aplica decay lineal interno w_max→0.4.
        # Pasamos w_max directamente — no hace falta promediar.
        # El fallback multi-kernel sigue usando w_avg para compatibilidad.
        if self._has_fused_pso:
            return self._fused_nano_pso(population, constants, x, y, steps, num_particles, w_max, c1, c2)
        w_avg = (w_max + w_min) / 2.0
        return self._multi_kernel_nano_pso(population, constants, x, y, steps, num_particles, w_avg, c1, c2)

    def _fused_nano_pso(self, population: torch.Tensor, constants: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                       steps: int, num_particles: int, w: float, c1: float, c2: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fused PSO: entire PSO loop in a single CUDA kernel launch."""
        B, K = constants.shape
        
        # FIX: El kernel CUDA espera x en formato [Vars, D] (variables × datos).
        # El engine pasa x como [D, Vars] (datos × variables) → transponer.
        # Detectar por comparación con y.numel() (= número de puntos D).
        D_expected = y.numel()
        if x.ndim == 2 and x.shape[0] == D_expected and x.shape[1] != D_expected:
            # x es [D, Vars] → transponer a [Vars, D]
            x = x.T.contiguous()
        elif not x.is_contiguous():
            x = x.contiguous()
        
        # Pre-allocate outputs
        gbest_pos = torch.empty((B, K), device=self.device, dtype=self.dtype)
        gbest_err = torch.empty((B,), device=self.device, dtype=self.dtype)
        
        # Get opcode IDs from the evaluator's VM
        vm = self.evaluator.vm
        
        self._rpn_cuda.fused_pso(
            population.contiguous(),
            constants.contiguous(),
            x.contiguous(),
            y.contiguous(),
            gbest_pos,
            gbest_err,
            num_particles, steps,
            w, c1, c2,
            GpuGlobals.CONSTANT_MIN_VALUE, GpuGlobals.CONSTANT_MAX_VALUE,
            # OpCode IDs
            vm.PAD_ID, vm.id_x_start,
            vm.id_C, vm.id_pi, vm.id_e,
            vm.id_0, vm.id_1, vm.id_2, vm.id_3, vm.id_4, vm.id_5, vm.id_6, vm.id_10,
            vm.op_add, vm.op_sub, vm.op_mul, vm.op_div, vm.op_pow, vm.op_mod,
            vm.op_sin, vm.op_cos, vm.op_tan,
            vm.op_log, vm.op_exp,
            vm.op_sqrt, vm.op_abs, vm.op_neg,
            vm.op_fact, vm.op_floor, vm.op_ceil, vm.op_sign,
            vm.op_gamma, vm.op_lgamma,
            vm.op_asin, vm.op_acos, vm.op_atan,
            math.pi, math.e
        )
        
        return gbest_pos, gbest_err

    def _multi_kernel_nano_pso(self, population: torch.Tensor, constants: torch.Tensor, x: torch.Tensor, y: torch.Tensor, 
                steps: int = 20, num_particles: int = 20, w: float = 0.5, c1: float = 1.5, c2: float = 1.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Particle Swarm Optimization (Gradient-Free) for constants.
        Runs 'num_particles' for EACH individual in 'population' in parallel.
        
        Args:
           population: [B, L]
           constants: [B, K] (Initial guess)
           x, y: Data
        
        Returns:
           refined_constants: [B, K]
           refined_errors: [B]
        """
        B, K = constants.shape
        # Replicate population for particles
        # We process B * num_particles individuals
        
        # 1. Expand Population indices
        # We don't need to copy full formula tensors if we use indexing trick in evaluator?
        # But evaluator expects [Total, L].
        # For simplicity, let's expand. [B, L] -> [B*P, L]
        # B=200, P=20 -> 4000. Cheap.
        
        pop_expanded = population.repeat_interleave(num_particles, dim=0) # [B*P, L]
        
        # 2. Init Particles
        # First particle is the original guess (elitism within swarm)
        # Others are random perturbations
        
        # [B, P, K]
        # Particle 0: exact copy
        # Particle 1..P: perturbed
        
        pos = constants.unsqueeze(1).repeat(1, num_particles, 1) # [B, P, K]
        # Jitter
        noise = torch.randn(B, num_particles - 1, K, device=self.device, dtype=self.dtype) * 1.0
        pos[:, 1:, :] += noise
        
        vel = torch.randn_like(pos) * 0.1
        vel = vel.reshape(-1, K) # [B*P, K]
        
        # Flatten for batch evaluation
        flat_pos = pos.reshape(-1, K) # [B*P, K]
        
        # Best Memory
        pbest_pos = flat_pos.clone()
        pbest_err = torch.full((B * num_particles,), float('inf'), device=self.device, dtype=self.dtype)
        
        # Global Best (per swarm/individual)
        # We store this as [B, K] and [B] err
        gbest_pos = constants.clone()
        gbest_err = torch.full((B,), float('inf'), device=self.device, dtype=self.dtype)
        
        # P1-1: Detect CUDA availability ONCE before loop
        try:
            import rpn_cuda_native
            _use_cuda_pso = True
        except ImportError:
            _use_cuda_pso = False
        
        # Loop
        for step in range(steps):
            # 3. Evaluate Batch
            # shape [B*P]
            errors = self.evaluator.evaluate_batch(pop_expanded, x, y, flat_pos)
            
            # 4. & 5. Update Bests (CUDA or PyTorch)
            if _use_cuda_pso:
                # Resize errors to [B, P] for kernel
                curr_err_view = errors.view(B, num_particles)
                pbest_err_view = pbest_err.view(B, num_particles)
                
                # Reshape pos to [B, P, K]
                pbest_pos_view = pbest_pos.view(B, num_particles, K)
                curr_pos_view = flat_pos.view(B, num_particles, K)
                
                rpn_cuda_native.pso_update_bests(
                    curr_err_view, pbest_err_view, pbest_pos_view, curr_pos_view,
                    gbest_err, gbest_pos
                )
            else:
                # 4. Update Personal Bests
                improved = errors < pbest_err
                pbest_pos[improved] = flat_pos[improved]
                pbest_err[improved] = errors[improved]
                
                # 5. Update Global Bests
                # Reshape to [B, P]
                reshaped_err = pbest_err.view(B, num_particles)
                min_errs, min_indices = torch.min(reshaped_err, dim=1) # [B]
                
                improved_g = min_errs < gbest_err
                if improved_g.any():
                    gbest_err[improved_g] = min_errs[improved_g]
                    
                    pbest_pos_view = pbest_pos.view(B, num_particles, K)
                    
                    # Expand indices to [B, 1, K] to gather
                    gather_idx = min_indices.view(B, 1, 1).expand(B, 1, K)
                    new_gbests = pbest_pos_view.gather(1, gather_idx).squeeze(1)
                    
                    gbest_pos[improved_g] = new_gbests[improved_g]
                
                
            # 6. PSO Update (CUDA or PyTorch)
            if _use_cuda_pso:
                # CUDA Fast Path
                # Generate random numbers on GPU
                r1_3d = torch.rand(B, num_particles, K, device=self.device, dtype=self.dtype)
                r2_3d = torch.rand(B, num_particles, K, device=self.device, dtype=self.dtype)
                
                # Reshape views for kernel (must be 3D [B, P, K])
                pos_3d = flat_pos.view(B, num_particles, K)
                vel_3d = vel.view(B, num_particles, K)
                pbest_3d = pbest_pos.view(B, num_particles, K)
                
                rpn_cuda_native.pso_update(
                    pos_3d, vel_3d, pbest_3d, gbest_pos, r1_3d, r2_3d,
                    w, c1, c2
                )
            else:
                 # PyTorch Fallback
                 r1 = torch.rand_like(flat_pos)
                 r2 = torch.rand_like(flat_pos)
                 
                 gbest_expanded = gbest_pos.repeat_interleave(num_particles, dim=0)
                 
                 vel = w * vel + c1 * r1 * (pbest_pos - flat_pos) + c2 * r2 * (gbest_expanded - flat_pos)
                 flat_pos += vel
            
            # Handle Bounds
            flat_pos.clamp_(GpuGlobals.CONSTANT_MIN_VALUE, GpuGlobals.CONSTANT_MAX_VALUE)

        return gbest_pos, gbest_err
