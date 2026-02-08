
import torch
import math
from typing import Tuple
from .config import GpuGlobals
from .evaluation import GPUEvaluator
from .operators import GPUOperators

class GPUOptimizer:
    def __init__(self, evaluator: GPUEvaluator, operators: GPUOperators, device, dtype=torch.float64):
        self.evaluator = evaluator
        self.operators = operators
        self.device = device
        self.dtype = dtype
        # Detect fused PSO availability once
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
        # Try fused kernel first (single CUDA launch, ~5-10x faster for PSO portion)
        if self._has_fused_pso and self.dtype == torch.float32:
            return self._fused_nano_pso(population, constants, x, y, steps, num_particles, w, c1, c2)
        
        return self._multi_kernel_nano_pso(population, constants, x, y, steps, num_particles, w, c1, c2)

    def _fused_nano_pso(self, population: torch.Tensor, constants: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
                       steps: int, num_particles: int, w: float, c1: float, c2: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fused PSO: entire PSO loop in a single CUDA kernel launch."""
        B, K = constants.shape
        
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
            vm.id_0, vm.id_1, vm.id_2, vm.id_3, vm.id_5, vm.id_10,
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
