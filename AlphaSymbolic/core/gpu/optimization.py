
import torch
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
            
            # Track best
            current_mse = loss_per_ind.detach()
            # Handle potential NaNs in output
            valid = ~torch.isnan(current_mse)
            
            improved = (current_mse < best_mse) & valid
            if improved.any():
                best_mse[improved] = current_mse[improved]
                best_consts[improved] = optimized_consts[improved].detach()
                
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
        
        # Loop
        for step in range(steps):
            # 3. Evaluate Batch
            # shape [B*P]
            errors = self.evaluator.evaluate_batch(pop_expanded, x, y, flat_pos)
            
            # 4. & 5. Update Bests (CUDA or PyTorch)
            try:
                import rpn_cuda_native
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
            except ImportError:
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
                    # Get indices in flat array
                    
                    pbest_pos_view = pbest_pos.view(B, num_particles, K)
                    
                    # Expand indices to [B, 1, K] to gather
                    gather_idx = min_indices.view(B, 1, 1).expand(B, 1, K)
                    new_gbests = pbest_pos_view.gather(1, gather_idx).squeeze(1)
                    
                    gbest_pos[improved_g] = new_gbests[improved_g]
                
                
            # 6. PSO Update (CUDA or PyTorch)
            if hasattr(self.operators, 'RPN_CUDA_AVAILABLE') and self.operators.RPN_CUDA_AVAILABLE is False:
                 pass # Check flag from operators if module loaded?
                 # Actually easier: try import here or use checking
                 
            try:
                import rpn_cuda_native
                # CUDA Fast Path
                # Generate random numbers on GPU
                r1_3d = torch.rand(B, num_particles, K, device=self.device, dtype=self.dtype)
                r2_3d = torch.rand(B, num_particles, K, device=self.device, dtype=self.dtype)
                
                # Reshape views for kernel (must be 3D [B, P, K])
                pos_3d = flat_pos.view(B, num_particles, K)
                vel_3d = vel.view(B, num_particles, K)
                pbest_3d = pbest_pos.view(B, num_particles, K)
                
                # gbest_pos is [B, K], kernel handles broadcasting if logic allows, 
                # BUT pso_update_kernel signature takes `const scalar_t* gbest` and indexes it as `gbest[b * K + k]`.
                # This means gbest should be passed as [B, K] directly to C++, C++ treats it as flat buffer.
                # PyTorch check in C++ might require contiguous.
                
                rpn_cuda_native.pso_update(
                    pos_3d, vel_3d, pbest_3d, gbest_pos, r1_3d, r2_3d,
                    w, c1, c2
                )
            except ImportError:
                 # PyTorch Fallback
                 r1 = torch.rand_like(flat_pos)
                 r2 = torch.rand_like(flat_pos)
                 
                 # Broadcast Gbest [B, K] -> [B*P, K] of flattened view?
                 # gbest_pos is [B, K]. We need it for each particle.
                 # gbest_expanded = gbest_pos.repeat_interleave(num_particles, dim=0) -> [B*P, K]
                 # But wait, the kernel handles broadcasting inside.
                 # PyTorch fallback needs explicit expansion
                 gbest_expanded = gbest_pos.repeat_interleave(num_particles, dim=0)
                 
                 vel = w * vel + c1 * r1 * (pbest_pos - flat_pos) + c2 * r2 * (gbest_expanded - flat_pos)
                 flat_pos += vel
            
             # Handle Bounds? (Optional)
            
        return gbest_pos, gbest_err
