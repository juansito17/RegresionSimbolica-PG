"""
Library Learning — SOTA P2 Feature
====================================
Identifies frequently occurring, high-fitness subtrees (building blocks)
from the population and stores them in a reusable library. These blocks can
then be injected into the mutation bank or used as structural initializers.

Inspired by: LaSR (Li et al. 2024), GP library learning, ALP (Automated
Library-based Programming).

Design:
  - GPU-native subtree hashing (no Python loops per individual)
  - Fixed-size hash-table library (circular buffer keyed by hash)
  - Fitness-weighted storage: only high-quality subtrees are kept
  - Sampling: operator can sample blocks for injection into mutation bank

Usage in engine.py:
    library = LibraryLearner(grammar, pop_size, max_len, device, dtype)
    # Each N gens:
    library.update(population, constants, fitness_rmse)
    # At mutation bank refresh:
    library_blocks = library.sample(k=64)
"""

import torch
import random
from typing import Optional
from .grammar import PAD_ID, GPUGrammar
from .config import GpuGlobals


class LibraryLearner:
    """
    GPU-native building-block library for symbolic regression.
    
    Tracks short subtrees extracted from high-fitness individuals
    and makes them available for re-injection into the mutation bank.
    
    The library is a fixed-capacity hash table:
      - Key:   polyhash of the subtree token sequence (mod capacity)
      - Value: (subtree_tokens, fitness_score, count)
    Only subtrees shorter than MAX_BLOCK_LEN are stored.
    Eviction policy: replace if new entry has better fitness.
    """

    def __init__(
        self,
        grammar: GPUGrammar,
        pop_size: int,
        max_len: int,
        device,
        dtype=torch.float32,
        capacity: int = 512,
    ):
        self.grammar = grammar
        self.pop_size = pop_size
        self.max_len = max_len
        self.device = device
        self.dtype = dtype
        self.capacity = capacity

        # Config
        self.max_block_len = int(getattr(GpuGlobals, 'LIBRARY_MAX_BLOCK_LEN', 8))
        self.top_k_fraction = float(getattr(GpuGlobals, 'LIBRARY_TOP_K_FRACTION', 0.05))
        self.update_interval = int(getattr(GpuGlobals, 'LIBRARY_UPDATE_INTERVAL', 10))

        # Library storage: [capacity, max_block_len] token tensor
        self.library_tokens = torch.full(
            (capacity, self.max_block_len), PAD_ID,
            dtype=torch.int32, device=device
        )
        # Best fitness seen for each slot (lower = better)
        self.library_fitness = torch.full((capacity,), float('inf'), dtype=torch.float32, device=device)
        # Usage count per slot
        self.library_count = torch.zeros(capacity, dtype=torch.long, device=device)
        # Validity mask: True = slot is occupied
        self.valid = torch.zeros(capacity, dtype=torch.bool, device=device)

        # Hash multipliers (random large primes for poly-hash)
        self._hash_mults = torch.randint(
            1, 2**31 - 1, (self.max_block_len,),
            dtype=torch.long, device=device
        )

    def _hash_subtrees(self, subtrees: torch.Tensor) -> torch.Tensor:
        """
        Compute a polynomial rolling hash for each row in subtrees.
        Args:
            subtrees: [N, max_block_len] int32 tensor
        Returns:
            hashes: [N] long tensor, values in [0, capacity)
        """
        N, L = subtrees.shape
        tokens_long = subtrees.long()  # [N, L]
        # poly hash: sum of token_i * mult_i mod capacity
        mults = self._hash_mults[:L].unsqueeze(0)  # [1, L]
        hashes = (tokens_long * mults).sum(dim=1)  # [N]
        return hashes.abs() % self.capacity

    def update(
        self,
        population: torch.Tensor,
        fitness_rmse: torch.Tensor,
    ) -> None:
        """
        Extract subtrees from top-K individuals and update the library.
        
        Args:
            population:   [pop_size, max_len] int tensor (RPN)
            fitness_rmse: [pop_size] float tensor
        """
        try:
            self._do_update(population, fitness_rmse)
        except Exception:
            pass  # Library learning is always non-fatal

    def _do_update(self, population: torch.Tensor, fitness_rmse: torch.Tensor) -> None:
        n = population.shape[0]
        k = max(1, int(n * self.top_k_fraction))
        # Sample top-K by fitness (smallest RMSE)
        _, top_idx = torch.topk(fitness_rmse, k, largest=False)
        top_pop = population[top_idx]  # [k, max_len]

        # Extract all windows of length max_block_len from each individual
        # Shape: [k, n_windows, max_block_len] where n_windows = max_len - block_len + 1
        blen = self.max_block_len
        L = top_pop.shape[1]
        n_windows = max(1, L - blen + 1)

        # Unroll windows: [k * n_windows, blen]
        windows = top_pop.unfold(1, blen, 1)  # [k, n_windows, blen]
        windows = windows.contiguous().view(-1, blen).int()  # [k*n_windows, blen]

        # Filter out all-PAD windows and windows starting with PAD
        not_all_pad = (windows != PAD_ID).any(dim=1)  # [N]
        no_leading_pad = (windows[:, 0] != PAD_ID)
        valid_windows = not_all_pad & no_leading_pad
        windows = windows[valid_windows]
        if windows.shape[0] == 0:
            return

        # Replicate fitness for each window from its source individual
        window_frac = (L - blen + 1) if (L - blen + 1) > 0 else 1
        # Each individual contributes window_frac windows; map back
        # (approximate — use the top individual's fitness for all its windows)
        k_actual = top_pop.shape[0]
        fit_expanded = fitness_rmse[top_idx].unsqueeze(1).expand(k_actual, n_windows)
        fit_expanded = fit_expanded.contiguous().view(-1)
        fit_expanded = fit_expanded[valid_windows.cpu() if not valid_windows.is_cuda else valid_windows]
        if fit_expanded.shape[0] != windows.shape[0]:
            # Shape mismatch fallback: use mean fitness
            fit_expanded = fit_expanded[:windows.shape[0]] if fit_expanded.shape[0] > windows.shape[0] else \
                           torch.cat([fit_expanded, fit_expanded.mean().expand(windows.shape[0] - fit_expanded.shape[0])])

        # Hash each window and insert into library
        slot_ids = self._hash_subtrees(windows)  # [N]

        # Only update slot if this subtree is better than current occupant
        better_mask = fit_expanded.to(self.library_fitness.device) < self.library_fitness[slot_ids]
        if not better_mask.any():
            return

        # Vectorized Update (No Python Loop)
        # 1. Select only improving candidates
        better_idx = better_mask.nonzero(as_tuple=False).squeeze(1)
        
        # 2. Sort by fitness (DESCENDING) so the BEST fitness (lowest value) is last.
        #    When we do scatter/index_put, the last write to a slot wins.
        #    We want the best candidate to win if multiple map to same slot.
        cand_fits = fit_expanded[better_idx]
        sorted_indices = torch.argsort(cand_fits, descending=True)
        better_idx = better_idx[sorted_indices]
        
        # 3. Gather data for update
        target_slots = slot_ids[better_idx]
        target_windows = windows[better_idx]
        target_fits = fit_expanded[better_idx]
        
        # 4. Bulk Write (Overwrite mode: last one wins -> best fitness wins)
        self.library_tokens[target_slots] = target_windows
        self.library_fitness[target_slots] = target_fits.float()
        self.valid[target_slots] = True
        
        # 5. Count updates (Use index_add to handle collisions properly)
        #    We increment count for every *candidate* that was better than previous.
        ones = torch.ones_like(target_slots, dtype=torch.long)
        self.library_count.index_add_(0, target_slots, ones)

    def sample(self, k: int = 64) -> Optional[torch.Tensor]:
        """
        Sample k blocks from the library, weighted by inverse fitness and count.

        Returns:
            [k, max_block_len] int tensor, or None if library is empty.
        """
        try:
            n_valid = self.valid.sum().item()
            if n_valid == 0:
                return None
            valid_idx = self.valid.nonzero(as_tuple=False).squeeze(1)
            # Weight: high count + low fitness = more likely sampled
            fit = self.library_fitness[valid_idx].float()
            cnt = self.library_count[valid_idx].float()
            # Inverse fitness weight (guard against inf)
            fit_w = 1.0 / (fit.clamp(min=1e-8))
            cnt_w = (cnt + 1.0).log()
            weights = fit_w * cnt_w
            # Sample with replacement
            n_sample = min(k, int(n_valid))
            sampled_local = torch.multinomial(weights, n_sample, replacement=(n_sample > n_valid))
            sampled_global = valid_idx[sampled_local]
            blocks = self.library_tokens[sampled_global]  # [n_sample, max_block_len]
            return blocks
        except Exception:
            return None

    def inject_into_mutation_bank(self, mutation_bank: torch.Tensor, fraction: float = 0.1) -> torch.Tensor:
        """
        Replace a fraction of mutation_bank rows with library blocks.
        Pads short blocks to mutation_bank's max_len with PAD_ID.

        Args:
            mutation_bank: [B, bank_max_len] tensor
            fraction:      fraction of bank to replace with library blocks

        Returns:
            Updated mutation_bank (same tensor, modified in-place).
        """
        try:
            B, bank_len = mutation_bank.shape
            n_inject = max(1, int(B * fraction))
            blocks = self.sample(n_inject)
            if blocks is None:
                return mutation_bank
            n_actual = blocks.shape[0]
            blen = blocks.shape[1]
            # Pad if needed
            if blen < bank_len:
                pad = torch.full(
                    (n_actual, bank_len - blen), PAD_ID,
                    dtype=mutation_bank.dtype, device=mutation_bank.device
                )
                blocks = torch.cat([blocks.to(mutation_bank.device, mutation_bank.dtype), pad], dim=1)
            elif blen > bank_len:
                blocks = blocks[:, :bank_len].to(mutation_bank.device, mutation_bank.dtype)
            else:
                blocks = blocks.to(mutation_bank.device, mutation_bank.dtype)
            # Write to random positions in the bank
            rand_pos = torch.randint(0, B, (n_actual,), device=mutation_bank.device)
            mutation_bank[rand_pos] = blocks
        except Exception:
            pass
        return mutation_bank

    @property
    def size(self) -> int:
        return int(self.valid.sum().item())

    def __repr__(self) -> str:
        return f"LibraryLearner(capacity={self.capacity}, filled={self.size}, block_len={self.max_block_len})"
