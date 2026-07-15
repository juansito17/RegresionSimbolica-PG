"""Persistent tensors used by the native CUDA evolution path.

The native orchestrator still owns short-lived implementation details, but all
large boundary buffers live here so Python does not allocate/cast/sanitize the
same tensors on every generation.
"""
from __future__ import annotations

import torch


class CUDAEvolutionWorkspace:
    def __init__(self, device: torch.device, population_size: int, max_len: int,
                 max_constants: int, n_islands: int):
        self.device = torch.device(device)
        self.population_size = int(population_size)
        self.max_len = int(max_len)
        self.max_constants = int(max_constants)
        self.n_islands = int(n_islands)

        self.empty_float = torch.empty(0, dtype=torch.float32, device=self.device)
        self.empty_uint8 = torch.empty(0, dtype=torch.uint8, device=self.device)
        self.fitness_f32 = torch.empty(self.population_size, dtype=torch.float32, device=self.device)
        self.lengths_f32 = torch.empty(self.population_size, dtype=torch.float32, device=self.device)
        self._abs_errors = None
        self._mad_eps = None

    def sanitized_fitness(self, fitness: torch.Tensor) -> torch.Tensor:
        fitness = fitness if fitness.dtype == torch.float32 else fitness.float()
        if fitness.numel() != self.population_size:
            return torch.nan_to_num(fitness, nan=float('inf'))
        torch.nan_to_num(fitness, nan=float('inf'), posinf=float('inf'),
                         neginf=float('inf'), out=self.fitness_f32)
        return self.fitness_f32

    def formula_lengths(self, population: torch.Tensor, pad_id: int) -> torch.Tensor:
        # ``sum(..., out=)`` is not supported for this reduction on every PyTorch
        # build, so copy into the persistent destination after the reduction.
        self.lengths_f32.copy_((population != pad_id).sum(dim=1))
        return self.lengths_f32

    def sanitized_errors(self, errors: torch.Tensor | None) -> torch.Tensor:
        if errors is None or errors.numel() == 0:
            return self.empty_float
        errors = errors if errors.dtype == torch.float32 else errors.float()
        if self._abs_errors is None or self._abs_errors.shape != errors.shape:
            self._abs_errors = torch.empty(errors.shape, dtype=torch.float32, device=self.device)
        torch.nan_to_num(errors, nan=float('inf'), posinf=float('inf'),
                         neginf=float('inf'), out=self._abs_errors)
        return self._abs_errors

    def mad_buffer(self, size: int) -> torch.Tensor:
        if size <= 0:
            return self.empty_float
        if self._mad_eps is None or self._mad_eps.numel() != size:
            self._mad_eps = torch.empty(size, dtype=torch.float32, device=self.device)
        return self._mad_eps
