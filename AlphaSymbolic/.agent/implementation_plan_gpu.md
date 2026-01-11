
# Implementation Plan - GPU Engine Optimization

This plan outlines the steps to upgrade the Python GPU (PyTorch) Genetic Programming engine to surpass the C++ implementation in terms of speed, accuracy, and robustness.

## Goal
Achieve SOTA performance by fully utilizing GPU parallelism, implementing advanced evolutionary operators on tensors, and fixing numerical instability.

## Phase 1: Robustness & Kernel Optimization (Critical)
The current GPU engine fails on multivariable problems due to numerical instability (gradients/NaNs) and dtype mismatches.

### 1.1 Protected Operators (Tensor-Native)
- [ ] Modify `evaluate_batch` to use fully protected math operations without branching where possible, or robust masking.
- [ ] Implement `Two-Argument Arctangent (atan2)` and other safe ops.
- [ ] Ensure all division/log/exp/sqrt operations handle invalid inputs gracefully (return large penalty or clamped value) to prevent `NaN` propagation to gradients or fitness.

### 1.2 Dtype Standardization
- [ ] Standardize all tensors to `float64` (Double) by default for regression tasks to match C++ precision.
- [ ] Ensure `benchmark_gp_vs_gpu.py` and `engine.py` initialization respects this global setting.

## Phase 2: Speed & Vectorization (The "Fast" Part)
The current `crossover_population` moves data to CPU for splicing, which is a massive bottleneck.

### 2.1 Fully Vectorized Crossover
- [ ] Implement `crossover_population` purely in PyTorch using `torch.gather` and mask manipulation.
- [ ] Strategy: 
    1. Select crossover points on GPU.
    2. Construct mask tensors for "Keep Left", "Insert Middle", "Keep Right".
    3. Use `torch.cat` or advanced indexing to construct offspring batches in parallel.

### 2.2 Massive Batching
- [ ] Optimize memory usage to support `PopSize=100,000` on 8GB VRAM.
- [ ] Use `int16` or `int8` for tokens if vocabulary < 256 to save memory (currently `long`).

## Phase 3: Smart Exploration (The "Best" Part)
C++ wins on accuracy because of Island Models and diversity preservation.

### 3.1 GPU Island Model
- [ ] Implement "Virtual Islands": Reshape population tensor to `[NumIslands, IslandSize, GenomeLen]`.
- [ ] Restrict crossover/selection to within islands for N generations.
- [ ] Implement vectorized migration (swap blocks of tensors between islands).

### 3.2 Advanced Selection
- [ ] Implement **Tensor Deterministic Crowding**: Offspring competes with parent.
- [ ] Implement **Epsilon-Lexicase Selection** vectorized: Calculate fitness on random subset of test cases per generation to speed up evaluation and maintain diversity.

## Phase 4: Hybrid Search Integration
- [ ] Fix `optimize_constants` to be robust. 
- [ ] Apply constant optimization only to the top 5% of islands every 10 generations to save compute.

## Verification
- [ ] Re-run `benchmark_gp_vs_gpu.py` after each phase.
- [ ] Target: < 1.0s for Simple Poly, < 5.0s for Trig Mix, correct solution for Multivariable.
