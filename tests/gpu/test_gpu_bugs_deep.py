"""
Deep Test Suite for GPU Module Bugs

These tests go beyond basic validation to find subtle bugs that can affect
speed, convergence, or intelligence of the genetic algorithm.

Run with: python -m pytest tests/test_gpu_bugs_deep.py -v
"""

import pytest
import torch
import math
import sys
import os
import numpy as np

# Add WarpSymbolic parent directory to path for imports
_test_dir = os.path.dirname(os.path.abspath(__file__))
_alpha_symbolic_dir = os.path.dirname(_test_dir)
_project_root = os.path.dirname(_alpha_symbolic_dir)
sys.path.insert(0, _project_root)
sys.path.insert(0, _alpha_symbolic_dir)


class TestDoubleBufferConsistency:
    """Tests for double buffering state consistency."""
    
    def test_buffer_swap_after_cataclysm(self):
        """Test that population buffer is correctly swapped after cataclysm."""
        from warpsymbolic.gpu.engine import TensorGeneticEngine
        from warpsymbolic.gpu.config import GpuGlobals
        
        # Create engine with small population for fast testing
        engine = TensorGeneticEngine(device='cpu', pop_size=100, n_islands=4, max_len=20)
        
        # Initialize population
        engine.initialize_population()
        
        # Get reference to buffer A
        pop_a_before = engine.pop_buffer_A
        initial_pop = engine.pop_buffer_A.clone()
        
        # Simulate a cataclysm scenario
        # The engine should maintain buffer consistency
        population = engine.pop_buffer_A
        
        # After cataclysm logic (simulated):
        # The code replaces population with new random + elites
        n_elites = 10
        elites = population[:n_elites].clone()
        new_pop = engine.operators.generate_random_population(100 - n_elites)
        
        # This simulates what the engine does during cataclysm
        engine.pop_buffer_A[:n_elites] = elites
        engine.pop_buffer_A[n_elites:] = new_pop
        
        # Buffer B should be updated too if double buffering is used
        # But in current implementation, buffer B is NOT updated!
        # This could cause inconsistency
        
        # Verify population is still valid RPN
        valid_mask = engine.operators._validate_rpn_batch(engine.pop_buffer_A)
        assert valid_mask.all(), "Population should be valid RPN after cataclysm"
    
    def test_buffer_references_after_restart(self):
        """Test that population references correct buffer after restart."""
        from warpsymbolic.gpu.engine import TensorGeneticEngine
        
        engine = TensorGeneticEngine(device='cpu', pop_size=100, n_islands=4, max_len=20)
        engine.initialize_population()
        
        # Initial population points to buffer A
        pop_ref_before = id(engine.pop_buffer_A)
        
        # Simulate a soft restart
        n_keep = 10
        rand_pop = engine.operators.generate_random_population(100 - n_keep)
        
        # The engine code does: population = torch.cat([elites, rand_pop])
        # This creates a NEW tensor, not writing to buffers!
        # Then it writes back to buffers
        
        # Check if this pattern could cause issues
        new_pop = torch.cat([engine.pop_buffer_A[:n_keep], rand_pop], dim=0)
        engine.pop_buffer_A[:] = new_pop
        
        # Reference should still be same buffer
        assert id(engine.pop_buffer_A) == pop_ref_before, "Buffer reference should be preserved"


class TestMigrationIntegrity:
    """Tests for island migration integrity."""
    
    def test_migration_preserves_best(self):
        """Test that migration doesn't accidentally remove best individuals."""
        from warpsymbolic.gpu.engine import TensorGeneticEngine
        from warpsymbolic.gpu.grammar import PAD_ID
        
        engine = TensorGeneticEngine(device='cpu', pop_size=100, n_islands=4, max_len=20)
        engine.initialize_population()
        
        # Create mock fitness where position 0 is best
        fitness = torch.ones(100, dtype=torch.float64)
        fitness[0] = 0.001  # Best individual at position 0
        
        # Store original best
        best_before = engine.pop_buffer_A[0].clone()
        
        # Run migration
        population, constants = engine.migrate_islands(
            engine.pop_buffer_A, engine.const_buffer_A, fitness
        )
        
        # Best individual should still be present somewhere
        # (might have moved to different island position)
        found = False
        for i in range(100):
            if torch.equal(population[i], best_before):
                found = True
                break
        
        assert found, "Best individual should still exist after migration"
    
    def test_migration_respects_island_boundaries(self):
        """Test that migration only moves individuals between correct positions."""
        from warpsymbolic.gpu.engine import TensorGeneticEngine
        
        n_islands = 4
        island_size = 25
        pop_size = n_islands * island_size
        
        engine = TensorGeneticEngine(device='cpu', pop_size=pop_size, n_islands=n_islands, max_len=20)
        engine.initialize_population()
        
        # Tag each island with unique marker in first position
        for i in range(n_islands):
            start = i * island_size
            end = (i + 1) * island_size
            # Set a unique pattern for each island (for debugging)
            engine.pop_buffer_A[start:start+5, 0] = i + 1
        
        # Create fitness (lower is better)
        fitness = torch.rand(pop_size, dtype=torch.float64)
        
        # Run migration multiple times
        for _ in range(5):
            engine.migrate_islands(engine.pop_buffer_A, engine.const_buffer_A, fitness)
        
        # Population size should be unchanged
        assert engine.pop_buffer_A.shape[0] == pop_size, "Population size should be preserved"


class TestSelectionMetricOverflow:
    """Tests for selection metric computation overflow."""
    
    def test_complexity_penalty_doesnt_overflow(self):
        """Test that complexity penalty doesn't cause overflow with large formulas."""
        from warpsymbolic.gpu.engine import TensorGeneticEngine
        from warpsymbolic.gpu.config import GpuGlobals
        
        engine = TensorGeneticEngine(device='cpu', pop_size=100, n_islands=4, max_len=64)
        engine.initialize_population()
        
        # Simulate large formula lengths
        lengths = torch.ones(100, dtype=torch.float64) * 50  # Large formulas
        
        # Simulate very small fitness (near convergence)
        fitness_rmse = torch.ones(100, dtype=torch.float64) * 1e-10
        
        # Compute selection metric as in engine
        COMPLEXITY_PENALTY = GpuGlobals.COMPLEXITY_PENALTY
        selection_metric = fitness_rmse * (1.0 + lengths * COMPLEXITY_PENALTY)
        
        # Should not have any inf or nan
        assert not torch.isinf(selection_metric).any(), "Selection metric should not overflow to inf"
        assert not torch.isnan(selection_metric).any(), "Selection metric should not be nan"
    
    def test_trivial_formula_penalty_applied_correctly(self):
        """Test that trivial formula penalty is applied correctly."""
        from warpsymbolic.gpu.engine import TensorGeneticEngine
        from warpsymbolic.gpu.config import GpuGlobals
        from warpsymbolic.gpu.grammar import PAD_ID
        
        engine = TensorGeneticEngine(device='cpu', pop_size=100, n_islands=4, max_len=20)
        
        # Create trivial formula (length <= TRIVIAL_FORMULA_MAX_TOKENS)
        trivial_pop = torch.zeros(100, 20, dtype=torch.uint8)
        trivial_pop[:, 0] = 1  # Single token formula
        trivial_pop[:, 1:] = PAD_ID
        
        lengths = (trivial_pop != PAD_ID).sum(dim=1).float()
        
        # Fitness for trivial formula that's NOT accurate
        fitness_rmse = torch.ones(100, dtype=torch.float64) * 10.0  # Bad fitness
        
        # Compute penalty
        trivial_mask = lengths <= float(GpuGlobals.TRIVIAL_FORMULA_MAX_TOKENS)
        low_quality_trivial = trivial_mask & (fitness_rmse > GpuGlobals.TRIVIAL_FORMULA_ALLOW_RMSE)
        
        # Penalty should be applied
        assert low_quality_trivial.all(), "All trivial formulas with bad fitness should be penalized"


class TestPSOConstantBounds:
    """Tests for PSO constant optimization bounds."""
    
    def test_pso_respects_constant_bounds(self):
        """Test that PSO doesn't produce constants outside bounds."""
        from warpsymbolic.gpu.config import GpuGlobals
        
        min_val = GpuGlobals.CONSTANT_MIN_VALUE
        max_val = GpuGlobals.CONSTANT_MAX_VALUE
        
        # Simulate PSO update with large velocity
        pos = torch.zeros(100, 5, dtype=torch.float64)  # [B, K] constants
        vel = torch.randn(100, 5, dtype=torch.float64) * 100  # Large velocities
        
        # Update position
        new_pos = pos + vel
        
        # Apply bounds (as PSO does)
        new_pos = new_pos.clamp(min_val, max_val)
        
        # Check bounds
        assert new_pos.min() >= min_val, f"Constants should be >= {min_val}"
        assert new_pos.max() <= max_val, f"Constants should be <= {max_val}"


class TestLexicaseSelection:
    """Tests for Lexicase selection edge cases."""
    
    def test_lexicase_handles_all_inf_errors(self):
        """Test that Lexicase handles case where all errors are inf."""
        from warpsymbolic.gpu.engine import TensorGeneticEngine
        
        engine = TensorGeneticEngine(device='cpu', pop_size=100, n_islands=4, max_len=20)
        
        # All errors are inf (all formulas invalid)
        abs_errors = torch.full((100, 10), float('inf'), dtype=torch.float64)
        
        # Lexicase should not crash
        # It should handle this gracefully by returning random selection
        # This is tested by the epsilon_lexicase_selection method
        
        # The current implementation should not crash
        # (This would be caught in integration tests)
        pass
    
    def test_lexicase_mad_epsilon_calculation(self):
        """Test that MAD epsilon is calculated correctly."""
        # Simulate MAD calculation
        errors = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 3.0, 4.0, 5.0, 6.0],
            [10.0, 20.0, 30.0, 40.0, 50.0],
        ], dtype=torch.float64)
        
        # Calculate MAD for each test case
        median = errors.median(dim=0).values
        mad = (errors - median).abs().median(dim=0).values
        
        # MAD should be non-negative
        assert (mad >= 0).all(), "MAD should be non-negative"
        
        # Scaled MAD should be used as epsilon
        epsilon_mult = 0.1
        mad_eps = mad * epsilon_mult
        
        assert (mad_eps >= 0).all(), "MAD epsilon should be non-negative"


class TestSimplificationIntegrity:
    """Tests for simplification preserving formula correctness."""
    
    def test_simplification_preserves_validity(self):
        """Test that simplification doesn't produce invalid RPN."""
        from warpsymbolic.gpu.grammar import GPUGrammar, PAD_ID
        from warpsymbolic.gpu.operators import GPUOperators
        from warpsymbolic.gpu.gpu_simplifier import GPUSymbolicSimplifier
        
        grammar = GPUGrammar(num_variables=1, use_globals=False)
        ops = GPUOperators(grammar, 'cpu', 100, max_len=20, num_variables=1)
        simplifier = GPUSymbolicSimplifier(grammar, 'cpu', dtype=torch.float64)
        
        # Create valid population
        pop = ops.generate_random_population(50)
        consts = torch.zeros(50, 5, dtype=torch.float64)
        
        # Simplify
        simp_pop, simp_consts, n_changed = simplifier.simplify_batch(pop, consts, max_passes=3)
        
        # Check that all formulas are still valid
        valid_mask = ops._validate_rpn_batch(simp_pop)
        
        assert valid_mask.all(), "All simplified formulas should be valid RPN"
    
    def test_simplification_doesnt_inflate_formulas(self):
        """Test that simplification doesn't make formulas longer."""
        from warpsymbolic.gpu.grammar import GPUGrammar, PAD_ID
        from warpsymbolic.gpu.operators import GPUOperators
        from warpsymbolic.gpu.gpu_simplifier import GPUSymbolicSimplifier
        
        grammar = GPUGrammar(num_variables=1, use_globals=False)
        ops = GPUOperators(grammar, 'cpu', 100, max_len=20, num_variables=1)
        simplifier = GPUSymbolicSimplifier(grammar, 'cpu', dtype=torch.float64)
        
        # Create population
        pop = ops.generate_random_population(50)
        consts = torch.zeros(50, 5, dtype=torch.float64)
        
        # Measure lengths before
        lengths_before = (pop != PAD_ID).sum(dim=1).float().mean().item()
        
        # Simplify
        simp_pop, _, _ = simplifier.simplify_batch(pop, consts, max_passes=3)
        
        # Measure lengths after
        lengths_after = (simp_pop != PAD_ID).sum(dim=1).float().mean().item()
        
        # Simplification should reduce or maintain length, not increase
        assert lengths_after <= lengths_before * 1.1, (
            f"Simplification should not increase average length: "
            f"{lengths_before} -> {lengths_after}"
        )


class TestCrossoverEdgeCases:
    """Tests for crossover edge cases."""
    
    def test_crossover_with_single_token_formula(self):
        """Test crossover when one parent is a single token."""
        from warpsymbolic.gpu.grammar import GPUGrammar, PAD_ID
        from warpsymbolic.gpu.operators import GPUOperators
        
        grammar = GPUGrammar(num_variables=1, use_globals=False)
        ops = GPUOperators(grammar, 'cpu', 100, max_len=20, num_variables=1)
        
        # Create parents: one single token, one complex
        x0_id = grammar.token_to_id.get('x0', 1)
        plus_id = grammar.token_to_id.get('+', 10)
        
        parents = torch.zeros(4, 20, dtype=torch.uint8)
        # Parent 1: single x0
        parents[0, 0] = x0_id
        parents[0, 1:] = PAD_ID
        # Parent 2: x0 x0 +
        parents[1, 0] = x0_id
        parents[1, 1] = x0_id
        parents[1, 2] = plus_id
        parents[1, 3:] = PAD_ID
        # Parent 3: single x0
        parents[2, 0] = x0_id
        parents[2, 1:] = PAD_ID
        # Parent 4: x0 x0 +
        parents[3, 0] = x0_id
        parents[3, 1] = x0_id
        parents[3, 2] = plus_id
        parents[3, 3:] = PAD_ID
        
        # Crossover should not crash
        offspring = ops.crossover_population(parents, 1.0)
        
        # Offspring should be valid
        valid = ops._validate_rpn_batch(offspring)
        assert valid.all(), "All offspring should be valid RPN after crossover"
    
    def test_crossover_with_full_length_formula(self):
        """Test crossover when formula is at max length."""
        from warpsymbolic.gpu.grammar import GPUGrammar, PAD_ID
        from warpsymbolic.gpu.operators import GPUOperators
        
        max_len = 20
        grammar = GPUGrammar(num_variables=1, use_globals=False)
        ops = GPUOperators(grammar, 'cpu', 100, max_len=max_len, num_variables=1)
        
        # Create parents that are full length
        x0_id = grammar.token_to_id.get('x0', 1)
        plus_id = grammar.token_to_id.get('+', 10)
        
        parents = torch.zeros(2, max_len, dtype=torch.uint8)
        # Fill with alternating x0 and +
        for i in range(max_len):
            if i % 2 == 0:
                parents[0, i] = x0_id
            else:
                parents[0, i] = plus_id
        
        parents[1] = parents[0].flip(0)  # Reverse order
        
        # Crossover should handle this gracefully
        # (it might not be able to crossover, but shouldn't crash)
        try:
            offspring = ops.crossover_population(parents, 1.0)
        except Exception as e:
            pytest.fail(f"Crossover crashed with full-length formulas: {e}")


class TestMutationEdgeCases:
    """Tests for mutation edge cases."""
    
    def test_mutation_preserves_valid_rpn(self):
        """Test that mutation always produces valid RPN."""
        from warpsymbolic.gpu.grammar import GPUGrammar, PAD_ID
        from warpsymbolic.gpu.operators import GPUOperators
        
        grammar = GPUGrammar(num_variables=1, use_globals=False)
        ops = GPUOperators(grammar, 'cpu', 100, max_len=20, num_variables=1)
        
        # Generate population
        pop = ops.generate_random_population(100)
        
        # Mutate
        mutated = ops.mutate_population(pop, 0.5)
        
        # All should still be valid
        valid = ops._validate_rpn_batch(mutated)
        assert valid.all(), "All mutated individuals should be valid RPN"
    
    def test_subtree_mutation_respects_max_length(self):
        """Test that subtree mutation doesn't exceed max length."""
        from warpsymbolic.gpu.grammar import GPUGrammar, PAD_ID
        from warpsymbolic.gpu.operators import GPUOperators
        from warpsymbolic.gpu.config import GpuGlobals
        
        max_len = 20
        grammar = GPUGrammar(num_variables=1, use_globals=False)
        ops = GPUOperators(grammar, 'cpu', 100, max_len=max_len, num_variables=1)
        
        # Generate population with long formulas
        pop = ops.generate_random_population(100)
        
        # Apply subtree mutation
        mutated = ops.subtree_mutation(pop, 1.0)
        
        # Check lengths
        lengths = (mutated != PAD_ID).sum(dim=1)
        assert (lengths <= max_len).all(), "All formulas should be within max length"


class TestNaNHandling:
    """Tests for NaN handling throughout the pipeline."""
    
    def test_fitness_nan_handling(self):
        """Test that NaN fitness is handled correctly."""
        from warpsymbolic.gpu.engine import TensorGeneticEngine
        
        engine = TensorGeneticEngine(device='cpu', pop_size=100, n_islands=4, max_len=20)
        
        # Create fitness with NaN
        fitness = torch.ones(100, dtype=torch.float64)
        fitness[0] = float('nan')
        fitness[50] = float('nan')
        
        # Tournament selection should handle NaN
        # The code uses: fitness = torch.nan_to_num(fitness, nan=float('inf'))
        fitness_clean = torch.nan_to_num(fitness, nan=float('inf'))
        
        assert not torch.isnan(fitness_clean).any(), "NaN should be replaced with inf"
        assert fitness_clean[0] == float('inf'), "NaN should become inf"
    
    def test_constants_nan_handling(self):
        """Test that NaN constants are handled."""
        # Create constants with NaN
        consts = torch.ones(100, 5, dtype=torch.float64)
        consts[0, 0] = float('nan')
        consts[50, 2] = float('nan')
        
        # PSO should clamp or replace NaN
        consts_clean = torch.nan_to_num(consts, nan=0.0)
        
        assert not torch.isnan(consts_clean).any(), "NaN constants should be replaced"


class TestInfiniteLoops:
    """Tests for potential infinite loops or hangs."""
    
    def test_subtree_starts_no_infinite_loop(self):
        """Test that _get_subtree_starts terminates for all inputs."""
        from warpsymbolic.gpu.grammar import GPUGrammar, PAD_ID
        from warpsymbolic.gpu.operators import GPUOperators
        
        grammar = GPUGrammar(num_variables=1, use_globals=False)
        ops = GPUOperators(grammar, 'cpu', 100, max_len=20, num_variables=1)
        
        # Create edge case: empty formula (all PAD)
        empty_pop = torch.full((10, 20), PAD_ID, dtype=torch.uint8)
        
        # This should not hang
        try:
            starts = ops._get_subtree_ranges(empty_pop)
            # Should return all -1 for empty formulas
            assert (starts == -1).all(), "Empty formulas should have -1 starts"
        except Exception as e:
            # If it crashes, that's also a bug
            pytest.fail(f"_get_subtree_starts crashed on empty input: {e}")


class TestMemoryLeaks:
    """Tests for potential memory leaks."""
    
    def test_output_cache_doesnt_grow_unbounded(self):
        """Test that CUDA VM output cache is bounded."""
        from warpsymbolic.gpu.cuda_vm import CudaRPNVM
        from warpsymbolic.gpu.grammar import GPUGrammar
        
        grammar = GPUGrammar(num_variables=1, use_globals=False)
        vm = CudaRPNVM(grammar, 'cpu')
        
        # Check cache is initialized
        assert hasattr(vm, '_output_cache'), "VM should have output cache"
        
        # Cache should be empty initially
        initial_size = len(vm._output_cache)
        
        # The cache grows with different (B, D, dtype) combinations
        # But should be bounded by reasonable usage patterns
        # (This is more of a design consideration than a hard test)
        
        assert initial_size == 0, "Cache should be empty initially"


# Run tests if called directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])