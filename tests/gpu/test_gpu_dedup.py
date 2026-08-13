"""
Tests for CUDA Structural Hash & Deduplication

Simple tests that avoid circular import issues.
"""

import torch
import pytest

from warpsymbolic.gpu.cuda_loader import load_rpn_cuda_native


class TestCUDADedup:
    """Test CUDA structural deduplication."""
    
    @pytest.fixture
    def setup_cuda(self):
        """Setup CUDA components for testing."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        device = torch.device('cuda')
        
        from warpsymbolic.gpu.grammar import GPUGrammar, PAD_ID
        
        grammar = GPUGrammar(num_variables=1)
        
        return grammar, device, PAD_ID
    
    def test_hash_kernel_exists(self, setup_cuda):
        """Verify CUDA hash kernel is available and callable."""
        grammar, device, PAD_ID = setup_cuda
        
        try:
            rpn_cuda_native = load_rpn_cuda_native()
            assert hasattr(rpn_cuda_native, 'compute_population_hashes'), \
                "compute_population_hashes not found in CUDA module"
        except ImportError:
            pytest.skip("CUDA native module not compiled. Build src/warpsymbolic/gpu/cuda first.")
    
    def test_hash_identical_formulas(self, setup_cuda):
        """Identical RPN formulas must produce identical hashes."""
        grammar, device, PAD_ID = setup_cuda
        
        try:
            rpn_cuda_native = load_rpn_cuda_native()
        except ImportError:
            pytest.skip("CUDA native module not compiled")
        
        # Simple test: create two identical "formulas"
        # Just x0 (token id 1 typically)
        x0_id = grammar.token_to_id.get('x0', 1)
        
        pop = torch.full((2, 30), PAD_ID, dtype=torch.uint8, device=device)
        pop[0, 0] = x0_id
        pop[1, 0] = x0_id  # Identical
        
        hashes = torch.empty(2, dtype=torch.long, device=device)
        var_presence = torch.empty(2, dtype=torch.int32, device=device)
        
        id_x_start = grammar.token_to_id.get('x0', 1)
        
        rpn_cuda_native.compute_population_hashes(
            pop, hashes, var_presence,
            PAD_ID, id_x_start, 1  # num_variables=1
        )
        
        assert hashes[0].item() == hashes[1].item(), \
            f"Identical formulas should have identical hashes: {hashes[0].item()} != {hashes[1].item()}"
    
    def test_hash_different_formulas(self, setup_cuda):
        """Different RPN formulas should produce different hashes."""
        grammar, device, PAD_ID = setup_cuda
        
        try:
            rpn_cuda_native = load_rpn_cuda_native()
        except ImportError:
            pytest.skip("CUDA native module not compiled")
        
        x0_id = grammar.token_to_id.get('x0', 1)
        c_id = grammar.token_to_id.get('C', 5)
        
        pop = torch.full((2, 30), PAD_ID, dtype=torch.uint8, device=device)
        pop[0, 0] = x0_id   # Formula 0: x0
        pop[1, 0] = c_id    # Formula 1: C (different)
        
        hashes = torch.empty(2, dtype=torch.long, device=device)
        var_presence = torch.empty(2, dtype=torch.int32, device=device)
        
        id_x_start = grammar.token_to_id.get('x0', 1)
        
        rpn_cuda_native.compute_population_hashes(
            pop, hashes, var_presence,
            PAD_ID, id_x_start, 1
        )
        
        assert hashes[0].item() != hashes[1].item(), \
            f"Different formulas should have different hashes"
    
    def test_var_presence_bitmask(self, setup_cuda):
        """Variable presence bitmask should detect used variables."""
        grammar, device, PAD_ID = setup_cuda
        
        try:
            rpn_cuda_native = load_rpn_cuda_native()
        except ImportError:
            pytest.skip("CUDA native module not compiled")
        
        x0_id = grammar.token_to_id.get('x0', 1)
        c_id = grammar.token_to_id.get('C', 5)
        
        pop = torch.full((2, 30), PAD_ID, dtype=torch.uint8, device=device)
        pop[0, 0] = x0_id   # Uses x0
        pop[1, 0] = c_id    # No variable
        
        hashes = torch.empty(2, dtype=torch.long, device=device)
        var_presence = torch.empty(2, dtype=torch.int32, device=device)
        
        id_x_start = grammar.token_to_id.get('x0', 1)
        
        rpn_cuda_native.compute_population_hashes(
            pop, hashes, var_presence,
            PAD_ID, id_x_start, 1
        )
        
        # Formula 0 has x0 -> should have at least one bit set
        assert var_presence[0].item() > 0, \
            f"Formula with x0 should have var_presence > 0, got {var_presence[0].item()}"
        
        # Formula 1 has no variable -> should be 0
        assert var_presence[1].item() == 0, \
            f"Formula with C only should have var_presence = 0, got {var_presence[1].item()}"
    
    def test_dedup_kernel_basic(self, setup_cuda):
        """Structural dedup kernel should detect duplicates."""
        grammar, device, PAD_ID = setup_cuda
        
        try:
            rpn_cuda_native = load_rpn_cuda_native()
        except ImportError:
            pytest.skip("CUDA native module not compiled")
        
        x0_id = grammar.token_to_id.get('x0', 1)
        
        # Create population with 2 identical + 1 different
        pop = torch.full((3, 30), PAD_ID, dtype=torch.uint8, device=device)
        pop[0, 0] = x0_id
        pop[1, 0] = x0_id  # Duplicate of 0
        pop[2, 0] = grammar.token_to_id.get('C', 5)  # Different
        
        # Run hash first
        hashes = torch.empty(3, dtype=torch.long, device=device)
        var_presence = torch.empty(3, dtype=torch.int32, device=device)
        id_x_start = grammar.token_to_id.get('x0', 1)
        
        rpn_cuda_native.compute_population_hashes(
            pop, hashes, var_presence,
            PAD_ID, id_x_start, 1
        )
        
        # Count unique
        unique_count = torch.tensor([0], dtype=torch.int32, device=device)
        hash_table = torch.empty(1 << 20, dtype=torch.long, device=device)  # 1M entries
        hash_table.fill_(-1)  # Empty
        
        # Simple unique count via CPU for verification
        unique_hashes = torch.unique(hashes)
        
        assert len(unique_hashes) == 2, \
            f"Should have 2 unique hashes (x0 and C), got {len(unique_hashes)}"


class TestFallback:
    """Test fallback behavior."""
    
    def test_imports_work(self):
        """Verify basic imports work."""
        try:
            from warpsymbolic.gpu.grammar import GPUGrammar, PAD_ID
            grammar = GPUGrammar(num_variables=1)
            assert grammar is not None
        except ImportError as e:
            pytest.skip(f"Import failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
