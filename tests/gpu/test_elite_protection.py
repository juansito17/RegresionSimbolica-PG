"""
test_elite_protection.py — Verifica que el elite (índice 0) está protegido de crossover y mutación.

El fix en rpn_kernels.cu debería:
1. Excluir el índice 0 del pool de crossover.
2. Excluir el índice 0 del pool de mutación.
3. Preservar population[0] en next_pop[0] exactamente.
"""
import sys
import os
import unittest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from warpsymbolic.gpu.engine import TensorGeneticEngine
from warpsymbolic.gpu.cuda_vm import CudaRPNVM

def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TestEliteProtection(unittest.TestCase):
    def setUp(self):
        self.device = _get_device()
        if not torch.cuda.is_available():
            self.skipTest("CUDA no disponible")
            
        # Motor con pop_size pequeño para el test
        self.engine = TensorGeneticEngine(
            device=self.device, pop_size=100, max_len=32, n_islands=1,
            num_variables=1
        )

    def test_elite_is_preserved_identically(self):
        """
        Verifica que el individuo 0 no cambia tras una evolución, incluso con rates de 1.0.
        """
        engine = self.engine
        B, L = engine.pop_size, engine.max_len
        
        # 1. Crear una población donde el individuo 0 es único y largo (para forzar crossover/bloat)
        vm = CudaRPNVM(engine.grammar, engine.device)
        
        # Formula: x0 + x0 + x0 + x0 ... (larga)
        tokens = [vm.id_x_start] + [vm.id_x_start, vm.op_add] * 10
        prog = torch.full((L,), vm.PAD_ID, dtype=torch.uint8, device=self.device)
        for i, t in enumerate(tokens):
            prog[i] = t
        
        population = engine.operators.generate_random_population(B)
        population[0] = prog.clone()
        
        constants = torch.randn(B, engine.max_constants, device=self.device)
        fitness = torch.ones(B, device=self.device)
        # El 0 es el mejor (fitness mas bajo)
        fitness[0] = 0.0
        
        abs_errors = torch.ones(B, 10, device=self.device)
        x_t = torch.randn(1, 10, device=self.device)
        y_t = torch.randn(10, device=self.device)
        
        # 2. Ejecutar evolución con rates de 1.0 para maximizar posibilidad de destrucción
        # Si el elite no está protegido, se destruirá casi seguro.
        next_pop, next_c, next_fit = engine.evolve_generation_cuda(
            population, constants, fitness, abs_errors, x_t, y_t, engine.mutation_bank,
            mutation_rate=1.0, crossover_rate=1.0, tournament_size=10
        )
        
        # 3. Verificar que next_pop[0] es IDÉNTICO a population[0] (el elite inicial)
        is_same = torch.all(next_pop[0] == prog)
        
        # También verificar que las constantes se preservan
        is_const_same = torch.all(next_c[0] == constants[0])
        
        self.assertTrue(is_same.item(), "BUG-ELT-1: El elite en posición 0 fue modificado por la evolución (crossover o mutación).")
        self.assertTrue(is_const_same.item(), "El elite en posición 0 tuvo sus constantes modificadas.")
        
        print("\n[VERIFIED] Elite protection confirmed: population[0] remains unchanged after evolve_generation_cuda.")

if __name__ == "__main__":
    unittest.main()
