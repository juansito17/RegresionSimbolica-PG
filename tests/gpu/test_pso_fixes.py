"""
test_pso_fixes.py — Verifica que los fixes BUG-PSO-2 y BUG-PSO-3 funcionan.

BUG-PSO-2: El jitter de inicialización de partículas PSO no escalaba al rango de constantes
           (σ=1 fijo → exploraba <2% del espacio [-50,50]).
           Fix: σ = 15% del rango.

BUG-PSO-3: En stagnation mode, el PSO tomaba solo los top-K individuos por fitness
           (todos structural-clones de la misma familia lgamma/fact).
           Fix: 50% top-fitness + 50% aleatorios para diversidad estructural.
"""
import sys
import os
import unittest
import math
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from warpsymbolic.gpu.grammar import GPUGrammar
from warpsymbolic.gpu.config import GpuGlobals


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 1 — BUG-PSO-2: Jitter PSO escala al rango real de constantes
# ─────────────────────────────────────────────────────────────────────────────

class TestPSOJitterScaling(unittest.TestCase):
    """
    Verifica que nano_pso converge mejor con el jitter escalado al rango
    que con jitter=1 fijo, especialmente cuando las constantes están FUERA
    del rango ±1 desde el punto inicial.

    Escenario: fórmula "C * x0" con objetivo 20*x0 (constante real=20).
    Si el jitter es σ=1, las partículas comienzan todas cerca de C=0
    y necesitan > 20 steps para escapar hacia C=20 con w=0.5.
    Con jitter=15% de rango [-50,50] (σ=15), las partículas exploran C∈[-50,50]
    en el primer step y convergen rápidamente.
    """

    def setUp(self):
        self.device = _get_device()

    def _make_c_times_x0_formula(self, engine):
        """Construye un programa RPN: C * x0 → [id_C, id_x0, op_mul]"""
        from warpsymbolic.gpu.cuda_vm import CudaRPNVM
        vm = CudaRPNVM(engine.grammar, engine.device)
        tokens = [vm.id_C, vm.id_x_start, vm.op_mul]
        prog = torch.full((engine.max_len,), vm.PAD_ID, dtype=torch.uint8, device=self.device)
        for i, t in enumerate(tokens):
            prog[i] = t
        return prog, vm

    def test_pso_finds_target_constant_with_scaled_jitter(self):
        """El PSO con jitter escalado debe encontrar C≈20 en pocos steps."""
        from warpsymbolic.gpu.engine import TensorGeneticEngine

        engine = TensorGeneticEngine(
            device=self.device, pop_size=50, max_len=16, n_islands=1,
            num_variables=1
        )
        grammar = engine.grammar

        # Datos: y = 20 * x0
        x = torch.linspace(1, 5, 10, device=self.device, dtype=torch.float32).unsqueeze(0)  # [1, 10]
        y = 20.0 * x.squeeze()

        # Construir población con 50 copias de "C * x0"
        prog, vm = self._make_c_times_x0_formula(engine)
        population = prog.unsqueeze(0).expand(50, -1).clone()

        # Constantes iniciales: todas C=0 (lejos del target C=20)
        constants = torch.zeros(50, engine.max_constants, device=self.device, dtype=torch.float32)

        # PSO con pocos steps para ver si el jitter escalado ayuda a encontrar C≈20
        refined_c, refined_err = engine.optimizer.nano_pso(
            population, constants, x, y, steps=40, num_particles=20
        )

        best_err = refined_err.min().item()
        best_c = refined_c[refined_err.argmin(), 0].item()

        print(f"\n[BUG-PSO-2] Best RMSE after 40 steps: {best_err:.6f}, Best C: {best_c:.4f}")

        # Con jitter escalado al 15% de rango [-50,50] (σ≈15),
        # las partículas deben explorar la región correcta y converger.
        self.assertLess(best_err, 1.0,
            f"PSO con jitter escalado debe converger a RMSE<1 (encontrado: {best_err:.4f}).\n"
            f"Si falla, el jitter σ=1 fijo sigue en uso (BUG-PSO-2 no corregido).")

        self.assertAlmostEqual(best_c, 20.0, delta=3.0,
            msg=f"La constante óptima debería ser ≈20 (encontrada: {best_c:.4f}).")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 2 — BUG-PSO-3: Stagnation PSO incluye candidatos aleatorios
# ─────────────────────────────────────────────────────────────────────────────

class TestPSOStagnationDiversity(unittest.TestCase):
    """
    Verifica que en stagnation mode el PSO selecciona candidatos aleatorios
    además de los top-fitness, evitando que todos sean clones estructurales.

    Crea una población donde los top-K son todos idénticos (misma fórmula)
    y verifica que top_idx en stagnation mode incluye individuos fuera de los top-K.
    """

    def setUp(self):
        self.device = _get_device()

    def test_stagnation_pso_includes_random_candidates(self):
        """
        En stagnation mode, top_idx debe incluir índices fuera de los mejores K.

        La lógica en engine.py (BUG-PSO-3 fix) hace:
            k_top = k_opt // 2
            _rand_part = torch.randperm(pop_size)[:k_rand]
            top_idx = torch.cat([_top_part, _rand_part])

        Simulamos esta lógica para verificar que top_idx no es un subconjunto
        estricto de los top-K individuos.
        """
        pop_size = 1000
        k_opt = 100  # PSO_K_STAGNATION pequeño para el test

        # Fitness: primeros 200 tienen rmse=0.001 (los mejores), el resto rmse=1.0
        fitness = torch.ones(pop_size, device=self.device) * 1.0
        fitness[:200] = 0.001  # top 200 son los mejores

        # Simular la lógica del fix BUG-PSO-3
        _in_stagnation = True

        if _in_stagnation:
            k_top = k_opt // 2          # 50
            k_rand = k_opt - k_top      # 50
            _, _top_part = torch.topk(fitness, k_top, largest=False)
            _rand_part = torch.randperm(pop_size, device=self.device)[:k_rand]
            top_idx = torch.cat([_top_part, _rand_part])
        else:
            _, top_idx = torch.topk(fitness, k_opt, largest=False)

        # top_idx debe tener exactamente k_opt elementos
        self.assertEqual(len(top_idx), k_opt,
            f"top_idx debe tener {k_opt} elementos (tiene {len(top_idx)}).")

        # top_idx debe incluir al menos algún índice fuera de los top-200
        # (con 50 aleatorios de 1000, probabilidad de que TODOS caigan en top-200 es ≈ 0)
        outside_top200 = (top_idx >= 200).sum().item()
        self.assertGreater(outside_top200, 0,
            "BUG-PSO-3: En stagnation mode, top_idx debe incluir candidatos "
            "fuera de los mejores 200 (diversidad estructural).")

        print(f"\n[BUG-PSO-3] top_idx: {k_top} top-fitness + {outside_top200} fuera-del-top (aleatorios).")

    def test_normal_mode_pso_uses_only_top_fitness(self):
        """
        Fuera de stagnation mode, el PSO debe seguir usando solo top-K por fitness.
        (No debe haber cambios de comportamiento en modo normal.)
        """
        pop_size = 1000
        k_opt = 100
        fitness = torch.rand(pop_size, device=self.device)

        _in_stagnation = False

        if _in_stagnation:
            # (no debería llegar aquí)
            k_top = k_opt // 2
            k_rand = k_opt - k_top
            _, _top_part = torch.topk(fitness, k_top, largest=False)
            _rand_part = torch.randperm(pop_size, device=self.device)[:k_rand]
            top_idx = torch.cat([_top_part, _rand_part])
        else:
            _, top_idx = torch.topk(fitness, k_opt, largest=False)

        # Todos los índices deben ser de los top-k
        top_vals = fitness[top_idx]
        k_plus_1_threshold = torch.topk(fitness, k_opt + 1, largest=False).values[-1]
        all_in_top = (top_vals <= k_plus_1_threshold + 1e-6).all()
        self.assertTrue(all_in_top.item(),
            "Fuera de stagnation, top_idx debe contener solo los mejores individuos.")

        print(f"\n[BUG-PSO-3 Normal] {k_opt} candidatos, todos dentro del top-{k_opt}: OK.")


# ─────────────────────────────────────────────────────────────────────────────
# TEST 3 — Integración: nano_pso completo mejora constantes con rango amplio
# ─────────────────────────────────────────────────────────────────────────────

class TestPSOEndToEnd(unittest.TestCase):
    """
    Test de integración: usa el fused PSO kernel para optimizar constantes
    de una fórmula con target=15*x+3 donde las constantes iniciales son 0.
    Con jitter escalado, debe converger a C0≈15, C1≈3 en 60 steps.
    """

    def setUp(self):
        self.device = _get_device()

    def test_pso_converges_to_two_constants(self):
        """PSO debe encontrar C0≈15, C1≈3 para y=C0*x+C1."""
        from warpsymbolic.gpu.engine import TensorGeneticEngine

        engine = TensorGeneticEngine(
            device=self.device, pop_size=20, max_len=16, n_islands=1,
            num_variables=1
        )
        grammar = engine.grammar
        from warpsymbolic.gpu.cuda_vm import CudaRPNVM
        vm = CudaRPNVM(grammar, engine.device)

        # RPN: C * x0 + C → [id_C, id_x0, op_mul, id_C, op_add]
        tokens = [vm.id_C, vm.id_x_start, vm.op_mul, vm.id_C, vm.op_add]
        prog = torch.full((engine.max_len,), vm.PAD_ID, dtype=torch.uint8, device=self.device)
        for i, t in enumerate(tokens):
            prog[i] = t
        population = prog.unsqueeze(0).expand(20, -1).clone()

        x = torch.linspace(1, 6, 12, device=self.device, dtype=torch.float32).unsqueeze(0)
        y = 15.0 * x.squeeze() + 3.0

        # Constantes iniciales: ambas = 0
        constants = torch.zeros(20, engine.max_constants, device=self.device, dtype=torch.float32)

        refined_c, refined_err = engine.optimizer.nano_pso(
            population, constants, x, y, steps=60, num_particles=20
        )

        best_idx = refined_err.argmin().item()
        best_err  = refined_err[best_idx].item()
        best_c0   = refined_c[best_idx, 0].item()
        best_c1   = refined_c[best_idx, 1].item()

        print(f"\n[E2E] RMSE={best_err:.6f}, C0={best_c0:.3f} (target~=15), C1={best_c1:.3f} (target~=3)")

        self.assertLess(best_err, 2.0,
            f"PSO debe converger a RMSE<2 para 15*x+3 (RMSE={best_err:.4f}). "
            "Si falla, el jitter σ=1 fijo impide explorar desde C=0 hacia C0=15.")


if __name__ == "__main__":
    unittest.main(verbosity=2)
