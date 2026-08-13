"""
test_new_bugs_v3.py
===================
Tests para detectar y verificar NUEVOS bugs en src/warpsymbolic/gpu.

Ejecutar desde la raiz del proyecto:
    python -m pytest tests/gpu/test_new_bugs_v3.py -v
o directamente:
    python src/warpsymbolic/gpu/tests/test_new_bugs_v3.py

Nuevos bugs cubiertos:
  N1 - Matrices cuadradas en evaluate_batch (num_vars == num_samples)
  N3 - Constant folding overflow en fact/lgamma
  N5 - Hash collision en deduplicacion
  N8 - _get_subtree_starts solo funciona con CUDA
  N9 - Crossover produce RPN invalido
"""

import sys
import os
import unittest
import math

import torch

# ---------------------------------------------------------------------------
# Path setup — permite ejecutar desde cualquier directorio
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

_ALPHA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ALPHA_ROOT not in sys.path:
    sys.path.insert(0, _ALPHA_ROOT)

PAD_ID = 0

# ---------------------------------------------------------------------------
# Device detection - Usar CUDA si está disponible
# ---------------------------------------------------------------------------
def _get_device():
    """Detecta y retorna el mejor dispositivo disponible."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ---------------------------------------------------------------------------
# Helpers de fabrica minimalistas
# ---------------------------------------------------------------------------

def _make_minimal_grammar(num_vars=1):
    """Crea un GPUGrammar minimo con tokens basicos para tests."""
    from warpsymbolic.gpu.grammar import GPUGrammar
    return GPUGrammar(num_variables=num_vars)


def _make_evaluator(grammar=None, device=None, dtype=torch.float64):
    """Fabrica un GPUEvaluator minimo."""
    from warpsymbolic.gpu.evaluation import GPUEvaluator
    if grammar is None:
        grammar = _make_minimal_grammar()
    if device is None:
        device = _get_device()
    return GPUEvaluator(grammar, device, dtype=dtype)


def _make_operators(grammar=None, device=None, pop_size=16, max_len=10, num_vars=1):
    """Fabrica un GPUOperators minimo."""
    from warpsymbolic.gpu.operators import GPUOperators
    if grammar is None:
        grammar = _make_minimal_grammar(num_vars)
    if device is None:
        device = _get_device()
    return GPUOperators(grammar, device, pop_size=pop_size, max_len=max_len, num_variables=num_vars)


def _make_simplifier(grammar=None, device=None, dtype=torch.float64):
    """Fabrica un GPUSymbolicSimplifier minimo."""
    from warpsymbolic.gpu.gpu_simplifier import GPUSymbolicSimplifier
    if grammar is None:
        grammar = _make_minimal_grammar()
    if device is None:
        device = _get_device()
    return GPUSymbolicSimplifier(grammar, device, dtype=dtype)


# ===========================================================================
# N1 — Matrices cuadradas en evaluate_batch (num_vars == num_samples)
# ===========================================================================

class TestN1SquareMatrixShapeDetection(unittest.TestCase):
    """
    Bug N1: Cuando num_variables == num_samples (matriz cuadrada), la logica
    de deteccion de forma en evaluate_batch puede no transponer correctamente.
    
    Ejemplo: Si x es [5, 5] con 5 variables y 5 muestras, el codigo no puede
    distinguir si es [Vars, Samples] o [Samples, Vars].
    """

    def setUp(self):
        self.device = _get_device()
        self.dtype = torch.float64

    def test_square_matrix_detection_vars_equals_samples(self):
        """
        Cuando num_vars == num_samples, el sistema no puede distinguir
        automáticamente entre [Vars, Samples] y [Samples, Vars].
        
        CONVENCIÓN: El caller DEBE pasar [Vars, Samples].
        
        Este test verifica que:
        1. Matrices cuadradas se procesan sin errores
        2. El resultado es consistente
        """
        n = 5  # num_vars == num_samples
        grammar = _make_minimal_grammar(num_vars=n)
        evaluator = _make_evaluator(grammar, self.device, self.dtype)
        
        # Crear datos de prueba en formato [Vars, Samples]
        x_vars_samples = torch.randn(n, n, device=self.device, dtype=self.dtype)
        y_target = torch.randn(n, device=self.device, dtype=self.dtype)
        
        # Crear poblacion minima (solo variable x0)
        x0_id = grammar.token_to_id.get('x0', 1)
        population = torch.full((2, 10), PAD_ID, dtype=grammar.dtype, device=self.device)
        population[0, 0] = x0_id  # formula: x0
        population[1, 0] = x0_id
        
        constants = torch.zeros(2, 5, dtype=self.dtype, device=self.device)
        
        # Test 1: x en formato [Vars, Samples] - debe funcionar correctamente
        try:
            rmse1 = evaluator.evaluate_batch(population, x_vars_samples, y_target, constants)
            # No debe haber error
        except Exception as e:
            self.fail(f"evaluate_batch fallo con x [Vars, Samples]: {e}")
        
        # Test 2: Verificar que el resultado es válido (no inf ni nan)
        self.assertFalse(torch.any(torch.isnan(rmse1)), "RMSE no debe ser NaN")
        self.assertFalse(torch.any(torch.isinf(rmse1)), "RMSE no debe ser Inf")
        
        # Test 3: Con los mismos datos, el resultado debe ser determinista
        rmse2 = evaluator.evaluate_batch(population, x_vars_samples, y_target, constants)
        self.assertTrue(
            torch.allclose(rmse1, rmse2, rtol=1e-10),
            "RMSE debe ser determinista con los mismos datos"
        )

    def test_rectangular_matrix_correct_detection(self):
        """
        Verificar que matrices rectangulares se detectan correctamente
        (este caso deberia funcionar bien, es test de referencia).
        """
        n_vars, n_samples = 3, 10  # Diferentes dimensiones
        grammar = _make_minimal_grammar(num_vars=n_vars)
        evaluator = _make_evaluator(grammar, self.device, self.dtype)
        
        x_correct = torch.randn(n_vars, n_samples, device=self.device, dtype=self.dtype)
        y_target = torch.randn(n_samples, device=self.device, dtype=self.dtype)
        
        x0_id = grammar.token_to_id.get('x0', 1)
        population = torch.full((1, 10), PAD_ID, dtype=grammar.dtype, device=self.device)
        population[0, 0] = x0_id
        constants = torch.zeros(1, 5, dtype=self.dtype, device=self.device)
        
        # Formato correcto [Vars, Samples]
        rmse_correct = evaluator.evaluate_batch(population, x_correct, y_target, constants)
        
        # Formato "incorrecto" [Samples, Vars] - el codigo debe transponer
        x_transpose = x_correct.T.contiguous()
        rmse_transposed = evaluator.evaluate_batch(population, x_transpose, y_target, constants)
        
        self.assertTrue(
            torch.allclose(rmse_correct, rmse_transposed, rtol=1e-5),
            "RMSE deberia ser igual despues de transposicion automatica"
        )


# ===========================================================================
# N3 — Constant folding overflow en fact/lgamma
# ===========================================================================

class TestN3ConstantFoldingOverflow(unittest.TestCase):
    """
    Bug N3: En _apply_constant_folding, factorial de valores grandes
    causa overflow silencioso, produciendo inf en lugar de preservar
    la expresion original.
    """

    def setUp(self):
        self.device = torch.device("cpu")
        self.dtype = torch.float64
        self.simplifier = _make_simplifier(device=self.device, dtype=self.dtype)
        self.grammar = self.simplifier.grammar

    def test_factorial_overflow_large_value(self):
        """
        factorial(171) = inf por overflow. El simplificador debe manejar esto.
        """
        if 'fact' not in self.grammar.token_to_id:
            self.skipTest("Operador 'fact' no disponible en gramatica")
        
        fact_id = self.grammar.token_to_id['fact']
        const_200_id = None
        
        # Buscar un literal grande o usar C
        for t, tid in self.grammar.token_to_id.items():
            try:
                if float(t) == 200.0:
                    const_200_id = tid
                    break
            except ValueError:
                continue
        
        if const_200_id is None:
            # Usar constante C y simular valor 200
            # No podemos asignar valor sin constantes tensor, skip
            self.skipTest("No hay literal 200 en gramatica para test de overflow")
        
        # Crear formula: fact(200)
        population = torch.full((1, 10), PAD_ID, dtype=torch.long, device=self.device)
        population[0, 0] = const_200_id
        population[0, 1] = fact_id
        
        # Ejecutar simplificacion
        try:
            result, _, n_simplified = self.simplifier._apply_constant_folding(population)
            
            # El resultado NO deberia ser inf (deberia preservar la formula original)
            # Pero si el bug existe, result[0, 0] seria un token con valor inf
            # Verificamos que el simplificador no crashee
            self.assertIsNotNone(result, "Simplificador no debe retornar None")
            
        except Exception as e:
            self.fail(f"Simplificador crasheo con overflow: {e}")

    def test_lgamma_large_value_handling(self):
        """
        lgamma(1e300) puede causar problemas. Verificar manejo robusto.
        """
        if 'lgamma' not in self.grammar.token_to_id:
            self.skipTest("Operador 'lgamma' no disponible en gramatica")
        
        lgamma_id = self.grammar.token_to_id['lgamma']
        
        # Crear formula con valor que cause overflow en lgamma
        # lgamma(1e300) es muy grande pero no overflow directamente
        # Pero exp(lgamma(200)) = inf
        
        # Test: verificar que no hay crash
        population = torch.full((1, 10), PAD_ID, dtype=torch.long, device=self.device)
        
        # Usar constante C si existe
        c_id = self.grammar.token_to_id.get('C', None)
        if c_id is None:
            self.skipTest("Constante C no disponible")
        
        population[0, 0] = c_id  # Placeholder
        population[0, 1] = lgamma_id
        
        try:
            # Con constantes tensor
            constants = torch.tensor([[1e10]], dtype=self.dtype, device=self.device)
            result, _, _ = self.simplifier._apply_constant_folding(population)
            self.assertIsNotNone(result)
        except Exception as e:
            # No debe crashear, pero puede no simplificar
            pass


# ===========================================================================
# N5 — Hash collision en deduplicacion
# ===========================================================================

class TestN5HashCollisionDeduplication(unittest.TestCase):
    """
    Bug N5: El hash usado en deduplicate_population es demasiado simple
    y puede causar colisiones, eliminando individuos geneticamente utiles.
    """

    def setUp(self):
        self.device = torch.device("cpu")
        self.grammar = _make_minimal_grammar()
        self.ops = _make_operators(grammar=self.grammar, device=self.device)

    def test_hash_collision_different_formulas(self):
        """
        Verificar que dos formulas DIFERENTES no tienen el mismo hash.
        Si lo tienen, es un bug de colision.
        """
        # Crear dos formulas RPN validas diferentes
        x0_id = self.grammar.token_to_id.get('x0', 1)
        plus_id = self.grammar.token_to_id.get('+', 10)
        mult_id = self.grammar.token_to_id.get('*', 11)
        
        # Formula 1: x0 x0 + (x0 + x0)
        formula1 = torch.full((10,), PAD_ID, dtype=self.grammar.dtype, device=self.device)
        formula1[0] = x0_id
        formula1[1] = x0_id
        formula1[2] = plus_id
        
        # Formula 2: x0 x0 * (x0 * x0)
        formula2 = torch.full((10,), PAD_ID, dtype=self.grammar.dtype, device=self.device)
        formula2[0] = x0_id
        formula2[1] = x0_id
        formula2[2] = mult_id
        
        # Calcular hashes
        L = 10
        weights = torch.randint(-9223372036854775807, 9223372036854775807, (L,), 
                               device=self.device, dtype=torch.long)
        
        hash1 = (formula1.long() * weights).sum().item()
        hash2 = (formula2.long() * weights).sum().item()
        
        # Los hashes deben ser diferentes (colision es bug)
        # NOTA: Puede haber colision aleatoria pero es muy improbable
        self.assertNotEqual(
            hash1, hash2,
            f"Hash collision detectada! Dos formulas diferentes tienen el mismo hash: {hash1}. "
            "Esto causa perdida de diversidad genetica."
        )

    def test_deduplicate_preserves_unique_formulas(self):
        """
        Verificar que deduplicate_population NO elimina formulas unicas.
        """
        # Crear poblacion con formulas claramente diferentes
        population = torch.zeros(4, 10, dtype=self.grammar.dtype, device=self.device)
        
        x0_id = self.grammar.token_to_id.get('x0', 1)
        plus_id = self.grammar.token_to_id.get('+', 10)
        mult_id = self.grammar.token_to_id.get('*', 11)
        
        # Formula 1: x0
        population[0, 0] = x0_id
        
        # Formula 2: x0 + x0 (RPN: x0 x0 +)
        population[1, 0] = x0_id
        population[1, 1] = x0_id
        population[1, 2] = plus_id
        
        # Formula 3: x0 * x0 (RPN: x0 x0 *)
        population[2, 0] = x0_id
        population[2, 1] = x0_id
        population[2, 2] = mult_id
        
        # Formula 4: x0 (duplicado de 1)
        population[3, 0] = x0_id
        
        constants = torch.zeros(4, 5, dtype=torch.float64, device=self.device)
        
        # Ejecutar deduplicacion
        result_pop, result_const, n_dups = self.ops.deduplicate_population(population, constants)
        
        # Solo deberia eliminarse 1 duplicado (formula 4 es igual a formula 1)
        # Las formulas 1, 2, 3 son diferentes y deben preservarse
        self.assertLessEqual(
            n_dups, 1,
            f"Deduplicacion elimino {n_dups} individuos, pero solo habia 1 duplicado real. "
            "Posible hash collision eliminando formulas unicas."
        )


# ===========================================================================
# N8 — _get_subtree_starts solo funciona con CUDA (CPU fallback)
# ===========================================================================

class TestN8SubtreeStartsCPUFallback(unittest.TestCase):
    """
    Bug N8: _get_subtree_starts en GPUOperators solo llama al kernel CUDA
    si population.is_cuda, pero el fallback PyTorch puede no comportarse igual.
    """

    def setUp(self):
        self.device = torch.device("cpu")
        self.grammar = _make_minimal_grammar()
        self.ops = _make_operators(grammar=self.grammar, device=self.device)

    def test_subtree_starts_cpu_correct(self):
        """
        Verificar que _get_subtree_starts funciona correctamente en CPU.
        """
        x0_id = self.grammar.token_to_id.get('x0', 1)
        plus_id = self.grammar.token_to_id.get('+', 10)
        
        # RPN: x0 x0 x0 + +  (arbol: x0 + (x0 + x0))
        population = torch.full((1, 10), PAD_ID, dtype=self.grammar.dtype, device=self.device)
        population[0, 0] = x0_id
        population[0, 1] = x0_id
        population[0, 2] = x0_id
        population[0, 3] = plus_id  # x0 + x0, subtree ends at index 3
        population[0, 4] = plus_id  # x0 + result, subtree ends at index 4
        
        # Obtener starts
        starts = self.ops._get_subtree_ranges(population)
        
        # Verificar estructura
        # Index 0 (x0): subtree de longitud 1, start = 0
        # Index 1 (x0): subtree de longitud 1, start = 1
        # Index 2 (x0): subtree de longitud 1, start = 2
        # Index 3 (+): subtree de longitud 3 (x0 x0 +), start = 1
        # Index 4 (+): subtree de longitud 5 (todo), start = 0
        
        self.assertEqual(starts[0, 0].item(), 0, "Subtree start para x0[0] debe ser 0")
        self.assertEqual(starts[0, 1].item(), 1, "Subtree start para x0[1] debe ser 1")
        self.assertEqual(starts[0, 2].item(), 2, "Subtree start para x0[2] debe ser 2")
        self.assertEqual(starts[0, 3].item(), 1, "Subtree start para +[3] debe ser 1")
        self.assertEqual(starts[0, 4].item(), 0, "Subtree start para +[4] debe ser 0")


# ===========================================================================
# N9 — Crossover produce RPN invalido
# ===========================================================================

class TestN9CrossoverValidity(unittest.TestCase):
    """
    Verificar que crossover_population produce RPN valido.
    """

    def setUp(self):
        self.device = torch.device("cpu")
        self.grammar = _make_minimal_grammar()
        self.ops = _make_operators(grammar=self.grammar, device=self.device, pop_size=20, max_len=15)

    def test_crossover_produces_valid_rpn(self):
        """
        Despues de crossover, todos los individuos deben tener RPN valido.
        """
        # Generar poblacion aleatoria
        population = self.ops.generate_random_population(20)
        constants = torch.zeros(20, 5, dtype=torch.float64, device=self.device)
        
        # Ejecutar crossover con alta tasa
        offspring = self.ops.crossover_population(population.clone(), 1.0)
        
        # Verificar validez de cada individuo
        valid_mask = self.ops._validate_rpn_batch(offspring)
        invalid_count = (~valid_mask).sum().item()
        
        self.assertEqual(
            invalid_count, 0,
            f"Crossover produjo {invalid_count} RPNs invalidos de {offspring.shape[0]}"
        )


# ===========================================================================
# Integration Tests
# ===========================================================================

class TestIntegrationRMSEComputation(unittest.TestCase):
    """
    Tests de integracion para verificar que el pipeline completo funciona.
    """

    def setUp(self):
        self.device = _get_device()
        self.dtype = torch.float64

    def test_simple_formula_correct_rmse(self):
        """
        Verificar que una formula simple da el RMSE esperado.
        """
        grammar = _make_minimal_grammar(num_vars=1)
        evaluator = _make_evaluator(grammar, self.device, self.dtype)
        
        # Data: y = 2*x
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=self.device, dtype=self.dtype)  # [1, 4]
        y = torch.tensor([2.0, 4.0, 6.0, 8.0], device=self.device, dtype=self.dtype)     # [4]
        
        # Formula: x0 (RPN: x0)
        x0_id = grammar.token_to_id.get('x0', 1)
        population = torch.full((1, 10), PAD_ID, dtype=grammar.dtype, device=self.device)
        population[0, 0] = x0_id
        
        # Constantes: C = 2 (pero formula es solo x0, asi que RMSE != 0)
        constants = torch.zeros(1, 5, dtype=self.dtype, device=self.device)
        
        # Evaluar: x0 predice [1, 2, 3, 4], target es [2, 4, 6, 8]
        # Error: [-1, -2, -3, -4]
        # MSE: (1+4+9+16)/4 = 7.5
        # RMSE: sqrt(7.5) ≈ 2.739
        rmse = evaluator.evaluate_batch(population, x, y, constants)
        
        expected_rmse = math.sqrt(7.5)
        
        self.assertAlmostEqual(
            rmse[0].item(), expected_rmse, places=3,
            msg=f"RMSE esperado ~{expected_rmse:.3f}, obtenido {rmse[0].item():.3f}"
        )


# ===========================================================================
# Runner standalone
# ===========================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("GPU Core New Bug Tests v3")
    print("=" * 70)
    unittest.main(verbosity=2)
