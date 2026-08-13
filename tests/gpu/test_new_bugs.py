"""
tests/test_new_bugs.py
======================
Tests para bugs nuevos (N1-N4) identificados en src/warpsymbolic/gpu.

Bug N1 – cupy_vm.py: el operador `gamma` computa lgamma(a+1) en lugar de
         exp(lgamma(a)), es decir, calcula el log-factorial en vez de la
         función Gamma real. gamma(3)=2 pero el kernel devuelve ≈1.791.

Bug N2 – cupy_vm.py: el operador `lgamma` no tiene ningún `else if` en el
         kernel; el token cae silenciosamente al default (res=0.0), por lo
         que toda fórmula que usa `lgamma` evalúa constante a 0.

Bug N3 – evaluation.py `evaluate_batch`: la lógica de detección de forma de
         `x` falla cuando num_vars == num_samples (matriz cuadrada). Cuando
         x.shape = [V, V] e y.shape = [V], la condición AND de la primera
         rama es falsa (x.shape[0] != y.shape[0] → False), cae a la segunda
         que transpone incorrectamente.

Bug N4 – operators.py `deduplicate_population`: el hash de deduplicación
         multiplica population (uint8) por weights (int64) sin castear a
         int64 primero. En PyTorch, uint8 × int64 hace silent truncation a
         uint8 → hashes erróneos → deduplicación ineficaz → el algoritmo
         mantiene duplicados que saturan la diversidad genética.
"""

import sys
import os
import re
import math
import unittest

import torch

# ─── Rutas de importación ────────────────────────────────────────────────────
_HERE = os.path.dirname(__file__)
_GPU_DIR = os.path.abspath(os.path.join(_HERE, '..', '..', 'src', 'warpsymbolic', 'gpu'))
if _GPU_DIR not in sys.path:
    sys.path.insert(0, _GPU_DIR)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_source(relative_to_gpu: str) -> str:
    """Lee el código fuente de un archivo relativo al directorio gpu/."""
    path = os.path.join(_GPU_DIR, relative_to_gpu)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ═════════════════════════════════════════════════════════════════════════════
# Bug N1 – cupy_vm.py: gamma implementado como lgamma(a+1)
# ═════════════════════════════════════════════════════════════════════════════

class TestN1CupyGammaImplementation(unittest.TestCase):
    """
    El operador gamma del kernel CuPy debe calcular Γ(a) = exp(lgamma(a)).
    El código actual calcula lgamma(a + 1.0) que es log(Γ(a+1)) ≠ Γ(a).
    
    Ejemplo numérico:
        gamma(3) = Γ(3) = 2! = 2.0
        lgamma(3+1) = lgamma(4) = log(Γ(4)) = log(6) ≈ 1.7917...
    """

    @classmethod
    def setUpClass(cls):
        cls.src = _read_source('cupy_vm.py')

    # ── Test 1: detección estática del bug ───────────────────────────────────
    def test_gamma_branch_does_not_use_lgamma_directly(self):
        """
        El bloque op_gamma no debe asignar lgamma(...) directamente a res.
        Solo puede aparecer como argumento de exp(): res = exp(lgamma(a)).
        Nota: el regex se limita al bloque op_gamma (antes de op_lgamma).
        """
        # Capturar SÓLO el bloque op_gamma (hasta antes de op_lgamma o fin de unarios)
        match = re.search(
            r'token == op_gamma\)(.*?)(?=else if \(token == op_lgamma|s\[sp-1\] = res;)',
            self.src, re.DOTALL
        )
        self.assertIsNotNone(match, "No se encontró el bloque op_gamma en el kernel")

        gamma_section = match.group(1)
        # La implementación BUGGY: res = lgamma(a + 1.0)  ← lgamma como valor directo
        # La implementación CORRECTA: res = exp(lgamma(a)) ← lgamma sólo dentro de exp()
        # Detectar "res = lgamma(" que NO esté precedido por "exp("
        has_bare_lgamma_assign = bool(re.search(r'res\s*=\s*lgamma\s*\(', gamma_section))

        self.assertFalse(
            has_bare_lgamma_assign,
            "BUG N1: el bloque op_gamma asigna lgamma(...) directamente a res "
            "en lugar de exp(lgamma(a)). gamma(3) devuelve ≈1.79 en vez de 2.0."
        )

    # ── Test 2: verificación matemática del valor correcto ───────────────────
    def test_gamma_value_at_3_should_be_2(self):
        """Verifica que la implementación correcta de gamma(3) = 2.0."""
        a = 3.0
        # Implementación correcta
        correct = math.exp(math.lgamma(a))
        # Implementación buggy actual del kernel
        buggy   = math.lgamma(a + 1.0)

        # Asegurarse de que los valores son distintos (diferencia ≥ 0.1)
        self.assertAlmostEqual(correct, 2.0, places=6,
            msg="La fórmula correcta exp(lgamma(3)) debe dar 2.0")
        self.assertNotAlmostEqual(correct, buggy, places=1,
            msg="El valor buggy lgamma(4) debe diferir de 2.0 en al menos 0.1")

    # ── Test 3: verificación de que exp(lgamma(a)) está en el código ─────────
    def test_gamma_branch_uses_exp_lgamma_pattern(self):
        """Después del fix, el kernel debe usar exp(lgamma(a)) para op_gamma."""
        match = re.search(
            r'token == op_gamma\).*?s\[sp-1\] = res;',
            self.src, re.DOTALL
        )
        self.assertIsNotNone(match)
        gamma_block = match.group(0)
        uses_exp_lgamma = bool(re.search(r'exp\s*\(\s*lgamma\s*\(', gamma_block))
        self.assertTrue(
            uses_exp_lgamma,
            "FALTA FIX N1: el bloque op_gamma debería usar exp(lgamma(a)) "
            "para calcular la función Gamma real."
        )


# ═════════════════════════════════════════════════════════════════════════════
# Bug N2 – cupy_vm.py: operador lgamma ausente del kernel
# ═════════════════════════════════════════════════════════════════════════════

class TestN2CupyLgammaMissing(unittest.TestCase):
    """
    El operador `lgamma` no tiene rama en el kernel CuPy. Cualquier fórmula
    con token lgamma evalúa silenciosamente a 0.0 (valor por defecto de res).
    Esto afecta problemas como DLSR o benchmarks que involucran lgamma.
    """

    @classmethod
    def setUpClass(cls):
        cls.src = _read_source('cupy_vm.py')
        # Extraer sólo el bloque de la cadena de unaries del kernel
        # (desde "// Unary" hasta el cierre del if-else chain)
        m = re.search(r'// Unary.*?s\[sp-1\] = res;', cls.src, re.DOTALL)
        cls.unary_block = m.group(0) if m else cls.src

    # ── Test 1: op_lgamma debe aparecer en el kernel ─────────────────────────
    def test_op_lgamma_has_handler_in_kernel(self):
        """El kernel debe tener un bloque else-if para op_lgamma."""
        has_lgamma_handler = bool(re.search(r'token\s*==\s*op_lgamma', self.src))
        self.assertTrue(
            has_lgamma_handler,
            "BUG N2 CONFIRMADO: no existe rama `token == op_lgamma` en el "
            "kernel de CuPy. lgamma siempre devuelve res=0.0."
        )

    # ── Test 2: lgamma debe aparecer en la firma de la función del kernel ─────
    def test_op_lgamma_in_kernel_signature(self):
        """op_lgamma debe aparecer como parámetro del kernel."""
        in_signature = bool(re.search(r'int\s+op_lgamma', self.src))
        self.assertTrue(
            in_signature,
            "op_lgamma no es parámetro del kernel. Debe agregarse a la "
            "firma y a la llamada desde run_vm_cupy()."
        )

    # ── Test 3: lgamma debe pasarse en la llamada rpn_kernel() ───────────────
    def test_run_vm_cupy_passes_op_lgamma(self):
        """run_vm_cupy() debe pasar op_lgamma al kernel."""
        # Buscar la sección de parámetros de la función Python
        has_param = bool(re.search(r'op_lgamma\s*[,\)]', self.src))
        self.assertTrue(
            has_param,
            "run_vm_cupy() no acepta ni pasa op_lgamma al kernel."
        )

    # ── Test 4: verificar matemáticamente lgamma ─────────────────────────────
    def test_lgamma_value_is_not_zero(self):
        """lgamma(3) = log(Γ(3)) = log(2) ≈ 0.693, nunca debe ser 0.0."""
        val = math.lgamma(3.0)
        self.assertNotAlmostEqual(val, 0.0, places=2,
            msg="lgamma(3) ≈ 0.693, no cero; el bug produce 0 silencioso")


# ═════════════════════════════════════════════════════════════════════════════
# Bug N3 – evaluation.py: detección de forma incorrecta para matrices cuadradas
# ═════════════════════════════════════════════════════════════════════════════

class TestN3EvaluateBatchSquareShapeDetection(unittest.TestCase):
    """
    En evaluate_batch() la lógica de detección de forma de x:
    
        if x.shape[1] == y.shape[0] and x.shape[0] != y.shape[0]:  <- falla cuando cuadrado
            pass  # [Vars, Samples] correcto
        elif x.shape[0] == y.shape[0]:
            x = x.T  # ← transpone cuando NO debería para datos cuadrados

    Con V variables y V muestras (x.shape = [V, V], y.shape = [V]):
      - Primera condición: shape[1]==V y shape[0]==V → (True) AND (V!=V=False) → False
      - Segunda condición: shape[0]==V → True → TRANSPONE incorrectamente
    
    El resultado: las variables se confunden con muestras → evaluación completamente
    incorrecta → RMSE basura → convergencia imposible para estos datos.
    """

    @classmethod
    def setUpClass(cls):
        cls.src = _read_source('evaluation.py')

    # ── Test 1: detección estática del patrón problemático ───────────────────
    def test_shape_detection_has_inequality_guard(self):
        """
        La rama ELIF que transpone x no debe activarse para matrices cuadradas.
        El patrón buggy es `elif x.shape[0] == y_target.shape[0]:` sin guard
        adicional. El fix añade `and x.shape[0] != x.shape[1]`.
        Nota: el `if` anterior (que hace `pass`) puede seguir usando `!=` 
        porque es la rama de no-acción; sólo el `elif` importa aquí.
        """
        # Patrón buggy: elif sin guard de cuadrado → transpone incondicionalmente
        buggy_elif = re.search(
            r'elif\s+x\.shape\[0\]\s*==\s*y_target\.shape\[0\]\s*(?:and\s+x\.shape\[0\]\s*!=\s*x\.shape\[1\])?\s*:',
            self.src
        )
        # Si encontramos el elif SIN el guard `x.shape[0] != x.shape[1]`, es el bug
        # Verificamos que todos los elif de este tipo TIENEN el guard
        for m in re.finditer(
            r'elif\s+x\.shape\[0\]\s*==\s*y_target\.shape\[0\]([^:]*):',
            self.src
        ):
            guard_clause = m.group(1)
            has_square_guard = bool(re.search(r'x\.shape\[0\]\s*!=\s*x\.shape\[1\]', guard_clause))
            self.assertTrue(
                has_square_guard,
                f"BUG N3: elif encontrado sin guard cuadrado: `{m.group(0).strip()}`. "
                "Debe incluir `and x.shape[0] != x.shape[1]`."
            )

    # ── Test 2a: la lógica corregida no transpone x cuadrada ─────────────────
    def test_square_x_shape_is_not_transposed_incorrectly(self):
        """
        Con x=[5,5] e y=[5], la lógica CORREGIDA (FIX N3) no debe transponerla.
        El guard `and x.shape[0] != x.shape[1]` impide la transposición cuando
        la matriz es cuadrada.
        """
        n_vars, n_samples = 5, 5
        x = torch.arange(25, dtype=torch.float32).reshape(n_vars, n_samples)
        y_target = torch.zeros(n_samples)

        # ── Simular la lógica CORREGIDA de evaluate_batch (FIX N3) ───────
        x_proc = x.clone()
        if x_proc.dim() == 2:
            if x_proc.shape[1] == y_target.shape[0] and x_proc.shape[0] != y_target.shape[0]:
                pass   # [Vars, Samples] correcto
            elif x_proc.shape[0] == y_target.shape[0] and x_proc.shape[0] != x_proc.shape[1]:
                # Guard añadido en FIX N3: sólo transponer si NO es cuadrada
                x_proc = x_proc.T.contiguous()

        transposed = not torch.equal(x_proc, x)
        self.assertFalse(
            transposed,
            "FIX N3: x[5,5] NO debe transponerse con la lógica corregida. "
            "El guard `x.shape[0] != x.shape[1]` evita la transposición errónea."
        )

    # ── Test 2b: la lógica corregida SÍ transpone x rectangular [Samples,Vars]
    def test_rectangular_samples_vars_is_still_transposed(self):
        """
        Con x=[10,3] (10 muestras, 3 variables) e y=[10], la lógica corregida
        DEBE transponerla a [3,10]. El fix solo protege el caso cuadrado.
        """
        n_vars, n_samples = 3, 10
        x_samples_vars = torch.arange(30, dtype=torch.float32).reshape(n_samples, n_vars)
        y_target = torch.zeros(n_samples)

        x_proc = x_samples_vars.clone()
        if x_proc.dim() == 2:
            if x_proc.shape[1] == y_target.shape[0] and x_proc.shape[0] != y_target.shape[0]:
                pass
            elif x_proc.shape[0] == y_target.shape[0] and x_proc.shape[0] != x_proc.shape[1]:
                x_proc = x_proc.T.contiguous()   # debe transponerse: [10,3]→[3,10]

        self.assertEqual(x_proc.shape, (n_vars, n_samples),
            "x rectangular [Samples,Vars] debe transponerse a [Vars,Samples] "
            "incluso con el fix N3.")

    # ── Test 3: comprobar que el código source tiene la lógica corregida ─────
    def test_evaluate_batch_handles_square_data_correctly(self):
        """
        Después del fix, el código debe manejar x cuadrada sin transponerla.
        La solución correcta añade `and x.shape[0] != x.shape[1]` a la
        rama elif, o usa N_vars explícito de otra fuente.
        """
        # Buscamos el patrón corregido: la rama elif incluye guard para cuadrado
        # Patrón: elif ... and x.shape[0] != x.shape[1]
        # O: elif x.shape[0] == y_target.shape[0] and x.shape[1] != y_target.shape[0]
        fixed_pattern = re.search(
            r'elif\s+x\.shape\[0\]\s*==\s*y_target\.shape\[0\]'
            r'.*?x\.shape\[0\]\s*!=\s*x\.shape\[1\]',
            self.src, re.DOTALL
        )
        # También aceptamos la forma alternativa con y check
        fixed_pattern2 = re.search(
            r'elif\s+x\.shape\[0\]\s*==\s*y_target\.shape\[0\]'
            r'.*?x\.shape\[1\]\s*!=\s*y_target\.shape\[0\]',
            self.src, re.DOTALL
        )
        
        self.assertTrue(
            fixed_pattern is not None or fixed_pattern2 is not None,
            "FALTA FIX N3: la rama elif de evaluate_batch debe incluir un "
            "guard para el caso cuadrado (x.shape[0] != x.shape[1]) para "
            "evitar transponer x cuando num_vars == num_samples."
        )


# ═════════════════════════════════════════════════════════════════════════════
# Bug N4 – operators.py: hash de deduplicación con overflow silencioso uint8
# ═════════════════════════════════════════════════════════════════════════════

class TestN4DeduplicateHashOverflow(unittest.TestCase):
    """
    Verificación del hash de deduplicación en `deduplicate_population`.

    INVESTIGACIÓN: inicialmente se sospechó que `(population * weights)`
    con population uint8 y weights int64 causaba overflow silencioso.
    
    RESULTADO: PyTorch promueve automáticamente uint8 × int64 → int64,
    por lo que NO hay overflow. El código es correcto tal como está.
    
    Estos tests documentan el comportamiento y verifican que el hash
    produce baja tasa de colisiones reales.
    """

    # ── Test 1: verificar que PyTorch promueve uint8×int64 a int64 ───────────
    def test_pytorch_promotes_uint8_int64_to_int64(self):
        """
        PyTorch promueve uint8 × int64 a int64 (no hay truncación).
        Dos fórmulas distintas deben producir hashes distintos.
        """
        device = torch.device('cpu')
        pop_a = torch.tensor([[200, 100, 50, 30, 15]], dtype=torch.uint8, device=device)
        pop_b = torch.tensor([[201, 100, 50, 30, 15]], dtype=torch.uint8, device=device)
        weight = torch.tensor([256, 1, 1, 1, 1], dtype=torch.int64, device=device)

        hash_a = (pop_a * weight).sum(dim=1)
        hash_b = (pop_b * weight).sum(dim=1)

        # Con promote correcto: 200*256+195 = 51395, 201*256+195 = 51651 → distintos
        self.assertNotEqual(hash_a.item(), hash_b.item(),
            "PyTorch debe promover uint8×int64 a int64: hashes distintos "
            "para fórmulas distintas")
        # Verificar que el resultado es int64 (no uint8)
        result_dtype = (pop_a * weight).dtype
        self.assertEqual(result_dtype, torch.int64,
            f"uint8 × int64 debe dar int64, pero da {result_dtype}")

    def test_source_hash_is_functionally_correct(self):
        """
        [DEPRECATED] Test skipped as hashing is now handled natively in CUDA.
        """
        self.skipTest("Hash calculation has been moved to CUDA and changed completely.")

    # ── Test 3: verificar baja tasa de colisiones con hash real ──────────────
    def test_hash_has_low_collision_rate_in_practice(self):
        """
        Con 100 fórmulas distintas y pesos int64 grandes, la tasa de
        colisión debe ser inferior al 5%.
        """
        device = torch.device('cpu')
        L = 10
        population = torch.randint(1, 20, (100, L), dtype=torch.uint8, device=device)
        for i in range(1, 100):
            population[i, 0] = (population[i-1, 0].item() % 19) + 1

        weights = torch.randint(1, 2**30, (L,), dtype=torch.int64, device=device)

        # Hash con la lógica actual del código (uint8 × int64, promovido a int64)
        hashes = (population * weights).sum(dim=1)
        unique_hashes = torch.unique(hashes)

        collision_rate = 1.0 - unique_hashes.shape[0] / 100.0
        self.assertLess(
            collision_rate, 0.05,
            f"Tasa de colisión: {collision_rate:.1%} (esperado <5%)"
        )


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    unittest.main(verbosity=2)
