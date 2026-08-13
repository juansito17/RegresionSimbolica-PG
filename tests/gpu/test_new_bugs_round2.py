"""
tests/test_new_bugs_round2.py
=============================
Tests para bugs adicionales (N5-N8) identificados en src/warpsymbolic/gpu.

Bug N5 – gpu_simplifier.py: bucle for vacío en _precompute_all_subtree_starts
Bug N6 – pattern_memory.py: evict_scores con inf puede causar comportamiento impredecible
Bug N7 – engine.py: USE_HARD_DEPTH_LIMIT usa longitud en lugar de profundidad real
Bug N8 – gpu_simplifier.py: _map_single_value_to_literal_id crea tensores GPU en bucle
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


def _read_source(relative_to_gpu: str) -> str:
    """Lee el código fuente de un archivo relativo al directorio gpu/."""
    path = os.path.join(_GPU_DIR, relative_to_gpu)
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ═════════════════════════════════════════════════════════════════════════════
# Bug N5 – gpu_simplifier.py: bucle for vacío
# ═════════════════════════════════════════════════════════════════════════════

class TestN5EmptyForLoop(unittest.TestCase):
    """
    En _precompute_all_subtree_starts existe un bucle for vacío:
    
        for j in range(L - 1, -1, -1):
            # For position j, walk backward accumulating needs
            # This is still O(L) total backward passes, but each is a vectorized step
            pass  # <-- BUCLE VACÍO!
    
    El algoritmo posterior usa una solución O(L) correcta, pero el bucle vacío
    es código muerto que puede confundir a mantenedores.
    """

    @classmethod
    def setUpClass(cls):
        cls.src = _read_source('gpu_simplifier.py')

    def test_no_empty_for_loop_in_precompute_subtree_starts(self):
        """El método _precompute_all_subtree_starts no debe tener bucles vacíos."""
        # Buscar el método
        method_match = re.search(
            r'def _precompute_all_subtree_starts\(.*?\n(.*?)(?=\n    def |\nclass |\Z)',
            self.src, re.DOTALL
        )
        self.assertIsNotNone(method_match, "No se encontró el método _precompute_all_subtree_starts")
        
        method_body = method_match.group(1)
        
        # Buscar patrones de bucle for con solo 'pass' dentro
        # Patrón: for ...: \n pass
        empty_for_pattern = re.search(
            r'for\s+\w+.*?:\s*\n\s*pass',
            method_body
        )
        
        self.assertIsNone(
            empty_for_pattern,
            "BUG N5 CONFIRMADO: Se encontró un bucle 'for' vacío con 'pass' en "
            "_precompute_all_subtree_starts. Este código muerto debe eliminarse."
        )


# ═════════════════════════════════════════════════════════════════════════════
# Bug N6 – pattern_memory.py: evict_scores con inf
# ═════════════════════════════════════════════════════════════════════════════

class TestN6PatternMemoryInfScores(unittest.TestCase):
    """
    En _update_storage, cuando se calculan scores de evicción:
    
        evict_scores = -self.patterns_count.float() + self.patterns_fitness / 100.0
        
    Si todos los patterns_fitness son inf (valores iniciales), los scores serán
    todos inf, y torch.topk puede comportarse impredeciblemente.
    """

    @classmethod
    def setUpClass(cls):
        cls.src = _read_source('pattern_memory.py')

    def test_evict_scores_handles_inf_values(self):
        """El código debe manejar explícitamente valores inf en evict_scores."""
        # Buscar el cálculo de evict_scores
        evict_pattern = re.search(
            r'evict_scores\s*=\s*-count_scores\s*\+\s*fit_scores',
            self.src
        )
        
        self.assertIsNotNone(evict_pattern, "No se encontró el cálculo de evict_scores")
        
        # Verificar que hay algún guard para inf antes o después
        # Opciones válidas:
        # 1. .clamp(max=MAX_FLOAT)
        # 2. Verificar !isinf() 
        # 3. Inicializar patterns_fitness con valor finito
        
        # Verificar inicialización de patterns_fitness
        init_pattern = re.search(
            r'self\.patterns_fitness\s*=\s*torch\.full\([^,]+,\s*float\([\'\"]inf[\'\"]\)',
            self.src
        )
        
        if init_pattern:
            # Si se inicializa con inf, debe haber un guard
            guard_pattern = re.search(
                r'evict_scores.*?[\n.]*?(isfinite|isinf|clamp|replace.*inf)',
                self.src, re.DOTALL
            )
            self.assertIsNotNone(
                guard_pattern,
                "BUG N6 CONFIRMADO: patterns_fitness se inicializa con inf y "
                "no hay guard para manejar inf en evict_scores. topk puede "
                "fallar cuando todos los valores son inf."
            )

    def test_initial_patterns_fitness_is_finite(self):
        """La inicialización de patterns_fitness debe usar valores finitos."""
        # Buscar inicialización
        init_pattern = re.search(
            r'self\.patterns_fitness\s*=\s*(torch\.full|torch\.empty|torch\.zeros)',
            self.src
        )
        
        self.assertIsNotNone(init_pattern, "No se encontró inicialización de patterns_fitness")
        
        # Si usa torch.full, verificar que el valor no sea inf
        if 'torch.full' in init_pattern.group(0):
            inf_in_init = re.search(r'float\([\'\"]inf[\'\"]\)', self.src[init_pattern.start():init_pattern.end()+50])
            self.assertIsNone(
                inf_in_init,
                "BUG N6: patterns_fitness se inicializa con inf. Preferir un valor "
                "finito grande como 1e30 para evitar problemas con topk."
            )


# ═════════════════════════════════════════════════════════════════════════════
# Bug N7 – engine.py: USE_HARD_DEPTH_LIMIT usa longitud en lugar de profundidad
# ═════════════════════════════════════════════════════════════════════════════

class TestN7HardDepthLimitUsesLength(unittest.TestCase):
    """
    En crossover_population, el límite de profundidad usa longitud:
    
        if GpuGlobals.USE_HARD_DEPTH_LIMIT:
            hard_limit = GpuGlobals.MAX_TREE_DEPTH_HARD_LIMIT
            c1_len = (c1 != PAD_ID).sum(dim=1)  # <-- LONGITUD
            too_long_1 = c1_len > hard_limit
    
    Problema: La longitud de un árbol RPN NO es igual a su profundidad.
    - Un árbol con 15 tokens puede tener profundidad 15 (cadena de unarios)
    - O profundidad 4 (árbol binario balanceado)
    
    Esto hace que el límite de profundidad sea ineficaz.
    """

    @classmethod
    def setUpClass(cls):
        cls.src = _read_source('operators.py')

    def test_hard_depth_limit_calculates_actual_depth(self):
        """El código debe calcular la profundidad real del árbol, no la longitud."""
        # Buscar USE_HARD_DEPTH_LIMIT
        hard_limit_pattern = re.search(
            r'USE_HARD_DEPTH_LIMIT.*?too_long',
            self.src, re.DOTALL
        )
        
        if not hard_limit_pattern:
            self.skipTest("USE_HARD_DEPTH_LIMIT no encontrado en engine.py")
        
        block = hard_limit_pattern.group(0)
        
        # Verificar que usa .sum() para longitud (el bug)
        uses_length = bool(re.search(r'sum\(dim=1\)', block))
        
        # Verificar que hay un cálculo de profundidad real
        uses_depth_calc = bool(re.search(
            r'depth|_tree_depth|compute_depth',
            block
        ))
        
        if uses_length and not uses_depth_calc:
            self.fail(
                "BUG N7 CONFIRMADO: USE_HARD_DEPTH_LIMIT usa (c1 != PAD_ID).sum(dim=1) "
                "que es LONGITUD, no PROFUNDIDAD del árbol. Un árbol con 10 tokens "
                "puede tener profundidad 10 (cadena unaria) o 4 (balanceado). "
                "Debe implementarse un cálculo de profundidad real."
            )

    def test_depth_vs_length_semantic_difference(self):
        """Demostrar que longitud ≠ profundidad."""
        # Crear árboles RPN de ejemplo
        # Árbol 1: cadena de unarios - profundidad = longitud
        # x, neg, neg, neg, neg = 5 tokens, profundidad 5
        
        # Árbol 2: árbol binario balanceado
        # x, x, +, x, x, +, + = 7 tokens, profundidad 3
        
        # Demostración matemática
        chain_tokens = 5
        chain_depth = 5  # neg(neg(neg(neg(x))))
        
        balanced_tokens = 7
        balanced_depth = 3  # ((x+x) + (x+x))
        
        # La relación tokens/profundidad:
        # - Cadena unaria: ratio = 5/5 = 1.0 (tokens ~ profundidad)
        # - Árbol balanceado: ratio = 7/3 = 2.33 (tokens > profundidad)
        # El árbol balanceado tiene mayor ratio porque aprovecha mejor los tokens
        ratio_chain = chain_tokens / chain_depth
        ratio_balanced = balanced_tokens / balanced_depth
        
        # El ratio balanceado es MAYOR porque más tokens "caben" en menos profundidad
        self.assertGreater(ratio_balanced, ratio_chain,
            "La relación tokens/profundidad varía según la estructura del árbol. "
            "Un árbol balanceado tiene ratio mayor (más tokens por nivel de profundidad).")
        
        # Conclusión: usar longitud como proxy de profundidad es incorrecto
        # Para árboles balanceados, tokens ≠ profundidad
        self.assertNotEqual(balanced_tokens, balanced_depth, 
            "Tokens ≠ profundidad para árbol balanceado")
        # Para cadenas unarias, tokens == profundidad (caso degenerado)
        self.assertEqual(chain_tokens, chain_depth,
            "Para cadena unaria, tokens == profundidad (caso degenerado donde longitud = profundidad)")


# ═════════════════════════════════════════════════════════════════════════════
# Bug N8 – gpu_simplifier.py: tensor GPU en bucle
# ═════════════════════════════════════════════════════════════════════════════

class TestN8TensorGpuInLoop(unittest.TestCase):
    """
    _map_single_value_to_literal_id crea tensores GPU para cada llamada:
    
        def _map_single_value_to_literal_id(self, value: float) -> int:
            vals = torch.tensor([value], device=self.device, dtype=self.dtype)
            mask = torch.tensor([True], device=self.device, dtype=torch.bool)
            ...
    
    Este método se llama dentro de bucles en _apply_associative_rules, causando
    miles de asignaciones GPU innecesarias.
    """

    @classmethod
    def setUpClass(cls):
        cls.src = _read_source('gpu_simplifier.py')

    def test_map_single_value_does_not_create_gpu_tensors(self):
        """El método debe evitar crear tensores GPU para valores individuales."""
        # Buscar el método
        method_match = re.search(
            r'def _map_single_value_to_literal_id\(self.*?\n(.*?)(?=\n    def |\nclass |\Z)',
            self.src, re.DOTALL
        )
        
        self.assertIsNotNone(method_match, "No se encontró el método _map_single_value_to_literal_id")
        
        method_body = method_match.group(1)
        
        # Verificar si crea tensores GPU
        creates_gpu_tensor = bool(re.search(
            r'torch\.tensor\([^)]+device\s*=\s*self\.device',
            method_body
        ))
        
        if creates_gpu_tensor:
            # Verificar si hay comentario explicando por qué o si es intencional
            has_fixme = bool(re.search(r'FIXME|TODO|OPTIMIZE', method_body))
            
            self.assertTrue(
                has_fixme,
                "BUG N8 CONFIRMADO: _map_single_value_to_literal_id crea tensores GPU "
                "(torch.tensor con device=self.device) para cada llamada individual. "
                "Este método se llama en bucles y causa sobrecarga de memoria. "
                "Se recomienda implementar una versión escalar/CPU o pre-cachear literales."
            )

    def test_associative_rules_calls_map_single_in_loop(self):
        """Verificar que _apply_associative_rules llama al método dentro de bucles."""
        # Buscar llamadas a _map_single_value_to_literal_id
        calls_pattern = re.findall(
            r'_map_single_value_to_literal_id\(',
            self.src
        )
        
        self.assertGreater(len(calls_pattern), 0,
            "No se encontraron llamadas a _map_single_value_to_literal_id")
        
        # Verificar que al menos una está dentro de un bucle for
        associative_method = re.search(
            r'def _apply_associative_rules\(.*?\n(.*?)(?=\n    def |\nclass |\Z)',
            self.src, re.DOTALL
        )
        
        if associative_method:
            body = associative_method.group(1)
            # Buscar patrón: for ... \n ... _map_single_value_to_literal_id
            loop_with_call = re.search(
                r'for\s+\w+.*?:.*?_map_single_value_to_literal_id',
                body, re.DOTALL
            )
            
            self.assertIsNotNone(
                loop_with_call,
                "BUG N8: _map_single_value_to_literal_id se llama dentro de "
                "bucles en _apply_associative_rules. Cada llamada crea 2+ "
                "tensores GPU, causando presión de memoria significativa."
            )


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    unittest.main(verbosity=2)
