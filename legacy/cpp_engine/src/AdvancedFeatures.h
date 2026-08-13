#ifndef ADVANCEDFEATURES_H
#define ADVANCEDFEATURES_H

#include "ExpressionTree.h"
#include "Globals.h" // Incluir Globals.h para INF
#include <vector>
#include <string>
#include <map>
#include <set>
#include <utility> // Para std::pair
#include <unordered_map>

// Meta-evolución: Parámetros que pueden adaptarse durante la ejecución.
struct EvolutionParameters {
    double mutation_rate;    // Tasa de mutación actual
    double elite_percentage; // Porcentaje de élite actual
    int tournament_size;     // Tamaño del torneo actual
    double crossover_rate;   // Tasa de cruce actual

    // Crea un conjunto de parámetros con valores por defecto (iniciales).
    static EvolutionParameters create_default();

    // Adapta (muta) los parámetros ligeramente.
    // AHORA RECIBE el contador de estancamiento para ajustar la intensidad.
    void mutate(int stagnation_counter);
};

// Memoria de patrones: Almacena sub-estructuras exitosas (Reinforcement Learning).
class PatternMemory {
    struct PatternInfo {
        std::string pattern_str; // Representación del patrón
        double best_fitness = INF; // Mejor fitness visto para este patrón
        int uses = 0;             // Número de veces usado/visto
        double success_rate = 0.0; // Tasa de éxito estimada
    };
    std::unordered_map<std::string, PatternInfo> patterns; // Mapa para almacenar patrones
    int min_uses_for_suggestion = 3; // Mínimo de usos para considerar sugerir un patrón

public:
    // Registra el éxito de un árbol (y su patrón) basado en su fitness.
    void record_success(const NodePtr& tree, double fitness);
    // Sugiere un árbol basado en los patrones exitosos almacenados.
    NodePtr suggest_pattern_based_tree(int max_depth);

private:
    // Extrae la representación estructural (string) de un árbol.
    std::string extract_pattern(const NodePtr& tree);
    // Intenta construir un árbol a partir de un patrón (string) - función simplificada.
    NodePtr parse_pattern(const std::string& pattern, int max_depth);
};


// Optimización Pareto: Mantiene un frente de soluciones no dominadas (compromiso precisión/complejidad).
struct ParetoSolution {
    NodePtr tree = nullptr;   // Árbol de la solución
    double accuracy = INF;    // Objetivo 1: Precisión (fitness)
    double complexity = INF;  // Objetivo 2: Complejidad (tamaño)
    bool dominated = false;   // Bandera: ¿está dominada por otra solución?

    // Constructor por defecto (necesario si se usa en contenedores)
    ParetoSolution() = default;
    // Constructor principal
    ParetoSolution(NodePtr t, double acc, double complexity_val);

    // Comprueba si esta solución domina a otra.
    bool dominates(const ParetoSolution& other) const;
};

class ParetoOptimizer {
    std::vector<ParetoSolution> pareto_front; // Almacena las soluciones del frente
    size_t max_front_size = 50; // Límite opcional para el tamaño del frente

public:
    // Actualiza el frente de Pareto con individuos de la población actual.
    void update(const std::vector<struct Individual>& population, // Usa Individual struct
                const std::vector<double>& targets,
                const std::vector<std::vector<double>>& x_values);

    // Obtiene los árboles (NodePtr) de las soluciones en el frente actual.
    std::vector<NodePtr> get_pareto_solutions();

    // Obtiene una referencia constante al frente de Pareto completo.
    const std::vector<ParetoSolution>& get_pareto_front() const { return pareto_front; }
};


// Restricciones de Dominio: Verifica y corrige/simplifica árboles problemáticos.
class DomainConstraints {
public:
    // Comprueba si un árbol cumple reglas básicas de validez estática.
    static bool is_valid(const NodePtr& tree);

    // Intenta simplificar/corregir un árbol (devuelve una copia modificada).
    static NodePtr fix_or_simplify(NodePtr tree);

private:
     // Ayudante recursivo para la simplificación.
    static NodePtr simplify_recursive(NodePtr node);
    // Ayudante recursivo para la validación estática.
    static bool is_valid_recursive(const NodePtr& node);
};

// Búsqueda Local: Intenta mejorar una solución dada explorando vecinos cercanos.
#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
std::pair<NodePtr, double> try_local_improvement(const NodePtr& tree,
                                                  double current_fitness,
                                                  const std::vector<double>& targets,
                                                  const std::vector<std::vector<double>>& x_values,
                                                  int attempts,
                                                  double* d_targets, double* d_x_values);
#else
std::pair<NodePtr, double> try_local_improvement(const NodePtr& tree,
                                                  double current_fitness,
                                                  const std::vector<double>& targets,
                                                  const std::vector<std::vector<double>>& x_values,
                                                  int attempts = 10);
#endif


// Detección de Patrones en los Datos Objetivo.
std::pair<std::string, double> detect_target_pattern(const std::vector<double>& targets);
NodePtr generate_pattern_based_tree(const std::string& pattern_type, double pattern_value);

// Epsilon Lexicase Selection
std::vector<int> epsilon_lexicase_selection(
    int num_parents_needed, 
    int current_pop_size,
    const std::vector<double>& error_matrix, // [PopSize * NumPoints]
    int num_points,
    int num_vars
);

#endif // ADVANCEDFEATURES_H
