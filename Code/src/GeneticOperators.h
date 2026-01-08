// ============================================================
// Archivo: src/GeneticOperators.h
// ============================================================
#ifndef GENETICOPERATORS_H
#define GENETICOPERATORS_H

#include "ExpressionTree.h"
#include "Globals.h" // Incluir Globals.h para INF
#include <vector>
#include <memory> // Para std::move

// Estructura para representar un individuo en la población.
// Contiene el árbol de expresión y su fitness cacheado.
struct Individual {
    NodePtr tree; // Puntero inteligente al árbol de expresión
    double fitness = INF; // Fitness cacheado (menor es mejor), inicializado a infinito
    std::vector<double> errors; // Cache of per-case errors for Lexicase Selection
    bool fitness_valid = false; // Indica si el fitness cacheado es válido

    // Constructor por defecto
    Individual() = default;
    // Constructor a partir de un árbol (mueve el puntero)
    explicit Individual(NodePtr t) : tree(std::move(t)) {}

    // Operador de comparación para ordenar individuos (menor fitness primero)
    bool operator<(const Individual& other) const {
        // Manejar casos donde uno o ambos fitness no son válidos
        if (!fitness_valid && !other.fitness_valid) return false; // Iguales si ambos inválidos
        if (!fitness_valid) return false; // Inválido es "peor" que válido (va después)
        if (!other.fitness_valid) return true; // Válido es "mejor" que inválido (va antes)
        // Comparar por fitness si ambos son válidos
        return fitness < other.fitness;
    }
};


// --- Funciones de Operadores Genéticos ---

// Genera un árbol de expresión aleatorio hasta una profundidad máxima.
NodePtr generate_random_tree(int max_depth, int current_depth = 0);

// Crea la población inicial de individuos.
std::vector<Individual> create_initial_population(int population_size);

// Selecciona un individuo usando selección por torneo con presión de parsimonia.
Individual tournament_selection(const std::vector<Individual>& population, int tournament_size);

// Selecciona un individuo usando Epsilon-Lexicase Selection (más inteligente)
Individual lexicase_selection(std::vector<Individual>& population, const std::vector<double>& targets, const std::vector<std::vector<double>>& x_values);

// Realiza el cruce (crossover) entre dos individuos y devuelve un nuevo individuo.
Individual crossover(const Individual& parent1, const Individual& parent2);

// Mutata un individuo in-place.
void mutate(Individual& individual, double mutation_rate);

// Simplifica un árbol in-place.
void simplify_tree(NodePtr& tree);

// Tipos de mutación posibles.
enum class MutationType {
    ConstantChange,
    OperatorChange,
    SubtreeReplace,
    NodeInsertion,
    NodeDeletion // <-- AÑADIDO: Tipo para eliminar un nodo
    // Simplification (manejado por DomainConstraints)
};

// Mutata un árbol aplicando uno de los tipos de mutación con cierta probabilidad.
// Devuelve un nuevo árbol (clonado y potencialmente mutado).
NodePtr mutate_tree(const NodePtr& tree, double mutation_rate, int max_depth);

// Realiza el cruce (crossover) entre dos árboles padres, modificándolos in-place.
void crossover_trees(NodePtr& tree1, NodePtr& tree2);


#endif // GENETICOPERATORS_H
