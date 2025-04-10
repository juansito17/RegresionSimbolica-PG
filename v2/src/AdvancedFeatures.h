#ifndef ADVANCEDFEATURES_H
#define ADVANCEDFEATURES_H

#include "ExpressionTree.h"
#include "Globals.h" // <--- AÑADIDO: Incluir Globals.h para INF
#include <vector>
#include <string>
#include <map>
#include <set>
#include <utility>
#include <unordered_map> // <--- AÑADIDO: Incluir para std::unordered_map

// Meta-evolution: Parameters that can adapt
struct EvolutionParameters {
    double mutation_rate;
    double elite_percentage;
    int tournament_size;
    double crossover_rate; // Probability of applying crossover

    static EvolutionParameters create_default();
    void mutate(); // Adapt parameters slightly
};

// Reinforcement Learning: Store successful structural patterns
class PatternMemory {
    struct PatternInfo {
        std::string pattern_str;
        double best_fitness = INF; // Usa INF de Globals.h
        int uses = 0;
        double success_rate = 0.0;
    };
    std::unordered_map<std::string, PatternInfo> patterns; // Usa std::unordered_map
    int min_uses_for_suggestion = 3;

public:
    void record_success(const NodePtr& tree, double fitness);
    NodePtr suggest_pattern_based_tree(int max_depth);

private:
    std::string extract_pattern(const NodePtr& tree);
    NodePtr parse_pattern(const std::string& pattern, int max_depth);
};


// Pareto Optimization: Track non-dominated solutions
struct ParetoSolution {
    NodePtr tree = nullptr; // Inicializar por defecto
    double accuracy = INF; // Inicializar por defecto
    double complexity = INF; // Inicializar por defecto
    bool dominated = false;

    // Constructor por defecto (añadido)
    ParetoSolution() = default; // Le dice al compilador que genere uno trivial

    // Tu constructor existente
    ParetoSolution(NodePtr t, double acc, double complexity_val);

    bool dominates(const ParetoSolution& other) const;
};

class ParetoOptimizer {
    std::vector<ParetoSolution> pareto_front;
    size_t max_front_size = 50; // Limit the size of the front

public:
    // Update the Pareto front with individuals from the current population
    void update(const std::vector<struct Individual>& population, // Use Individual struct
                const std::vector<double>& targets,
                const std::vector<double>& x_values);

    // Get the trees from the current Pareto front
    std::vector<NodePtr> get_pareto_solutions();

    const std::vector<ParetoSolution>& get_pareto_front() const { return pareto_front; }
};


// Domain Constraints: Check and fix problematic tree structures
class DomainConstraints {
public:
    // Check if a tree adheres to basic validity rules
    static bool is_valid(const NodePtr& tree);

    // Attempt to simplify/fix a tree that violates constraints
    static NodePtr fix_or_simplify(NodePtr tree); // Takes ownership/clones if needed

private:
     // Recursive simplification helper
    static NodePtr simplify_recursive(NodePtr node);
    // Static check helper
    static bool is_valid_recursive(const NodePtr& node);
};

// Local Improvement/Search (Example: small mutations around a good solution)
std::pair<NodePtr, double> try_local_improvement(const NodePtr& tree,
                                                  double current_fitness,
                                                  const std::vector<double>& targets,
                                                  const std::vector<double>& x_values,
                                                  int attempts = 10);


// Pattern Detection in Targets
std::pair<std::string, double> detect_target_pattern(const std::vector<double>& targets);
NodePtr generate_pattern_based_tree(const std::string& pattern_type, double pattern_value);


#endif // ADVANCEDFEATURES_H