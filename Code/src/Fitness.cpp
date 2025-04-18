#include "Fitness.h"
#include "Globals.h" // Necesario para constantes globales
#include <cmath>
#include <limits>
#include <vector>
#include <numeric>
#include <iostream>
#include <iomanip>

// Calcula el fitness "crudo" usando parámetros globales.
double calculate_raw_fitness(const NodePtr& tree,
                             const std::vector<double>& targets,
                             const std::vector<double>& x_values) {
    if (x_values.size() != targets.size() || x_values.empty()) return INF;

    double error_sum_pow13 = 0.0;
    double sum_sq_error = 0.0;
    bool all_precise = true;
    size_t num_points = x_values.size();

    for (size_t i = 0; i < num_points; ++i) {
        double predicted_val = evaluate_tree(tree, x_values[i]);
        if (std::isnan(predicted_val) || std::isinf(predicted_val)) return INF;

        double diff = std::fabs(predicted_val - targets[i]);
        if (diff >= FITNESS_PRECISION_THRESHOLD) all_precise = false; // <-- Usa constante global

        // Acumular error para ambas métricas
        if (!USE_RMSE_FITNESS) { // Solo calcular pow si se usa esa métrica
             error_sum_pow13 += std::pow(diff, FITNESS_ORIGINAL_POWER); // <-- Usa constante global
        }
        sum_sq_error += diff * diff;

        // Control de desbordamiento (mantener hardcoded o mover a Globals?)
        if (error_sum_pow13 > 1e100 || sum_sq_error > 1e100) return INF;
    }

    // Seleccionar métrica
    double raw_error;
    if (USE_RMSE_FITNESS) {
        if (num_points == 0) return INF;
        raw_error = std::sqrt(sum_sq_error / num_points);
    } else {
        raw_error = error_sum_pow13;
        // Considerar normalización: raw_error /= num_points;
    }

    // Aplicar bonus
    if (all_precise) raw_error *= FITNESS_PRECISION_BONUS; // <-- Usa constante global

    // Comprobar si el error final es inválido
    if (std::isnan(raw_error) || std::isinf(raw_error)) return INF;

    return raw_error;
}

// Calcula el fitness final usando parámetros globales.
double evaluate_fitness(const NodePtr& tree,
                        const std::vector<double>& targets,
                        const std::vector<double>& x_values) {
    double raw_fitness = calculate_raw_fitness(tree, targets, x_values);
    if (raw_fitness >= INF) return INF;

    // Penalización por complejidad
    double complexity = static_cast<double>(tree_size(tree));
    double penalty = complexity * COMPLEXITY_PENALTY_FACTOR; // <-- Usa constante global

    // Aplicar penalización
    double final_fitness = raw_fitness * (1.0 + penalty);

    // Comprobaciones finales
    if (std::isnan(final_fitness) || std::isinf(final_fitness)) return INF;
    if (final_fitness < 0) return 0;

    return final_fitness;
}
