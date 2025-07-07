#include "Fitness.h"
#include "FitnessGPU.cuh" // Include for GPU fitness evaluation
#include "Globals.h" // Necesario para constantes globales e INF
#include "ExpressionTree.h" // Necesario para tree_to_string
#include <cmath>
#include <limits>
#include <vector>
#include <numeric>
#include <iostream> // Para std::cerr en caso de error futuro
#include <iomanip>  // Para std::fixed/scientific si se necesita en errores

// Calculates the raw fitness using global parameters.
// This function will now dispatch to GPU if GPU_ACCELERATION is enabled.
double calculate_raw_fitness(const NodePtr& tree,
                             const std::vector<double>& targets,
                             const std::vector<double>& x_values) {
#ifdef GPU_ACCELERATION
    return evaluate_fitness_gpu(tree, targets, x_values);
#else
    if (x_values.size() != targets.size() || x_values.empty()) return INF;

    double error_sum_pow13 = 0.0; // Solo si USE_RMSE_FITNESS = false
    double sum_sq_error = 0.0;
    bool all_precise = true;
    size_t num_points = x_values.size();
    bool calculation_failed = false; // Flag para detectar INF/NaN

    for (size_t i = 0; i < num_points; ++i) {
        double predicted_val = evaluate_tree(tree, x_values[i]);

        // Comprobar si la evaluación falló (INF o NaN)
        if (std::isnan(predicted_val) || std::isinf(predicted_val)) {
            calculation_failed = true;
            break; // Salir del bucle si la evaluación falla para un punto
        }

        double target_val = targets[i];
        double diff = predicted_val - target_val;
        double abs_diff = std::fabs(diff);

        if (abs_diff >= FITNESS_PRECISION_THRESHOLD) all_precise = false;

        // Acumular error para ambas métricas (si aplica)
        if (!USE_RMSE_FITNESS) {
             error_sum_pow13 += std::pow(abs_diff, FITNESS_ORIGINAL_POWER);
        }

        // Calcular y acumular error cuadrático
        double sq_diff = diff * diff;
        sum_sq_error += sq_diff;

        // Control de desbordamiento/Infinito en la suma
        if (std::isinf(sum_sq_error) || (error_sum_pow13 >= INF / 10.0 && !USE_RMSE_FITNESS)) {
            calculation_failed = true;
            break;
        }
    } // Fin bucle for puntos

    // Si la evaluación o suma falló en algún punto, devolver INF
    if (calculation_failed) {
        return INF;
    }

    // Seleccionar métrica de error crudo
    double raw_error;
    if (USE_RMSE_FITNESS) {
        if (num_points == 0) return INF;
        double mse = sum_sq_error / num_points;
        if (std::isinf(mse) || std::isnan(mse) || mse < 0) {
             raw_error = INF;
        } else {
             raw_error = std::sqrt(mse); // Calcular RMSE
        }
    } else {
        raw_error = error_sum_pow13;
    }

    // Comprobar si el error crudo es inválido
    if (std::isnan(raw_error) || std::isinf(raw_error) || raw_error < 0) {
         return INF;
    }

    // Aplicar bonus de precisión si todos los puntos estaban dentro del umbral
    if (all_precise) {
         raw_error *= FITNESS_PRECISION_BONUS;
    }

    return raw_error; // Devolver el error crudo (sin penalización por complejidad aún)
#endif // GPU_ACCELERATION
}

// Calcula el fitness final usando parámetros globales.
double evaluate_fitness(const NodePtr& tree,
                        const std::vector<double>& targets,
                        const std::vector<double>& x_values) {

    // The logic to select between CPU and GPU should be handled in calculate_raw_fitness
    // or by a global flag that determines which version of evaluate_fitness is called.
    // For now, if GPU_ACCELERATION is defined, calculate_raw_fitness will use the GPU path.
    double raw_fitness = calculate_raw_fitness(tree, targets, x_values);

    if (raw_fitness >= INF / 10.0) {
         return INF; // Si el error crudo es infinito, el fitness final es infinito
    }

    // Penalización por complejidad
    double complexity = static_cast<double>(tree_size(tree));
    double penalty = complexity * COMPLEXITY_PENALTY_FACTOR; // Usa constante global

    // Aplicar penalización multiplicativa
    double final_fitness = raw_fitness * (1.0 + penalty);

    // Comprobaciones finales
    if (std::isnan(final_fitness) || std::isinf(final_fitness) || final_fitness < 0) {
         return INF;
    }

    return final_fitness;
}
