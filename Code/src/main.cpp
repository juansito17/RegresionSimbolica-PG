#include "Globals.h" // Necesario para las constantes globales
#include "GeneticAlgorithm.h"
#include "Fitness.h" // Para evaluate_fitness
#include "ExpressionTree.h" // Para tree_to_string si se necesita aquí
#include <iostream>
#include <vector>
#include <memory> // Para shared_ptr
#include <iomanip> // Para std::setprecision
#include <omp.h>   // Para configuración de OpenMP

#include <fstream> // Para leer archivo
#include <string>
#include <sstream>

int main(int argc, char* argv[]) {
    // === OPTIMIZACIÓN: Configuración explícita de hilos OpenMP ===
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    std::cout << "[OpenMP] Using " << num_threads << " threads" << std::endl;
    
    // Configurar precisión de salida para números flotantes
    // Force immediate flush for each output (important for subprocess capture)
    std::cout << std::unitbuf << std::fixed << std::setprecision(6);
    
    std::vector<std::string> seed_formulas;
    std::string seed_file_path = "";
    std::string data_file_path = "";
    
    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--seed" || arg == "-s") && i + 1 < argc) {
             seed_file_path = argv[i + 1];
             i++; // Skip next arg
        } else if ((arg == "--data" || arg == "-d") && i + 1 < argc) {
             data_file_path = argv[i + 1];
             i++;
        }
    }
    
    if (!seed_file_path.empty()) {
        std::cout << "Loading seeds from: " << seed_file_path << std::endl;
        std::ifstream file(seed_file_path);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty()) {
                    seed_formulas.push_back(line);
                }
            }
            file.close();
            std::cout << "Loaded " << seed_formulas.size() << " formulas." << std::endl;
        } else {
            std::cerr << "[Error] Could not open seed file: " << seed_file_path << std::endl;
        }
    }

    std::vector<double> targets;
    std::vector<double> final_x_values;

    if (!data_file_path.empty()) {
         std::cout << "Loading data from: " << data_file_path << std::endl;
         std::ifstream dfile(data_file_path);
         if (dfile.is_open()) {
             // Format:
             // Line 1: x1 x2 x3 ...
             // Line 2: y1 y2 y3 ...
             // Values separated by space or comma
             
             // Helper lambda to parse line
             auto parse_line = [](const std::string& line) {
                 std::vector<double> vals;
                 std::stringstream ss(line);
                 double val;
                 while (ss >> val) {
                     vals.push_back(val);
                     if (ss.peek() == ',' || ss.peek() == ' ') ss.ignore();
                 }
                 return vals;
             };
             
             std::string line;
             if (std::getline(dfile, line)) final_x_values = parse_line(line);
             if (std::getline(dfile, line)) targets = parse_line(line);
             
             dfile.close();
             
             if (final_x_values.size() != targets.size()) {
                 std::cerr << "[Error] Mismatch in data size: X(" << final_x_values.size() 
                           << ") vs Y(" << targets.size() << ")" << std::endl;
                 return 1;
             }
             std::cout << "Loaded " << final_x_values.size() << " data points." << std::endl;
         } else {
             std::cerr << "[Error] Could not open data file: " << data_file_path << std::endl;
             return 1;
         }
    } else {
        // Fallback to Globals.h
        if (USE_LOG_TRANSFORMATION) {
             std::cout << "Info: Log Transformation is ON (Target = ln(Q(N)))." << std::endl;
             for (size_t i = 0; i < RAW_TARGETS.size(); ++i) {
                 if (RAW_TARGETS[i] > 0) {
                     targets.push_back(std::log(RAW_TARGETS[i]));
                     final_x_values.push_back(X_VALUES[i]);
                 }
             }
        } else {
             std::cout << "Info: Log Transformation is OFF." << std::endl;
             targets = RAW_TARGETS;
             final_x_values = X_VALUES;
        }
    }

    std::cout << "Target Function Points (Effective):" << std::endl;
    // Imprimir los puntos objetivo
    for (size_t i = 0; i < targets.size(); ++i) {
        std::cout << "  f(" << final_x_values[i] << ") = " << targets[i] << std::endl;
    }
    std::cout << "----------------------------------------" << std::endl;
#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
    std::cout << "Info: Running with GPU acceleration." << std::endl;
#else
    std::cout << "Info: Running with CPU acceleration." << std::endl;
#endif
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Parameters:" << std::endl;
    // Imprimir los parámetros globales definidos en Globals.h
    std::cout << "  Total Population: " << TOTAL_POPULATION_SIZE << std::endl;
    std::cout << "  Generations: " << GENERATIONS << std::endl;
    std::cout << "  Islands: " << NUM_ISLANDS << std::endl;
    std::cout << "  Migration Interval: " << MIGRATION_INTERVAL << std::endl;
    std::cout << "  Migration Size: " << MIGRATION_SIZE << std::endl;
    // --- NOMBRES CORREGIDOS ---
    std::cout << "  Mutation Rate (Initial): " << BASE_MUTATION_RATE << std::endl; // <-- Nombre corregido
    std::cout << "  Elite Percentage (Initial): " << BASE_ELITE_PERCENTAGE << std::endl; // <-- Nombre corregido
    // --------------------------
    std::cout << "----------------------------------------" << std::endl;


    try {
        // Crear la instancia del Algoritmo Genético
        // Pasa las referencias a los vectores de datos y los parámetros principales
        GeneticAlgorithm ga(targets, final_x_values, TOTAL_POPULATION_SIZE, GENERATIONS, seed_formulas);

        // Ejecutar el algoritmo
        // La función run() contiene el bucle principal de generaciones y devuelve el mejor árbol encontrado
        NodePtr best_solution_tree = ga.run();

        // La función run() ya imprime el resumen final y la verificación.
        // Comprobar si se encontró alguna solución válida al final
        if (!best_solution_tree) {
            std::cerr << "\nFailed to find any valid solution." << std::endl;
            return 1; // Salir con código de error si no se encontró solución
        }
    } catch (const std::exception& e) {
        std::cerr << "[CRITICAL ERROR] Exception caught in main: " << e.what() << std::endl;
        return 2;
    } catch (...) {
        std::cerr << "[CRITICAL ERROR] Unknown exception caught in main." << std::endl;
        return 3;
    }

    return 0; // Salir con éxito
}
