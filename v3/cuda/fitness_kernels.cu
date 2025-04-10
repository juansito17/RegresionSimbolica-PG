#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include "../include/cuda_defs.h"

// Constants
const int VAR_CODE = -1;
const int CONST_CODE_OFFSET = 1000;
const int OP_ADD = 0;
const int OP_SUB = 1;
const int OP_MUL = 2;
const int OP_DIV = 3;
const int OP_POW = 4;
const int EVAL_STACK_SIZE = 64;

#define MAX_ALLOWED_VALUE 1e15
#define MIN_ALLOWED_VALUE -1e15

__device__ bool is_valid_number(double value) {
    return !isnan(value) && !isinf(value) 
           && value > MIN_ALLOWED_VALUE 
           && value < MAX_ALLOWED_VALUE;
}

__device__ double evaluate_tree_device(const int* structure,
                                     const double* constants,
                                     double x,
                                     int max_flat_size,
                                     int max_constants_per_tree)
{
    double stack[EVAL_STACK_SIZE];
    int stack_ptr = 0;

    for (int i = 0; stack_ptr >= 0 && i < max_flat_size; ++i) {
        int code = structure[i];

        if (code == VAR_CODE) {
            if (stack_ptr >= EVAL_STACK_SIZE) return LARGE_FINITE_PENALTY;
            stack[stack_ptr++] = x;
        }
        else if (code >= CONST_CODE_OFFSET) {
            int const_idx = code - CONST_CODE_OFFSET;
            if (stack_ptr >= EVAL_STACK_SIZE || const_idx < 0 || const_idx >= max_constants_per_tree) {
                return LARGE_FINITE_PENALTY;
            }
            stack[stack_ptr++] = constants[const_idx];
        }
        else if (code >= OP_ADD && code <= OP_POW) {
            if (stack_ptr < 2) return LARGE_FINITE_PENALTY;

            double right = stack[--stack_ptr];
            double left = stack[--stack_ptr];
            
            // Validate operands
            if (!is_valid_number(left) || !is_valid_number(right)) {
                return LARGE_FINITE_PENALTY;
            }

            double result;

            switch (code) {
                case OP_ADD: result = left + right; break;
                case OP_SUB: result = left - right; break;
                case OP_MUL:
                    // Check potential overflow
                    if (fabs(left) > sqrt(MAX_ALLOWED_VALUE) || 
                        fabs(right) > sqrt(MAX_ALLOWED_VALUE)) {
                        return LARGE_FINITE_PENALTY;
                    }
                    result = left * right;
                    break;
                case OP_DIV:
                    if (fabs(right) < 1e-9) return LARGE_FINITE_PENALTY;
                    // Check potential overflow
                    if (fabs(left) > MAX_ALLOWED_VALUE * fabs(right)) {
                        return LARGE_FINITE_PENALTY;
                    }
                    result = left / right;
                    break;
                case OP_POW:
                    // Strict power operation checks
                    if (left == 0.0 && right == 0.0) {
                        result = 1.0;
                    }
                    else if (left < 0.0 && floor(right) != right) {
                        return LARGE_FINITE_PENALTY;
                    }
                    else if (right > 10.0 || right < -10.0) { // Limit exponents
                        return LARGE_FINITE_PENALTY;
                    }
                    else {
                        result = pow(left, right);
                    }
                    break;
                default: 
                    return LARGE_FINITE_PENALTY;
            }

            // Validate result after operation
            if (!is_valid_number(result)) {
                return LARGE_FINITE_PENALTY;
            }

            if (stack_ptr >= EVAL_STACK_SIZE) return LARGE_FINITE_PENALTY;
            stack[stack_ptr++] = result;
        }
    }

    if (stack_ptr == 1) {
        double final_val = stack[0];
        if (!is_valid_number(final_val)) {
            return LARGE_FINITE_PENALTY;
        }
        return final_val;
    }
    return LARGE_FINITE_PENALTY;
}

__global__ void evaluate_fitness_kernel(const double* d_constants,
                                      const int* d_structures,
                                      const double* d_x_values,
                                      const double* d_y_values,
                                      double* d_fitness_results,
                                      int pop_size,
                                      int data_size,
                                      int max_flat_size,
                                      int max_constants_per_tree)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pop_size) return;

    if (d_fitness_results[idx] >= LARGE_FINITE_PENALTY) {
        d_fitness_results[idx] = LARGE_FINITE_PENALTY;
        return;
    }

    size_t struct_offset = (size_t)idx * max_flat_size;
    size_t const_offset = (size_t)idx * max_constants_per_tree;

    double error_sum = 0.0;
    bool all_precise = true;

    for (int i = 0; i < data_size; ++i) {
        double x = d_x_values[i];
        double target = d_y_values[i];

        double val = evaluate_tree_device(d_structures + struct_offset,
                                        d_constants + const_offset,
                                        x, max_flat_size, max_constants_per_tree);

        // Solo retornar LARGE_FINITE_PENALTY si el valor es realmente inválido
        if (!is_valid_number(val)) {
            d_fitness_results[idx] = LARGE_FINITE_PENALTY;
            return;
        }

        double diff = fabs(val - target);
        
        if (diff >= 0.001) {
            all_precise = false;
        }

        // Usar la misma fórmula que v2
        error_sum += pow(diff, 1.3);

        // Ajustar el límite de error para coincidir con v2
        if (error_sum > 1e100) {
            d_fitness_results[idx] = LARGE_FINITE_PENALTY;
            return;
        }
    }

    if (all_precise) {
        error_sum *= 0.0001; // Mismo bonus que v2
    }

    d_fitness_results[idx] = error_sum;
}

extern "C" void launch_fitness_evaluation(const double* d_constants,
                                        const int* d_structures,
                                        const double* d_x_values,
                                        const double* d_y_values,
                                        double* d_fitness_results,
                                        int pop_size,
                                        int data_size,
                                        int max_flat_size,
                                        int max_constants_per_tree)
{
    dim3 grid = calculateGrid(pop_size);
    dim3 block = calculateBlock();

    evaluate_fitness_kernel<<<grid, block>>>(
        d_constants, d_structures, d_x_values, d_y_values,
        d_fitness_results, pop_size, data_size,
        max_flat_size, max_constants_per_tree
    );
}
