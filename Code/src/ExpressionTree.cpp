#include "ExpressionTree.h"
#include "Globals.h"
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>

// --- Función auxiliar para formatear constantes ---
std::string format_constant(double val) {
    // Comprobar si es prácticamente un entero
    // Usar una tolerancia pequeña para la comparación de flotantes
    if (std::fabs(val - std::round(val)) < SIMPLIFY_NEAR_ZERO_TOLERANCE) { // Usar tolerancia global
        return std::to_string(static_cast<long long>(std::round(val)));
    } else {
        // Usar el default de std::to_string para flotantes
        return std::to_string(val);
    }
}


// --- evaluate_tree ---
// Evalúa el árbol de expresión para un valor dado de x.
double evaluate_tree(const NodePtr& node, double x) {
    if (!node) return std::nan(""); // Devolver NaN si el nodo es nulo
    switch (node->type) {
        case NodeType::Constant: return node->value;
        case NodeType::Variable: return x;
        case NodeType::Operator: {
            double leftVal = evaluate_tree(node->left, x);
            double rightVal = evaluate_tree(node->right, x);
            // Propagar NaN o INF si vienen de los hijos
            if (std::isnan(leftVal) || std::isnan(rightVal)) return std::nan("");
            if (std::isinf(leftVal) || std::isinf(rightVal)) return INF;

            double result = std::nan(""); // Inicializar
            switch (node->op) {
                case '+': result = leftVal + rightVal; break;
                case '-': result = leftVal - rightVal; break;
                case '*': result = leftVal * rightVal; break;
                case '/':
                    if (std::fabs(rightVal) < SIMPLIFY_NEAR_ZERO_TOLERANCE) return INF; // Div por cero
                    result = leftVal / rightVal;
                    break;
                case '^':
                    // Manejo de casos especiales de pow
                    if (leftVal == 0.0 && rightVal == 0.0) result = 1.0; // 0^0 = 1
                    else if (leftVal < 0.0 && std::floor(rightVal) != rightVal) return INF; // base neg, exp no entero -> INF
                    else if (leftVal == 0.0 && rightVal <= 0.0) return INF; // 0^0 o 0^-exp -> INF
                    else result = std::pow(leftVal, rightVal); // Calcular potencia
                    break;
                default: return std::nan(""); // Operador desconocido
            }
            // Chequear resultado de la operación para Inf/NaN (ej. por overflow)
            if (std::isinf(result)) return INF;
            if (std::isnan(result)) return std::nan("");

            return result; // Devolver resultado válido
        }
        default: return std::nan(""); // Tipo desconocido
    }
}

// --- tree_to_string (CON FORMATO MEJORADO) ---
// Convierte el árbol a una representación de string legible.
std::string tree_to_string(const NodePtr& node) {
     if (!node) return ""; // String vacío para nodo nulo

     switch (node->type) {
        case NodeType::Constant:
            return format_constant(node->value); // Usar helper
        case NodeType::Variable:
            return "x"; // Variable es siempre "x"
        case NodeType::Operator:
            {
                NodePtr left_node = node->left;
                NodePtr right_node = node->right;
                std::string left_str = tree_to_string(left_node);
                std::string right_str;
                char current_op = node->op;
                bool right_is_neg_const = (right_node && right_node->type == NodeType::Constant && right_node->value < 0.0);

                // Lógica para ajustar +/- y -/-
                if (right_is_neg_const) {
                    double abs_right_val = std::fabs(right_node->value);
                    std::string abs_right_str = format_constant(abs_right_val);
                    if (node->op == '+') { // A + (-B) -> A - B
                        current_op = '-';
                        right_str = abs_right_str;
                    } else if (node->op == '-') { // A - (-B) -> A + B
                        current_op = '+';
                        right_str = abs_right_str;
                    } else { // *, /, ^ con negativo: mantener op, string ya tiene '-'
                        right_str = tree_to_string(right_node);
                    }
                } else { // Caso normal
                    right_str = tree_to_string(right_node);
                }
                // Siempre añadir paréntesis para claridad (se podría mejorar con precedencia)
                return "(" + left_str + current_op + right_str + ")";
            }
        default:
            return "?"; // Tipo desconocido
    }
}

// --- tree_size (CUERPO COMPLETO RESTAURADO) ---
// Calcula el número de nodos en el árbol (tamaño/complejidad).
int tree_size(const NodePtr& node) {
    if (!node) return 0; // Árbol vacío tiene tamaño 0
    // Nodos terminales (hojas) tienen tamaño 1
    if (node->type == NodeType::Constant || node->type == NodeType::Variable) return 1;
    // Nodo operador: 1 (él mismo) + tamaño de hijos izquierdo y derecho
    if (node->type == NodeType::Operator) {
        return 1 + tree_size(node->left) + tree_size(node->right);
    }
    return 0; // Caso inesperado (tipo desconocido)
}

// --- clone_tree (CUERPO COMPLETO RESTAURADO) ---
// Crea una copia profunda (clon) de un árbol.
NodePtr clone_tree(const NodePtr& node) {
    if (!node) return nullptr; // Clon de nulo es nulo

    // Crear nuevo nodo y copiar datos básicos
    auto new_node = std::make_shared<Node>();
    new_node->type = node->type;
    new_node->value = node->value; // Copia valor (para constantes)
    new_node->op = node->op;       // Copia operador

    // Clonar recursivamente los hijos si existen
    if (node->left) {
        new_node->left = clone_tree(node->left);
    }
    if (node->right) {
        new_node->right = clone_tree(node->right);
    }

    return new_node; // Devolver el puntero al nuevo nodo raíz clonado
}

// --- collect_node_ptrs (CUERPO COMPLETO RESTAURADO) ---
// Recolecta punteros a los NodePtr dentro de un árbol (para mutación/cruce).
void collect_node_ptrs(NodePtr& node, std::vector<NodePtr*>& vec) {
    if (!node) return; // Parar si el nodo es nulo

    vec.push_back(&node); // Añadir puntero al shared_ptr actual

    // Recurrir solo si es un operador (los terminales no tienen hijos a modificar)
    if (node->type == NodeType::Operator) {
        if (node->left) {
             collect_node_ptrs(node->left, vec); // Pasar el shared_ptr del hijo izquierdo
        }
         if (node->right) {
             collect_node_ptrs(node->right, vec); // Pasar el shared_ptr del hijo derecho
        }
    }
}

// --- get_rng (CUERPO COMPLETO RESTAURADO) ---
// Definición y acceso al generador de números aleatorios global.
namespace { std::mt19937 global_rng(std::random_device{}()); }
std::mt19937& get_rng() {
    return global_rng;
}
