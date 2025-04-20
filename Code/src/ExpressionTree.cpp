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
#include <stack>
#include <unordered_map>
#include <algorithm> // Para std::remove_if
#include <cctype>    // Para isdigit, isspace

// --- Función auxiliar para formatear constantes ---
std::string format_constant(double val) {
    if (std::fabs(val - std::round(val)) < SIMPLIFY_NEAR_ZERO_TOLERANCE) {
        return std::to_string(static_cast<long long>(std::round(val)));
    } else {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(8) << val;
        std::string s = oss.str();
        s.erase(s.find_last_not_of('0') + 1, std::string::npos);
        if (!s.empty() && s.back() == '.') s.pop_back();
        return s.empty() ? "0" : s;
    }
}

// --- evaluate_tree ---
double evaluate_tree(const NodePtr& node, double x) {
    if (!node) return std::nan("");
    switch (node->type) {
        case NodeType::Constant: return node->value;
        case NodeType::Variable: return x;
        case NodeType::Operator: {
            double leftVal = evaluate_tree(node->left, x);
            double rightVal = evaluate_tree(node->right, x);
            if (std::isnan(leftVal) || std::isnan(rightVal)) return std::nan("");
            double result = std::nan("");
            try {
                switch (node->op) {
                    case '+': result = leftVal + rightVal; break;
                    case '-': result = leftVal - rightVal; break;
                    case '*': result = leftVal * rightVal; break;
                    case '/':
                        if (std::fabs(rightVal) < SIMPLIFY_NEAR_ZERO_TOLERANCE) return INF;
                        result = leftVal / rightVal;
                        break;
                    case '^':
                        if (leftVal == 0.0 && rightVal == 0.0) result = 1.0;
                        else if (leftVal == 0.0 && rightVal < 0.0) return INF;
                        else if (leftVal < 0.0 && std::fabs(rightVal - std::round(rightVal)) > SIMPLIFY_NEAR_ZERO_TOLERANCE) return INF;
                        else result = std::pow(leftVal, rightVal);
                        break;
                    default: return std::nan("");
                }
            } catch (const std::exception& e) { return INF; }
            if (std::isinf(result)) return INF;
            if (std::isnan(result)) return std::nan("");
            return result;
        }
        default: return std::nan("");
    }
}

// --- tree_to_string ---
std::string tree_to_string(const NodePtr& node) {
     if (!node) return "NULL";
     switch (node->type) {
        case NodeType::Constant: return format_constant(node->value);
        case NodeType::Variable: return "x";
        case NodeType::Operator: {
            NodePtr left_node = node->left;
            NodePtr right_node = node->right;
            std::string left_str = tree_to_string(left_node);
            std::string right_str = tree_to_string(right_node);
            char current_op = node->op;
            bool right_is_neg_const = (right_node && right_node->type == NodeType::Constant && right_node->value < 0.0);
            if (right_is_neg_const) {
                double abs_right_val = std::fabs(right_node->value);
                std::string abs_right_str = format_constant(abs_right_val);
                if (node->op == '+') { current_op = '-'; right_str = abs_right_str; }
                else if (node->op == '-') { current_op = '+'; right_str = abs_right_str; }
            }
            // Simplificar impresión de (0-A) a (-A)
            if (left_node && left_node->type == NodeType::Constant && left_node->value == 0.0 && current_op == '-') {
                 return "(-" + right_str + ")";
            }
            return "(" + left_str + current_op + right_str + ")";
        }
        default: return "?";
    }
}

// --- tree_size ---
int tree_size(const NodePtr& node) {
    if (!node) return 0;
    if (node->type == NodeType::Constant || node->type == NodeType::Variable) return 1;
    if (node->type == NodeType::Operator) {
        return 1 + tree_size(node->left) + tree_size(node->right);
    }
    return 0;
}

// --- clone_tree ---
NodePtr clone_tree(const NodePtr& node) {
    if (!node) return nullptr;
    auto new_node = std::make_shared<Node>();
    new_node->type = node->type;
    new_node->value = node->value;
    new_node->op = node->op;
    new_node->left = clone_tree(node->left);
    new_node->right = clone_tree(node->right);
    return new_node;
}

// --- collect_node_ptrs ---
void collect_node_ptrs(NodePtr& node, std::vector<NodePtr*>& vec) {
    if (!node) return;
    vec.push_back(&node);
    if (node->type == NodeType::Operator) {
        collect_node_ptrs(node->left, vec);
        collect_node_ptrs(node->right, vec);
    }
}

// --- get_rng ---
namespace { std::mt19937 global_rng(std::random_device{}()); }
std::mt19937& get_rng() {
    return global_rng;
}


// ============================================================
// --- Parser de Fórmulas desde String (v4 - Parser Corregido) ---
// ============================================================

// Helper para obtener precedencia de operadores
int get_precedence(char op) {
    switch (op) {
        case '+': case '-': return 1;
        case '*': case '/': return 2;
        case '^': return 3;
        default: return 0;
    }
}

// Helper para aplicar un operador binario
NodePtr apply_binary_operation(NodePtr right, NodePtr left, char op) {
    if (!left || !right) {
        throw std::runtime_error("Error al aplicar operación binaria '" + std::string(1, op) + "': operandos insuficientes.");
    }
    auto node = std::make_shared<Node>(NodeType::Operator);
    node->op = op;
    node->left = left;
    node->right = right;
    return node;
}

// Función principal para parsear la fórmula
NodePtr parse_formula_string(const std::string& formula_raw) {
    std::string formula = formula_raw;
    formula.erase(std::remove_if(formula.begin(), formula.end(), ::isspace), formula.end());
    if (formula.empty()) throw std::runtime_error("La fórmula está vacía.");

    std::stack<NodePtr> operand_stack;
    std::stack<char> operator_stack;

    // Función interna para procesar operadores según precedencia
    auto process_operators_by_precedence = [&](int current_precedence) {
        while (!operator_stack.empty() && operator_stack.top() != '(' &&
               get_precedence(operator_stack.top()) >= current_precedence) {
            if (operator_stack.top() == '^' && current_precedence == get_precedence('^')) break;
            char op = operator_stack.top(); operator_stack.pop();
            if (operand_stack.size() < 2) throw std::runtime_error("Operandos insuficientes para operador '" + std::string(1, op) + "'.");
            NodePtr right = operand_stack.top(); operand_stack.pop();
            NodePtr left = operand_stack.top(); operand_stack.pop();
            operand_stack.push(apply_binary_operation(right, left, op));
        }
    };

    bool last_token_was_operand = false;

    for (int i = 0; i < formula.length(); /* Incremento manual */ ) {
        char token = formula[i];

        // --- A. Parsear Números ---
        bool starts_number = isdigit(token) || (token == '.' && i + 1 < formula.length() && isdigit(formula[i+1]));
        if (starts_number) {
             if (last_token_was_operand) { // Implicit multiplication
                 process_operators_by_precedence(get_precedence('*'));
                 operator_stack.push('*');
                 last_token_was_operand = false;
             }
            std::string num_str;
            if (token == '.') num_str += '0';
            num_str += token;
            i++;
            while (i < formula.length() && (isdigit(formula[i]) || (formula[i] == '.' && num_str.find('.') == std::string::npos))) {
                num_str += formula[i];
                i++;
            }
            try {
                double value = std::stod(num_str);
                auto node = std::make_shared<Node>(NodeType::Constant); node->value = value;
                operand_stack.push(node);
                last_token_was_operand = true;
            } catch (...) { throw std::runtime_error("Número inválido: '" + num_str + "'"); }
            continue;
        }

        // --- B. Parsear Variable 'x' ---
        if (token == 'x') {
            if (last_token_was_operand) { // Implicit multiplication
                 process_operators_by_precedence(get_precedence('*'));
                 operator_stack.push('*');
                 last_token_was_operand = false;
            }
            auto node = std::make_shared<Node>(NodeType::Variable);
            operand_stack.push(node);
            last_token_was_operand = true;
            i++;
            continue;
        }

        // --- C. Parsear Paréntesis de Apertura '(' ---
        if (token == '(') {
            if (last_token_was_operand) { // Implicit multiplication
                 process_operators_by_precedence(get_precedence('*'));
                 operator_stack.push('*');
                 last_token_was_operand = false;
            }
            operator_stack.push('(');
            last_token_was_operand = false;
            i++;
            continue;
        }

        // --- D. Parsear Paréntesis de Cierre ')' ---
        if (token == ')') {
             if (!last_token_was_operand) {
                  if (!operator_stack.empty() && operator_stack.top() == '(') throw std::runtime_error("Paréntesis vacíos '()' encontrados.");
                  else throw std::runtime_error("Se esperaba un operando antes de ')'.");
             }
            while (!operator_stack.empty() && operator_stack.top() != '(') {
                process_operators_by_precedence(0);
            }
            if (operator_stack.empty()) throw std::runtime_error("Paréntesis ')' sin correspondiente '('.");
            operator_stack.pop(); // Sacar '('
            last_token_was_operand = true;
            i++;
            continue;
        }

        // --- E. Parsear Operadores (+ - * / ^) ---
        if (std::string("+-*/^").find(token) != std::string::npos) {
            // Manejar '-' unario vs binario
            if (token == '-' && !last_token_was_operand) {
                // Es un '-' unario. Insertar un 0 y tratar el '-' como binario.
                 if (last_token_was_operand) { // Check redundante
                     process_operators_by_precedence(get_precedence('*'));
                     operator_stack.push('*');
                 }
                auto zero_node = std::make_shared<Node>(NodeType::Constant); zero_node->value = 0.0;
                operand_stack.push(zero_node);
                last_token_was_operand = true; // 0 es el operando izquierdo para '-'

                // *** CORRECCIÓN: No procesar precedencia aquí ***
                // process_operators_by_precedence(get_precedence('-')); // <-- QUITAR ESTA LÍNEA
                operator_stack.push('-'); // Empujar '-' directamente
                last_token_was_operand = false; // Esperamos el operando derecho
            }
            // Ignorar '+' unario
            else if (token == '+' && !last_token_was_operand) {
                 last_token_was_operand = false; // Simplemente ignorar, esperando operando
            }
            // Operador binario normal
            else {
                 if (!last_token_was_operand) throw std::runtime_error("Operador binario '" + std::string(1, token) + "' inesperado. Se esperaba operando.");
                 // Procesar operadores con precedencia >= antes de empujar el actual
                 process_operators_by_precedence(get_precedence(token));
                 operator_stack.push(token);
                 last_token_was_operand = false;
            }
            i++;
            continue;
        }

        // --- F. Token Desconocido ---
        throw std::runtime_error("Token desconocido en la fórmula: '" + std::string(1, token) + "'");

    } // Fin del bucle for

    // --- G. Procesamiento Final después del bucle ---
    while (!operator_stack.empty()) {
        if (operator_stack.top() == '(') throw std::runtime_error("Paréntesis '(' sin cerrar al final.");
        process_operators_by_precedence(0);
    }

    // Verificación final de la pila de operandos
    if (operand_stack.size() != 1) {
         if (operand_stack.empty() && formula.length() > 0) throw std::runtime_error("Error: No se generó ningún resultado del parseo. Fórmula inválida?");
         else if (operand_stack.size() > 1) throw std::runtime_error("Error en la estructura final (operandos restantes: " + std::to_string(operand_stack.size()) + "). Verifique operadores.");
         else throw std::runtime_error("Error desconocido al finalizar el parseo.");
    }

    return operand_stack.top();
}
