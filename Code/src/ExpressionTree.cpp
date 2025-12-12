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
#include <thread>    // Para thread_local RNG

// --- Función auxiliar para formatear constantes ---
// --- Función auxiliar para formatear constantes ---
std::string format_constant(double val) {
    // Si es un entero o muy cercano a un entero, formatarlo como tal.
    if (std::fabs(val - std::round(val)) < SIMPLIFY_NEAR_ZERO_TOLERANCE) {
        return std::to_string(static_cast<long long>(std::round(val)));
    } else {
        std::ostringstream oss;
        // Usar notación científica para valores muy grandes o muy pequeños,
        // o notación fija para el resto, con precisión adecuada.
        // Esto evita cadenas muy largas o pérdida de información.
        if (std::fabs(val) >= 1e6 || std::fabs(val) <= 1e-6) { // Umbrales ajustables
            oss << std::scientific << std::setprecision(8) << val;
        } else {
            oss << std::fixed << std::setprecision(8) << val;
        }
        
        std::string s = oss.str();
        // Eliminar ceros finales y el punto decimal si no hay parte fraccionaria
        // Esto puede ser delicado con std::scientific, así que hay que ser cuidadosos.
        // Para std::fixed:
        if (s.find('.') != std::string::npos) {
            s.erase(s.find_last_not_of('0') + 1, std::string::npos);
            if (!s.empty() && s.back() == '.') s.pop_back();
        }
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
            // Determine arity
            bool is_unary = (node->op == 's' || node->op == 'c' || node->op == 'l' || node->op == 'e' || node->op == '!' || node->op == '_' || node->op == 'g');

            double leftVal = evaluate_tree(node->left, x);
            double rightVal = 0.0;
            if (!is_unary) {
                rightVal = evaluate_tree(node->right, x);
            }

            if (std::isnan(leftVal)) return std::nan("");
            if (!is_unary && std::isnan(rightVal)) return std::nan("");

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
                    case '%':
                        if (std::fabs(rightVal) < SIMPLIFY_NEAR_ZERO_TOLERANCE) return INF;
                        result = std::fmod(leftVal, rightVal);
                        break;
                    case 's': result = std::sin(leftVal); break;
                    case 'c': result = std::cos(leftVal); break;
                    case 'l': 
                        if (leftVal <= 1e-9) return INF; // Log domain check
                        result = std::log(leftVal); 
                        break;
                    case 'e': 
                        if (leftVal > 700.0) return INF; // Overflow check
                        result = std::exp(leftVal); 
                        break;
                    case '!': 
                        if (leftVal < 0 && std::floor(leftVal) == leftVal) return INF; // Negative integer check
                        if (leftVal > 170.0) return INF; // Overflow check
                        result = std::tgamma(leftVal + 1.0); 
                        break;
                    case '_': result = std::floor(leftVal); break;
                    case 'g':
                        if (leftVal <= -1.0) return INF; // Check for Gamma domain (approx)
                        result = std::lgamma(leftVal + 1.0); 
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
            std::string left_str = tree_to_string(left_node);
            
            // Check arity
            bool is_unary = (node->op == 's' || node->op == 'c' || node->op == 'l' || node->op == 'e' || node->op == '!' || node->op == '_' || node->op == 'g');

            if (is_unary) {
                switch(node->op) {
                    case 's': return "sin(" + left_str + ")";
                    case 'c': return "cos(" + left_str + ")";
                    case 'l': return "log(" + left_str + ")";
                    case 'e': return "exp(" + left_str + ")";
                    case '!': return "(" + left_str + ")!"; // Postfix for factorial
                    case '_': return "floor(" + left_str + ")";
                    case 'g': return "lgamma(" + left_str + ")";
                    default: return "op(" + left_str + ")";
                }
            }

            NodePtr right_node = node->right;
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
// === OPTIMIZACIÓN: RNG thread-local para evitar contención en OpenMP ===
std::mt19937& get_rng() {
    thread_local std::mt19937 local_rng(
        std::random_device{}() ^ 
        static_cast<unsigned>(std::hash<std::thread::id>{}(std::this_thread::get_id()))
    );
    return local_rng;
}


// ============================================================
// --- Parser de Fórmulas desde String (v4 - Parser Corregido) ---
// ============================================================

// Helper para obtener precedencia de operadores
int get_precedence(char op) {
    switch (op) {
        case '+': case '-': return 1;
        case '*': case '/': case '%': return 2;
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

    // Función interna para procesar operadores según precedencia y asociatividad
    auto process_operators_by_precedence = [&](int current_precedence, char current_op_char = 0) {
        // La asociatividad derecha para '^' significa que se procesa si el operador en la pila
        // tiene MAYOR precedencia, no MAYOR O IGUAL.
        bool is_right_associative = (current_op_char == '^');

        while (!operator_stack.empty() && operator_stack.top() != '(') {
            char top_op = operator_stack.top();
            int top_precedence = get_precedence(top_op);

            if (is_right_associative ? (top_precedence > current_precedence) : (top_precedence >= current_precedence)) {
                operator_stack.pop(); // Sacar operador de la pila
                if (operand_stack.size() < 2) throw std::runtime_error("Operandos insuficientes para operador '" + std::string(1, top_op) + "'.");
                NodePtr right = operand_stack.top(); operand_stack.pop();
                NodePtr left = operand_stack.top(); operand_stack.pop();
                operand_stack.push(apply_binary_operation(right, left, top_op));
            } else {
                break; // Parar si la precedencia es menor o si es asociativo a la derecha y es igual
            }
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
            } catch (const std::invalid_argument& e) {
                throw std::runtime_error("Número inválido (formato): '" + num_str + "' - " + e.what());
            } catch (const std::out_of_range& e) {
                throw std::runtime_error("Número inválido (rango): '" + num_str + "' - " + e.what());
            }
            continue;
        }

        // --- B. Parsear Funciones Unarias (l, g, sin, cos, exp, log, lgamma, floor, gamma) ---
        // Map of function names to their operator characters
        std::unordered_map<std::string, char> func_map = {
            {"sin", 's'}, {"cos", 'c'}, {"log", 'l'}, {"exp", 'e'},
            {"floor", '_'}, {"lgamma", 'g'}, {"gamma", '!'}, 
            {"l", 'l'}, {"g", 'g'}, {"e", 'e'}, {"s", 's'}, {"c", 'c'}
        };
        
        // Try to match function names (check longer names first)
        bool matched_func = false;
        for (const auto& [func_name, func_op] : func_map) {
            if (i + func_name.length() <= formula.length() && 
                formula.substr(i, func_name.length()) == func_name &&
                (i + func_name.length() >= formula.length() || formula[i + func_name.length()] == '(')) {
                
                // Check if this is actually a function call (followed by '(')
                size_t after_name = i + func_name.length();
                if (after_name < formula.length() && formula[after_name] == '(') {
                    if (last_token_was_operand) { // Implicit multiplication
                        process_operators_by_precedence(get_precedence('*'));
                        operator_stack.push('*');
                        last_token_was_operand = false;
                    }
                    
                    // Find the matching closing parenthesis
                    int paren_count = 1;
                    size_t arg_start = after_name + 1;
                    size_t j = arg_start;
                    while (j < formula.length() && paren_count > 0) {
                        if (formula[j] == '(') paren_count++;
                        else if (formula[j] == ')') paren_count--;
                        j++;
                    }
                    if (paren_count != 0) {
                        throw std::runtime_error("Paréntesis sin cerrar en función '" + func_name + "'.");
                    }
                    size_t arg_end = j - 1; // Position of closing ')'
                    
                    // Extract and recursively parse the argument
                    std::string arg_str = formula.substr(arg_start, arg_end - arg_start);
                    NodePtr arg_tree = parse_formula_string(arg_str);
                    
                    // Create unary operator node
                    auto func_node = std::make_shared<Node>(NodeType::Operator);
                    func_node->op = func_op;
                    func_node->left = arg_tree;
                    func_node->right = nullptr;
                    
                    operand_stack.push(func_node);
                    last_token_was_operand = true;
                    i = j; // Skip past the closing ')'
                    matched_func = true;
                    break;
                }
            }
        }
        if (matched_func) continue;

        // --- C. Parsear Variable 'x' ---
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

        // --- D. Parsear Paréntesis de Apertura '(' ---
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

        // --- E. Parsear Paréntesis de Cierre ')' ---
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

        // --- F. Parsear Operadores (+ - * / ^ %) ---
        if (std::string("+-*/^%").find(token) != std::string::npos) {
            // Manejar '-' unario vs binario
            if (token == '-' && !last_token_was_operand) {
                // Es un '-' unario. Insertar un 0 como operando izquierdo implícito.
                // Esto permite tratar el '-' como un operador binario normal.
                auto zero_node = std::make_shared<Node>(NodeType::Constant); zero_node->value = 0.0;
                operand_stack.push(zero_node);
                // No cambiar last_token_was_operand a true, ya que el 0 implícito
                // es solo para el operador unario y no un operando "real" previo.
                // Si hubiera una multiplicación implícita (ej. "2-x"), ya se habría manejado.
            }
            // Ignorar '+' unario (no afecta el valor, no necesita un 0 implícito)
            else if (token == '+' && !last_token_was_operand) {
                // No hacer nada, simplemente avanzar al siguiente token
                i++;
                continue;
            }
            
            // Operador binario normal
            if (!last_token_was_operand && (token == '*' || token == '/' || token == '^' || token == '%')) {
                throw std::runtime_error("Operador binario '" + std::string(1, token) + "' inesperado. Se esperaba operando.");
            }

            // Procesar operadores en la pila con mayor o igual precedencia (o solo mayor para asociativos a derecha)
            process_operators_by_precedence(get_precedence(token), token);
            operator_stack.push(token);
            last_token_was_operand = false; // Después de un operador, se espera un operando
            i++;
            continue;
        }

        // --- G. Token Desconocido ---
        throw std::runtime_error("Token desconocido en la fórmula: '" + std::string(1, token) + "'");

    } // Fin del bucle for

    // --- H. Procesamiento Final después del bucle ---
    while (!operator_stack.empty()) {
        if (operator_stack.top() == '(') throw std::runtime_error("Paréntesis '(' sin cerrar al final.");
        // Procesar todos los operadores restantes en la pila
        process_operators_by_precedence(0); // 0 como precedencia mínima para forzar el procesamiento
    }

    // Verificación final de la pila de operandos
    if (operand_stack.size() != 1) {
         if (operand_stack.empty() && formula.length() > 0) throw std::runtime_error("Error: No se generó ningún resultado del parseo. Fórmula inválida?");
         else if (operand_stack.size() > 1) throw std::runtime_error("Error en la estructura final (operandos restantes: " + std::to_string(operand_stack.size()) + "). Verifique operadores.");
         else throw std::runtime_error("Error desconocido al finalizar el parseo.");
    }

    return operand_stack.top();
}
