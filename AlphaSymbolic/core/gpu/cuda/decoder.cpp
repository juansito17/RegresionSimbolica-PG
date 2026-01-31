
#include <torch/extension.h>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <iomanip>

// Helper to format float constants
std::string format_const(double val) {
    std::ostringstream oss;
    if (std::abs(val) < 1e-4 || std::abs(val) > 1e4) {
        oss << std::scientific << std::setprecision(4) << val;
    } else {
        oss << std::fixed << std::setprecision(4) << val;
    }
    std::string s = oss.str();
    // Remove trailing zeros? logic could be more complex but this is fine for now
    return s;
}

std::vector<std::string> decode_rpn(
    torch::Tensor population, 
    torch::Tensor constants, 
    const std::vector<std::string>& vocab,
    const std::vector<int>& arities,
    int PAD_ID
) {
    // population: [B, L] (Long)
    // constants: [B, K] (Float/Double)
    // vocab: map id -> string
    // arities: map id -> arity
    
    auto B = population.size(0);
    auto L = population.size(1);
    
    // Ensure CPU
    auto pop_cpu = population.cpu();
    auto const_cpu = constants.cpu(); // We process string on CPU anyway, but C++ loop is faster
    
    auto pop_ptr = pop_cpu.data_ptr<int64_t>();
    
    // Handle float/double constants
    bool is_float = (constants.dtype() == torch::kFloat32);
    auto const_float_ptr = is_float ? const_cpu.data_ptr<float>() : nullptr;
    auto const_double_ptr = !is_float ? const_cpu.data_ptr<double>() : nullptr;
    auto K = constants.size(1);
    
    std::vector<std::string> results;
    results.reserve(B);
    
    for (int64_t i = 0; i < B; ++i) {
        std::vector<std::string> stack;
        stack.reserve(64);
        
        int64_t const_idx = 0;
        bool error = false;
        
        for (int64_t j = 0; j < L; ++j) {
            int64_t token_id = pop_ptr[i * L + j];
            
            if (token_id == PAD_ID) break;
            
            // Safety check
            if (token_id < 0 || token_id >= (int64_t)vocab.size()) {
                stack.push_back("?");
                continue;
            }
            
            std::string token = vocab[(size_t)token_id];
            int arity = (token_id < (int64_t)arities.size()) ? arities[(size_t)token_id] : 0;
            
            // Logic mimicking engine.py
            if (token == "C") {
                double val = 0.0;
                if (const_idx < K) {
                    val = is_float ? 
                        const_float_ptr[i * K + const_idx] : 
                        const_double_ptr[i * K + const_idx];
                    const_idx++;
                }
                stack.push_back(format_const(val));
            } else if (arity > 0) {
                // Operator
                if (arity == 1) {
                    if (stack.empty()) { error = true; break; }
                    std::string a = stack.back(); stack.pop_back();
                    
                    // Specific formatting
                    if (token == "sin" || token == "s") stack.push_back("sin(" + a + ")");
                    else if (token == "cos" || token == "c") stack.push_back("cos(" + a + ")");
                    else if (token == "log" || token == "l") stack.push_back("log(" + a + ")");
                    else if (token == "exp" || token == "e") stack.push_back("exp(" + a + ")");
                    else if (token == "sqrt" || token == "q") stack.push_back("sqrt(" + a + ")");
                    else if (token == "abs" || token == "a") stack.push_back("abs(" + a + ")");
                    else if (token == "floor" || token == "_") stack.push_back("floor(" + a + ")");
                    else if (token == "ceil") stack.push_back("ceil(" + a + ")");
                    
                    // Fact / Gamma aliases
                    else if (token == "!" || token == "fact") stack.push_back("fact(" + a + ")");
                    else if (token == "gamma") stack.push_back("gamma(" + a + ")");
                    else if (token == "lgamma" || token == "g") stack.push_back("lgamma(" + a + ")");
                    
                    // Trig
                    else if (token == "asin" || token == "S") stack.push_back("asin(" + a + ")");
                    else if (token == "acos" || token == "C") stack.push_back("acos(" + a + ")");
                    else if (token == "atan" || token == "T") stack.push_back("atan(" + a + ")");
                    else if (token == "neg") stack.push_back("neg(" + a + ")");
                    else if (token == "sign") stack.push_back("sign(" + a + ")");
                    else stack.push_back(token + "(" + a + ")");
                    
                } else if (arity == 2) {
                    if (stack.size() < 2) { error = true; break; }
                    std::string b = stack.back(); stack.pop_back();
                    std::string a = stack.back(); stack.pop_back();
                    
                    if (token == "+") stack.push_back("(" + a + " + " + b + ")");
                    else if (token == "-") stack.push_back("(" + a + " - " + b + ")"); // TODO: Handle unary minus edge cases if grammar allows
                    else if (token == "*") stack.push_back("(" + a + " * " + b + ")");
                    else if (token == "/") stack.push_back("(" + a + " / " + b + ")");
                    else if (token == "pow" || token == "^") stack.push_back("(" + a + " ^ " + b + ")");
                    else if (token == "%" || token == "mod") stack.push_back("(" + a + " % " + b + ")");
                    else stack.push_back("(" + a + " " + token + " " + b + ")");
                }
            } else {
                // Variable or unknown 0-arity
                if (token == "x") token = "x0"; // Alias
                stack.push_back(token);
            }
        }
        
        if (error || stack.size() != 1) {
            results.push_back("Invalid");
        } else {
            // Take the top of the stack (last item pushed) as the result
            results.push_back(stack.back());
        }
    }
    
    return results;
}
