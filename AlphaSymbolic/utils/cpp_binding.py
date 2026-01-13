"""
C++ Binding for AlphaSymbolic.
Uses pybind11 to connect Python with C++ evaluator for maximum speed.

To build:
1. Install pybind11: pip install pybind11
2. Compile: python cpp_binding.py build_ext --inplace
"""

# This file creates the Python bindings. The actual C++ code is separate.

CPP_SOURCE = '''
// cpp_evaluator.cpp
// Fast C++ expression evaluator for AlphaSymbolic

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cmath>
#include <vector>
#include <string>
#include <stack>
#include <unordered_map>

namespace py = pybind11;

// Operator lookup
std::unordered_map<std::string, int> OP_ARITY = {
    {"+", 2}, {"-", 2}, {"*", 2}, {"/", 2}, {"pow", 2}, {"mod", 2},
    {"sin", 1}, {"cos", 1}, {"tan", 1}, {"exp", 1}, {"log", 1},
    {"sqrt", 1}, {"abs", 1}, {"floor", 1}, {"ceil", 1}, {"neg", 1}
};

class ExpressionEvaluator {
public:
    py::array_t<double> evaluate(
        const std::vector<std::string>& tokens,
        py::array_t<double> x_values,
        const std::unordered_map<std::string, double>& constants = {}
    ) {
        auto x = x_values.unchecked<1>();
        size_t n = x.shape(0);
        
        py::array_t<double> result(n);
        auto r = result.mutable_unchecked<1>();
        
        for (size_t i = 0; i < n; i++) {
            r(i) = eval_at_point(tokens, x(i), constants);
        }
        
        return result;
    }
    
private:
    double eval_at_point(
        const std::vector<std::string>& tokens,
        double x_val,
        const std::unordered_map<std::string, double>& constants
    ) {
        std::stack<double> stack;
        
        // Process tokens in reverse (for prefix notation)
        for (int i = tokens.size() - 1; i >= 0; i--) {
            const std::string& token = tokens[i];
            
            auto it = OP_ARITY.find(token);
            
            if (it != OP_ARITY.end()) {
                // Operator
                int arity = it->second;
                
                if (arity == 1 && !stack.empty()) {
                    double a = stack.top(); stack.pop();
                    stack.push(apply_unary(token, a));
                } else if (arity == 2 && stack.size() >= 2) {
                    double a = stack.top(); stack.pop();
                    double b = stack.top(); stack.pop();
                    stack.push(apply_binary(token, a, b));
                }
            } else if (token == "x") {
                stack.push(x_val);
            } else if (token == "pi") {
                stack.push(M_PI);
            } else if (token == "e") {
                stack.push(M_E);
            } else if (token == "C") {
                auto cit = constants.find("C");
                stack.push(cit != constants.end() ? cit->second : 1.0);
            } else {
                // Try to parse as number
                try {
                    stack.push(std::stod(token));
                } catch (...) {
                    stack.push(0.0);
                }
            }
        }
        
        return stack.empty() ? 0.0 : stack.top();
    }
    
    double apply_unary(const std::string& op, double a) {
        if (op == "sin") return std::sin(a);
        if (op == "cos") return std::cos(a);
        if (op == "tan") return std::tan(a);
        if (op == "exp") return std::exp(std::min(100.0, std::max(-100.0, a)));
        if (op == "log") return std::log(std::abs(a) + 1e-10);
        if (op == "sqrt") return std::sqrt(std::abs(a));
        if (op == "abs") return std::abs(a);
        if (op == "floor") return std::floor(a);
        if (op == "ceil") return std::ceil(a);
        if (op == "neg") return -a;
        return a;
    }
    
    double apply_binary(const std::string& op, double a, double b) {
        if (op == "+") return a + b;
        if (op == "-") return a - b;
        if (op == "*") return a * b;
        if (op == "/") return b != 0 ? a / b : 0.0;
        if (op == "pow") return std::pow(std::abs(a) + 1e-10, std::min(10.0, std::max(-10.0, b)));
        if (op == "mod") return std::fmod(a, b + 1e-10);
        return 0.0;
    }
};

PYBIND11_MODULE(cpp_evaluator, m) {
    m.doc() = "Fast C++ expression evaluator for AlphaSymbolic";
    
    py::class_<ExpressionEvaluator>(m, "ExpressionEvaluator")
        .def(py::init<>())
        .def("evaluate", &ExpressionEvaluator::evaluate,
             py::arg("tokens"),
             py::arg("x_values"),
             py::arg("constants") = std::unordered_map<std::string, double>());
}
'''

def write_cpp_source():
    """Write the C++ source file."""
    with open("cpp_evaluator.cpp", "w") as f:
        f.write(CPP_SOURCE)
    print("Written cpp_evaluator.cpp")

def get_setup_script():
    """Return the setup.py content for building the extension."""
    return '''
from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        'cpp_evaluator',
        sources=['cpp_evaluator.cpp'],
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['-std=c++14', '-O3', '-fPIC'],
    ),
]

setup(
    name='cpp_evaluator',
    ext_modules=ext_modules,
)
'''

def build_extension():
    """Build the C++ extension."""
    import subprocess
    import sys
    
    # Write source
    write_cpp_source()
    
    # Write setup.py
    with open("setup_cpp.py", "w") as f:
        f.write(get_setup_script())
    
    # Build
    result = subprocess.run([sys.executable, "setup_cpp.py", "build_ext", "--inplace"])
    return result.returncode == 0


# Fallback Python evaluator if C++ not available
class PythonEvaluator:
    """Pure Python fallback evaluator."""
    
    def evaluate(self, tokens, x_values, constants=None):
        from grammar import ExpressionTree
        tree = ExpressionTree(tokens)
        return tree.evaluate(x_values, constants)


def get_evaluator():
    """Get the fastest available evaluator."""
    try:
        from cpp_evaluator import ExpressionEvaluator
        print("Using C++ evaluator (fast)")
        return ExpressionEvaluator()
    except ImportError:
        print("C++ evaluator not built, using Python fallback")
        return PythonEvaluator()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="C++ Binding Manager")
    parser.add_argument("--build", action="store_true", help="Build the C++ extension")
    parser.add_argument("--test", action="store_true", help="Test the evaluator")
    args = parser.parse_args()
    
    if args.build:
        print("Building C++ extension...")
        if build_extension():
            print("Build successful!")
        else:
            print("Build failed. Make sure pybind11 is installed: pip install pybind11")
    
    if args.test:
        import numpy as np
        
        evaluator = get_evaluator()
        
        x = np.linspace(-5, 5, 100)
        tokens = ['+', '*', '2', 'x', '3']  # 2*x + 3
        
        result = evaluator.evaluate(tokens, x)
        expected = 2 * x + 3
        
        error = np.max(np.abs(result - expected))
        print(f"Max error: {error}")
        print("Test passed!" if error < 1e-10 else "Test failed!")
