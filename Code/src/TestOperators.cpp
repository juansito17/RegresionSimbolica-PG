
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <cassert>
#include "ExpressionTree.h"
#include "AdvancedFeatures.h" // For fix_or_simplify
#include "Fitness.h"
#include "Globals.h"

// --- Helper Macros for Testing ---
#define ASSERT_NEAR(val1, val2, tol) \
    if (std::fabs((val1) - (val2)) > (tol)) { \
        std::cerr << "[FAIL] Line " << __LINE__ << ": Expected " << (val2) << ", got " << (val1) << " (diff: " << std::fabs((val1)-(val2)) << ")" << std::endl; \
        return false; \
    }

#define ASSERT_STR_EQ(str1, str2) \
    if ((str1) != (str2)) { \
        std::cerr << "[FAIL] Line " << __LINE__ << ": Expected '" << (str2) << "', got '" << (str1) << "'" << std::endl; \
        return false; \
    }

// --- Test Functions ---

bool test_binary_operators() {
    std::cout << "Testing Binary Operators..." << std::endl;
    
    // Test +
    NodePtr root = parse_formula_string("2+3");
    double val = evaluate_tree(root, 0.0);
    ASSERT_NEAR(val, 5.0, 1e-9);

    // Test -
    root = parse_formula_string("10-4");
    val = evaluate_tree(root, 0.0);
    ASSERT_NEAR(val, 6.0, 1e-9);

    // Test *
    root = parse_formula_string("3*4");
    val = evaluate_tree(root, 0.0);
    ASSERT_NEAR(val, 12.0, 1e-9);

    // Test /
    root = parse_formula_string("10/2");
    val = evaluate_tree(root, 0.0);
    ASSERT_NEAR(val, 5.0, 1e-9);

    // Test ^ (Power)
    root = parse_formula_string("2^3");
    val = evaluate_tree(root, 0.0);
    ASSERT_NEAR(val, 8.0, 1e-9);

    // Test % (Mod)
    root = parse_formula_string("10%3");
    val = evaluate_tree(root, 0.0);
    ASSERT_NEAR(val, 1.0, 1e-9);

    std::cout << "  -> Binary Operators Passed" << std::endl;
    return true;
}

bool test_unary_operators() {
    std::cout << "Testing Unary Operators..." << std::endl;

    // Test sin
    NodePtr root = parse_formula_string("sin(0)");
    double val = evaluate_tree(root, 0.0);
    ASSERT_NEAR(val, 0.0, 1e-9);

    // Test cos
    root = parse_formula_string("cos(0)");
    val = evaluate_tree(root, 0.0);
    ASSERT_NEAR(val, 1.0, 1e-9);

    // Test log (protected: log(|x|))
    root = parse_formula_string("log(2.7182818)");
    val = evaluate_tree(root, 0.0);
    ASSERT_NEAR(val, 1.0, 1e-5);

    // Test exp
    root = parse_formula_string("exp(1)");
    val = evaluate_tree(root, 0.0);
    ASSERT_NEAR(val, 2.7182818, 1e-5);

    // Test floor
    root = parse_formula_string("floor(2.9)");
    val = evaluate_tree(root, 0.0);
    ASSERT_NEAR(val, 2.0, 1e-9);

    // Test lgamma (g) - Note: Implementation is lgamma(|x|+1) => ln(|x|!)
    // lgamma(3) -> lgamma(4) = ln(3!) = ln(6) = 1.791759
    root = parse_formula_string("lgamma(3)"); 
    val = evaluate_tree(root, 0.0);
    ASSERT_NEAR(val, 1.791759, 1e-4);
    
    // Test g(x) alias
    root = parse_formula_string("g(3)");
    val = evaluate_tree(root, 0.0);
    ASSERT_NEAR(val, 1.791759, 1e-4);

    std::cout << "  -> Unary Operators Passed" << std::endl;
    return true;
}

bool test_simplification() {
    std::cout << "Testing Simplification..." << std::endl;

    // Test Identity: x - x -> 0 (via fix_or_simplify helper)
    // Note: simplification rules are in DomainConstraints::simplify_recursive called by fix_or_simplify
    NodePtr root = parse_formula_string("x-x");
    NodePtr simplified = DomainConstraints::fix_or_simplify(root);
    // Should be constant 0
    double val = evaluate_tree(simplified, 5.0); // x=5
    ASSERT_NEAR(val, 0.0, 1e-9);
    
    // Test Constant Folding: 2+3 -> 5
    root = parse_formula_string("2+3");
    simplified = DomainConstraints::fix_or_simplify(root);
    val = evaluate_tree(simplified, 0.0);
    ASSERT_NEAR(val, 5.0, 1e-9);
    // Tree should be size 1 (Constant)
    if (tree_size(simplified) != 1) {
         std::cerr << "[FAIL] Constant folding failed, tree size is " << tree_size(simplified) << std::endl;
         return false;
    }

    // Test Unary Operator Preservation (The BUG FIX check)
    // lgamma(x) should NOT simplify to just x
    root = parse_formula_string("lgamma(x)");
    simplified = DomainConstraints::fix_or_simplify(root);
    std::string str = tree_to_string(simplified);
    // Expect "lgamma(x)" or "g(x)"
    if (str.find("lgamma") == std::string::npos && str.find("g(") == std::string::npos) {
        std::cerr << "[FAIL] Unary operator stripped! Result: " << str << std::endl;
        return false;
    }

    // Test Unary Constant Folding
    // lgamma(3) -> 1.791759
    root = parse_formula_string("lgamma(3)");
    simplified = DomainConstraints::fix_or_simplify(root);
    val = evaluate_tree(simplified, 0.0);
    ASSERT_NEAR(val, 1.791759, 1e-4);
    if (tree_size(simplified) != 1) {
         std::cerr << "[FAIL] Unary constant folding failed, tree size is " << tree_size(simplified) << std::endl;
         return false;
    }

    std::cout << "  -> Simplification Passed" << std::endl;
    return true;
}

bool test_fitness_calc() {
    std::cout << "Testing Fitness Calculation..." << std::endl;

    // Define simple target: y = x
    std::vector<double> targets = {1.0, 2.0, 3.0};
    std::vector<double> x_values = {1.0, 2.0, 3.0};

    // Perfect solution: x
    NodePtr solution = parse_formula_string("x");
#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
    double fitness = evaluate_fitness(solution, targets, x_values, nullptr, nullptr);
#else
    double fitness = evaluate_fitness(solution, targets, x_values);
#endif
    // Raw fitness (MSE or SumSq) should be 0. 
    // Complexity penalty might add a tiny bit.
    // If USE_RMSE_FITNESS is true, it is RMSE * (1+penalty).
    // RMSE of 0 is 0.
    ASSERT_NEAR(fitness, 0.0, 1e-9);

    // Imperfect solution: x+1
    // Preds: 2, 3, 4. Errors: 1, 1, 1. SumSq=3. MSE=1. RMSE=1.
    NodePtr imperfect = parse_formula_string("x+1");
#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
    fitness = evaluate_fitness(imperfect, targets, x_values, nullptr, nullptr);
#else
    fitness = evaluate_fitness(imperfect, targets, x_values);
#endif
    // It should be > 0.
    if (fitness <= 0.001) {
         std::cerr << "[FAIL] Imperfect solution has fitness 0!" << std::endl;
         return false;
    }

    std::cout << "  -> Fitness Calculation Passed" << std::endl;
    return true;
}

int main() {
    std::cout << "=============================" << std::endl;
    std::cout << " Running Operator Test Suite " << std::endl;
    std::cout << "=============================" << std::endl;

    bool all_passed = true;
    all_passed &= test_binary_operators();
    all_passed &= test_unary_operators();
    all_passed &= test_simplification();
    all_passed &= test_fitness_calc();

    std::cout << "=============================" << std::endl;
    if (all_passed) {
        std::cout << "ALL TESTS PASSED" << std::endl;
        return 0;
    } else {
        std::cerr << "SOME TESTS FAILED" << std::endl;
        return 1;
    }
}
