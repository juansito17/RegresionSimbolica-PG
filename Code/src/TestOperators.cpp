
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <string>
#include <cassert>
#include "ExpressionTree.h"
#include "AdvancedFeatures.h"
#include "Fitness.h"
#include "Globals.h"

// Define global variable for tests
int NUM_VARIABLES = 1;

// --- Helper Macros for Testing ---
#define ASSERT_NEAR(val1, val2, tol) \
    if (std::fabs((val1) - (val2)) > (tol)) { \
        std::cerr << "[FAIL] Line " << __LINE__ << ": Expected " << (val2) << ", got " << (val1) << " (diff: " << std::fabs((val1)-(val2)) << ")" << std::endl; \
        return false; \
    }

#define ASSERT_TRUE(cond) \
    if (!(cond)) { \
        std::cerr << "[FAIL] Line " << __LINE__ << ": Condition failed: " #cond << std::endl; \
        return false; \
    }

#define ASSERT_INF(val) \
    if ((val) != INF && !std::isinf(val)) { \
        std::cerr << "[FAIL] Line " << __LINE__ << ": Expected INF, got " << (val) << std::endl; \
        return false; \
    }

// =============================
// BINARY OPERATORS
// =============================
bool test_binary_operators() {
    std::cout << "Testing Binary Operators..." << std::endl;
    
    NodePtr root; double val;

    // Addition
    root = parse_formula_string("2+3");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_NEAR(val, 5.0, 1e-9);

    // Subtraction
    root = parse_formula_string("10-4");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_NEAR(val, 6.0, 1e-9);

    // Multiplication
    root = parse_formula_string("3*4");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_NEAR(val, 12.0, 1e-9);

    // Division
    root = parse_formula_string("10/2");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_NEAR(val, 5.0, 1e-9);

    // Power
    root = parse_formula_string("2^3");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_NEAR(val, 8.0, 1e-9);

    // Modulo
    root = parse_formula_string("10%3");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_NEAR(val, 1.0, 1e-9);

    // Negative numbers
    root = parse_formula_string("-5+3");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_NEAR(val, -2.0, 1e-9);

    // Operator precedence: 2+3*4 = 14
    root = parse_formula_string("2+3*4");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_NEAR(val, 14.0, 1e-9);

    // Parentheses override: (2+3)*4 = 20
    root = parse_formula_string("(2+3)*4");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_NEAR(val, 20.0, 1e-9);

    std::cout << "  -> Binary Operators Passed" << std::endl;
    return true;
}

// =============================
// UNARY OPERATORS
// =============================
bool test_unary_operators() {
    std::cout << "Testing Unary Operators..." << std::endl;
    NodePtr root; double val;

    // sin
    root = parse_formula_string("sin(0)");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_NEAR(val, 0.0, 1e-9);
    
    root = parse_formula_string("sin(1.5708)"); // pi/2
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_NEAR(val, 1.0, 1e-4);

    // cos
    root = parse_formula_string("cos(0)");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_NEAR(val, 1.0, 1e-9);

    // log (protected: log(|x|))
    root = parse_formula_string("log(2.7182818)");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_NEAR(val, 1.0, 1e-5);

    // exp
    root = parse_formula_string("exp(1)");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_NEAR(val, 2.7182818, 1e-5);

    // floor
    root = parse_formula_string("floor(2.9)");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_NEAR(val, 2.0, 1e-9);
    
    root = parse_formula_string("floor(-2.1)");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_NEAR(val, -3.0, 1e-9);

    // lgamma: Implementation is lgamma(|x|+1) => ln(|x|!)
    // lgamma(3) -> lgamma(4) = ln(3!) = ln(6) = 1.791759
    root = parse_formula_string("lgamma(3)"); 
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_NEAR(val, 1.791759, 1e-4);

    // g(x) alias
    root = parse_formula_string("g(3)");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_NEAR(val, 1.791759, 1e-4);

    // Factorial (!): Implementation is tgamma(|x|+1) = |x|!
    // gamma(4) = 3! = 6
    root = parse_formula_string("gamma(3)");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_NEAR(val, 6.0, 1e-4);

    std::cout << "  -> Unary Operators Passed" << std::endl;
    return true;
}

// =============================
// EDGE CASES (Protection)
// =============================
bool test_edge_cases() {
    std::cout << "Testing Edge Cases..." << std::endl;
    NodePtr root; double val;

    // Division by zero -> INF
    root = parse_formula_string("1/0");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_INF(val);

    // Modulo by zero -> INF
    root = parse_formula_string("5%0");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_INF(val);

    // log(0) -> INF (protected)
    root = parse_formula_string("log(0)");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_INF(val);

    // exp(800) -> INF (overflow)
    root = parse_formula_string("exp(800)");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_INF(val);

    // 0^(-1) -> INF
    root = parse_formula_string("0^(-1)");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_INF(val);

    // Factorial of large number -> INF
    root = parse_formula_string("gamma(200)");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_INF(val);

    // Negative base with non-integer exp -> INF (complex result)
    root = parse_formula_string("(-2)^0.5");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_INF(val);

    std::cout << "  -> Edge Cases Passed" << std::endl;
    return true;
}

// =============================
// SIMPLIFICATION RULES
// =============================
bool test_simplification() {
    std::cout << "Testing Simplification..." << std::endl;
    NodePtr root, simplified; double val; std::string str;

    // x - x -> 0
    root = parse_formula_string("x-x");
    simplified = DomainConstraints::fix_or_simplify(root);
    val = evaluate_tree(simplified, std::vector<double>{5.0});
    ASSERT_NEAR(val, 0.0, 1e-9);

    // x / x -> 1
    root = parse_formula_string("x/x");
    simplified = DomainConstraints::fix_or_simplify(root);
    val = evaluate_tree(simplified, std::vector<double>{5.0});
    ASSERT_NEAR(val, 1.0, 1e-9);

    // Constant Folding: 2+3 -> 5
    root = parse_formula_string("2+3");
    simplified = DomainConstraints::fix_or_simplify(root);
    val = evaluate_tree(simplified, 0.0);
    ASSERT_NEAR(val, 5.0, 1e-9);
    ASSERT_TRUE(tree_size(simplified) == 1);

    // x + 0 -> x
    root = parse_formula_string("x+0");
    simplified = DomainConstraints::fix_or_simplify(root);
    str = tree_to_string(simplified);
    ASSERT_TRUE(str == "x0");

    // x * 1 -> x
    root = parse_formula_string("x*1");
    simplified = DomainConstraints::fix_or_simplify(root);
    str = tree_to_string(simplified);
    ASSERT_TRUE(str == "x0");

    // x * 0 -> 0
    root = parse_formula_string("x*0");
    simplified = DomainConstraints::fix_or_simplify(root);
    val = evaluate_tree(simplified, std::vector<double>{100.0});
    ASSERT_NEAR(val, 0.0, 1e-9);

    // x^1 -> x
    root = parse_formula_string("x^1");
    simplified = DomainConstraints::fix_or_simplify(root);
    str = tree_to_string(simplified);
    ASSERT_TRUE(str == "x0");

    // x^0 -> 1
    root = parse_formula_string("x^0");
    simplified = DomainConstraints::fix_or_simplify(root);
    val = evaluate_tree(simplified, std::vector<double>{100.0});
    ASSERT_NEAR(val, 1.0, 1e-9);

    // Unary Operator Preservation: lgamma(x) should NOT simplify to x
    root = parse_formula_string("lgamma(x)");
    simplified = DomainConstraints::fix_or_simplify(root);
    str = tree_to_string(simplified);
    ASSERT_TRUE(str.find("lgamma") != std::string::npos || str.find("g(") != std::string::npos);

    // Unary Constant Folding: lgamma(3) -> constant
    root = parse_formula_string("lgamma(3)");
    simplified = DomainConstraints::fix_or_simplify(root);
    val = evaluate_tree(simplified, std::vector<double>{0.0});
    ASSERT_NEAR(val, 1.791759, 1e-4);
    ASSERT_TRUE(tree_size(simplified) == 1);

    // sin(0) -> 0 (constant folding)
    root = parse_formula_string("sin(0)");
    simplified = DomainConstraints::fix_or_simplify(root);
    val = evaluate_tree(simplified, std::vector<double>{0.0});
    ASSERT_NEAR(val, 0.0, 1e-9);
    ASSERT_TRUE(tree_size(simplified) == 1);

    std::cout << "  -> Simplification Passed" << std::endl;
    return true;
}

// =============================
// COMPLEX PARSING
// =============================
bool test_complex_parsing() {
    std::cout << "Testing Complex Parsing..." << std::endl;
    NodePtr root; double val;

    // Nested functions: sin(cos(0)) = sin(1) ≈ 0.8415
    root = parse_formula_string("sin(cos(0))");
    val = evaluate_tree(root, std::vector<double>{0.0});
    ASSERT_NEAR(val, std::sin(1.0), 1e-4);

    // Mixed: lgamma(x+1) at x=3 -> lgamma(4) = lgamma(5) = ln(4!) = ln(24) ≈ 3.178
    root = parse_formula_string("lgamma(x+1)");
    val = evaluate_tree(root, std::vector<double>{3.0});
    ASSERT_NEAR(val, std::lgamma(5.0), 1e-4);

    // Formula from project: (g(x)-((x*909613)/1000000))+0.24423 at x=4
    root = parse_formula_string("(g(x)-((x*909613)/1000000))+0.24423");
    val = evaluate_tree(root, std::vector<double>{4.0});
    double expected = std::lgamma(5.0) - (4.0 * 909613.0 / 1000000.0) + 0.24423;
    ASSERT_NEAR(val, expected, 1e-4);

    // Implicit multiplication: 2x at x=3 -> 6
    root = parse_formula_string("2x");
    val = evaluate_tree(root, std::vector<double>{3.0});
    ASSERT_NEAR(val, 6.0, 1e-9);

    // Deep nesting: exp(log(x)) at x=5 -> 5
    root = parse_formula_string("exp(log(x))");
    val = evaluate_tree(root, std::vector<double>{5.0});
    ASSERT_NEAR(val, 5.0, 1e-4);

    // Chained operations: x^2+2*x+1 at x=3 -> 16
    root = parse_formula_string("x^2+2*x+1");
    val = evaluate_tree(root, std::vector<double>{3.0});
    ASSERT_NEAR(val, 16.0, 1e-9);

    std::cout << "  -> Complex Parsing Passed" << std::endl;
    return true;
}

// =============================
// FITNESS CALCULATION
// =============================
bool test_fitness_calc() {
    std::cout << "Testing Fitness Calculation..." << std::endl;

    std::vector<double> targets = {1.0, 2.0, 3.0};
    std::vector<std::vector<double>> x_values = {{1.0}, {2.0}, {3.0}};

    // Perfect solution: x
    NodePtr solution = parse_formula_string("x0");
#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
    double fitness = evaluate_fitness(solution, targets, x_values, (double*)nullptr, (double*)nullptr);
#else
    double fitness = evaluate_fitness(solution, targets, x_values);
#endif
    ASSERT_NEAR(fitness, 0.0, 1e-9);

    // Imperfect solution: x+1 (errors: 1, 1, 1)
    NodePtr imperfect = parse_formula_string("x+1");
#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
    fitness = evaluate_fitness(imperfect, targets, x_values, (double*)nullptr, (double*)nullptr);
#else
    fitness = evaluate_fitness(imperfect, targets, x_values);
#endif
    ASSERT_TRUE(fitness > 0.001);

    // Very bad solution: constant 100
    NodePtr bad = parse_formula_string("100");
#ifdef USE_GPU_ACCELERATION_DEFINED_BY_CMAKE
    double bad_fitness = evaluate_fitness(bad, targets, x_values, (double*)nullptr, (double*)nullptr);
#else
    double bad_fitness = evaluate_fitness(bad, targets, x_values);
#endif
    ASSERT_TRUE(bad_fitness > fitness); // Worse than x+1

    std::cout << "  -> Fitness Calculation Passed" << std::endl;
    return true;
}

// =============================
// TREE UTILITIES
// =============================
bool test_tree_utilities() {
    std::cout << "Testing Tree Utilities..." << std::endl;
    
    // tree_size
    NodePtr root = parse_formula_string("x+1");
    ASSERT_TRUE(tree_size(root) == 3); // +, x, 1

    root = parse_formula_string("lgamma(x)");
    ASSERT_TRUE(tree_size(root) == 2); // lgamma, x

    // tree_to_string roundtrip
    root = parse_formula_string("(x+1)*2");
    std::string str = tree_to_string(root);
    ASSERT_TRUE(str.find("x") != std::string::npos);
    ASSERT_TRUE(str.find("1") != std::string::npos);
    ASSERT_TRUE(str.find("2") != std::string::npos);

    // clone_tree
    NodePtr cloned = clone_tree(root);
    ASSERT_TRUE(tree_to_string(cloned) == tree_to_string(root));
    ASSERT_TRUE(cloned.get() != root.get()); // Different pointers

    std::cout << "  -> Tree Utilities Passed" << std::endl;
    return true;
}

// =============================
// MAIN
// =============================
int main() {
    std::cout << "=======================================" << std::endl;
    std::cout << " Running Comprehensive Operator Tests " << std::endl;
    std::cout << "=======================================" << std::endl;

    bool all_passed = true;
    all_passed &= test_binary_operators();
    all_passed &= test_unary_operators();
    all_passed &= test_edge_cases();
    all_passed &= test_simplification();
    all_passed &= test_complex_parsing();
    all_passed &= test_fitness_calc();
    all_passed &= test_tree_utilities();

    std::cout << "=======================================" << std::endl;
    if (all_passed) {
        std::cout << "ALL TESTS PASSED (" << 7 << " test suites)" << std::endl;
        return 0;
    } else {
        std::cerr << "SOME TESTS FAILED" << std::endl;
        return 1;
    }
}
