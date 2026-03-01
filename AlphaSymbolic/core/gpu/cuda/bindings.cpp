#include <torch/extension.h>
#include <vector>
#include <string>
#include <pybind11/stl.h>

// Forward declaration
void launch_rpn_kernel(
    const torch::Tensor& population,
    const torch::Tensor& x,
    const torch::Tensor& constants,
    torch::Tensor& out_preds,
    torch::Tensor& out_sp,
    torch::Tensor& out_error,
    int PAD_ID, 
    int id_x_start, 
    int id_C, int id_pi, int id_e,
    int id_0, int id_1, int id_2, int id_3, int id_4, int id_5, int id_6, int id_10,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan,
    double pi_val, double e_val,
    int strict_mode = 0
);

// Fused Eval Kernel (block-per-individual + RMSE)
void launch_rpn_eval_fused(
    const torch::Tensor& population,
    const torch::Tensor& x,
    const torch::Tensor& constants,
    const torch::Tensor& y_target,
    torch::Tensor& out_rmse,
    int PAD_ID, int id_x_start,
    int id_C, int id_pi, int id_e,
    int id_0, int id_1, int id_2, int id_3, int id_4, int id_5, int id_6, int id_10,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan, int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan,
    double pi_val, double e_val,
    int strict_mode = 0
);

void run_rpn_cuda(
    torch::Tensor population, 
    torch::Tensor x, // [Vars, D]
    torch::Tensor constants, 
    torch::Tensor out_preds,
    torch::Tensor out_sp,
    torch::Tensor out_error,
    int PAD_ID, 
    int id_x_start, 
    int id_C, int id_pi, int id_e,
    int id_0, int id_1, int id_2, int id_3, int id_4, int id_5, int id_6, int id_10,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan,
    double pi_val, double e_val,
    int strict_mode
) {
    launch_rpn_kernel(
        population, x, constants, out_preds, out_sp, out_error,
        PAD_ID, id_x_start, 
        id_C, id_pi, id_e,
        id_0, id_1, id_2, id_3, id_4, id_5, id_6, id_10,
        op_add, op_sub, op_mul, op_div, op_pow, op_mod,
        op_sin, op_cos, op_tan,
        op_log, op_exp,
        op_sqrt, op_abs, op_neg,
        op_fact, op_floor, op_ceil, op_sign,
        op_gamma, op_lgamma,
        op_asin, op_acos, op_atan,
        pi_val, e_val,
        strict_mode
    );
}

// Phase 6 Backward Wrapper
void launch_rpn_backward(
    const torch::Tensor& population,
    const torch::Tensor& x,
    const torch::Tensor& constants,
    const torch::Tensor& grad_output,
    torch::Tensor& grad_constants,
    int PAD_ID, int id_x_start, 
    int id_C, int id_pi, int id_e,
    int id_0, int id_1, int id_2, int id_3, int id_4, int id_5, int id_6, int id_10,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan,
    double pi_val, double e_val
);

void run_rpn_backward_cuda(
    torch::Tensor population, 
    torch::Tensor x,
    torch::Tensor constants, 
    torch::Tensor grad_output,
    torch::Tensor grad_constants,
    int PAD_ID, int id_x_start, 
    int id_C, int id_pi, int id_e,
    int id_0, int id_1, int id_2, int id_3, int id_4, int id_5, int id_6, int id_10,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan,
    double pi_val, double e_val
) {
    launch_rpn_backward(
        population, x, constants, grad_output, grad_constants,
        PAD_ID, id_x_start, 
        id_C, id_pi, id_e,
        id_0, id_1, id_2, id_3, id_4, id_5, id_6, id_10,
        op_add, op_sub, op_mul, op_div, op_pow, op_mod,
        op_sin, op_cos, op_tan,
        op_log, op_exp,
        op_sqrt, op_abs, op_neg,
        op_fact, op_floor, op_ceil, op_sign,
        op_gamma, op_lgamma,
        op_asin, op_acos, op_atan,
        pi_val, e_val
    );
}

// Forward declaration for decoder
std::vector<std::string> decode_rpn(
    torch::Tensor population, 
    torch::Tensor constants, 
    const std::vector<std::string>& vocab,
    const std::vector<int>& arities,
    int PAD_ID,
    int precision
);

// --- Phase 2 Forward Declarations ---
void launch_find_subtree_ranges(
    const torch::Tensor& population,
    const torch::Tensor& token_arities,
    torch::Tensor& out_starts,
    int PAD_ID
);

void launch_mutation_kernel(
    torch::Tensor& population,
    const torch::Tensor& rand_floats,
    const torch::Tensor& rand_ints,
    const torch::Tensor& token_arities,
    const torch::Tensor& arity_0_ids,
    const torch::Tensor& arity_1_ids,
    const torch::Tensor& arity_2_ids,
    float mutation_rate,
    int PAD_ID
);

void launch_validate_rpn_batch(
    const torch::Tensor& population,
    const torch::Tensor& token_arities,
    torch::Tensor& out_valid,
    int PAD_ID
);

void launch_crossover_splicing(
    const torch::Tensor& parent1,
    const torch::Tensor& parent2,
    const torch::Tensor& starts1,
    const torch::Tensor& ends1,
    const torch::Tensor& starts2,
    const torch::Tensor& ends2,
    torch::Tensor& child1,
    torch::Tensor& child2,
    int PAD_ID
);

// --- Phase 3 Forward Declarations ---
void launch_tournament_selection(
    const torch::Tensor& fitness,
    const torch::Tensor& errors,
    const torch::Tensor& rand_idx,
    const torch::Tensor& rand_cases,
    torch::Tensor& winner_idx,
    const torch::Tensor& lengths,
    const torch::Tensor& mad_eps
);

void launch_pso_update(
    torch::Tensor& particles,
    torch::Tensor& velocities,
    const torch::Tensor& pbest_pos,
    const torch::Tensor& gbest_pos,
    const torch::Tensor& r1,
    const torch::Tensor& r2,
    float w, float c1, float c2
);

void launch_pso_update_bests(
    const torch::Tensor& current_err,
    torch::Tensor& pbest_err,
    torch::Tensor& pbest_pos,
    const torch::Tensor& current_pos,
    torch::Tensor& gbest_err,
    torch::Tensor& gbest_pos
);

std::vector<torch::Tensor> evolve_generation(
    torch::Tensor population,      // [B, L]
    torch::Tensor constants,       // [B, K]
    torch::Tensor fitness,         // [B]
    torch::Tensor abs_errors,     // [B, N_data] or Empty
    torch::Tensor X,               // [Vars, N_data]
    torch::Tensor Y_target,        // [N_data]
    torch::Tensor lengths,         // [B] int32 (for parsimony)
    torch::Tensor token_arities,   // [VocabSize] int32
    torch::Tensor arity_0_ids,     // [n0] int64
    torch::Tensor arity_1_ids,     // [n1] int64
    torch::Tensor arity_2_ids,     // [n2] int64
    torch::Tensor mutation_bank,   // [BankSize, L] or Empty
    torch::Tensor mad_eps,         // [N_data] or Empty (Phase 3)
    float mutation_rate,
    float crossover_rate,
    int tournament_size,
    int pso_steps,
    int pso_particles,
    float pso_w, float pso_c1, float pso_c2,
    int PAD_ID,
    // OpCodes
    int id_x_start, 
    int id_C, int id_pi, int id_e,
    int id_0, int id_1, int id_2, int id_3, int id_4, int id_5, int id_6, int id_10,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan,
    double pi_val, double e_val,
    int n_islands
);

// --- Phase 6 Forward Declaration: Fused PSO ---
void launch_fused_pso(
    const torch::Tensor& population,
    const torch::Tensor& init_consts,
    const torch::Tensor& x,
    const torch::Tensor& y_target,
    torch::Tensor& out_gbest_pos,
    torch::Tensor& out_gbest_err,
    int num_particles, int num_steps,
    float w, float c1, float c2,
    float const_min, float const_max,
    int PAD_ID, int id_x_start,
    int id_C, int id_pi, int id_e,
    int id_0, int id_1, int id_2, int id_3, int id_4, int id_5, int id_6, int id_10,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan,
    double pi_val, double e_val
);

// --- Phase 5 Forward Declarations (Simplifier + Generator Kernels) ---
void launch_simplify_batch(
    torch::Tensor& population,
    const torch::Tensor& arities,
    const torch::Tensor& val_table,
    const torch::Tensor& literal_ids,
    const torch::Tensor& literal_vals,
    int max_passes,
    int op_plus, int op_minus, int op_mult, int op_div,
    int op_neg, int op_mod, int op_pow,
    int op_sin, int op_cos, int op_tan,
    int op_asin, int op_acos, int op_atan,
    int op_log, int op_exp, int op_sqrt, int op_abs,
    int op_gamma, int op_lgamma,
    int op_floor, int op_ceil, int op_sign,
    int id_0, int id_1, int id_2, int id_3, int id_4, int id_5, int id_6
);

void launch_precompute_subtree_starts(
    const torch::Tensor& population,
    const torch::Tensor& arities,
    torch::Tensor& out_starts
);

void launch_generate_random_rpn(
    torch::Tensor& population,
    const torch::Tensor& terminal_ids,
    const torch::Tensor& unary_ids,
    const torch::Tensor& binary_ids,
    uint64_t seed,
    float term_weight,   // OPTIMIZED: peso categoria terminal
    float unary_weight,  // OPTIMIZED: peso categoria unaria
    float bin_weight     // OPTIMIZED: peso categoria binaria
);

// --- Diversity Kernels Forward Declarations (Structural Hash & Dedup) ---
void launch_compute_population_hashes(
    const torch::Tensor& population,
    torch::Tensor& hashes,
    torch::Tensor& var_presence,
    int PAD_ID,
    int id_x_start,
    int num_vars
);

void launch_structural_dedup(
    const torch::Tensor& hashes,
    torch::Tensor& hash_table,
    torch::Tensor& duplicate_mask,
    torch::Tensor& original_index
);

int64_t launch_count_unique(const torch::Tensor& duplicate_mask);

int64_t launch_get_replacement_positions(
    const torch::Tensor& duplicate_mask,
    torch::Tensor& replacement_positions,
    torch::Tensor& n_replacements
);

void launch_compute_var_presence(
    const torch::Tensor& population,
    torch::Tensor& var_presence,
    int PAD_ID,
    int id_x_start,
    int num_vars
);

// --- L-BFGS-B Forward Declaration ---
void launch_lbfgs_optimize(
    const torch::Tensor& population,
    torch::Tensor& constants,
    const torch::Tensor& x,
    const torch::Tensor& y_target,
    torch::Tensor& out_rmse,
    int max_iter,
    int history_size,
    float gtol,
    float const_min,
    float const_max,
    int PAD_ID, int id_x_start,
    int id_C, int id_pi, int id_e,
    int id_0, int id_1, int id_2, int id_3, int id_4, int id_5, int id_6, int id_10,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_ceil, int op_sign,
    int op_gamma, int op_lgamma,
    int op_asin, int op_acos, int op_atan,
    double pi_val, double e_val
);

// --- Best Tracker Forward Declarations ---
void launch_update_best(
    const torch::Tensor& population,
    const torch::Tensor& constants,
    const torch::Tensor& fitness,
    torch::Tensor& tracker_rpn,
    torch::Tensor& tracker_consts,
    torch::Tensor& tracker_rmse,
    torch::Tensor& tracker_idx,
    torch::Tensor& tracker_gen,
    torch::Tensor& tracker_len,
    torch::Tensor& tracker_updated,
    int current_generation,
    float tolerance
);

void launch_check_improvement(
    const torch::Tensor& fitness,
    const torch::Tensor& tracked_rmse,
    torch::Tensor& improved,
    float tolerance
);

void launch_batch_update_best(
    const torch::Tensor& population,
    const torch::Tensor& constants,
    const torch::Tensor& fitness,
    torch::Tensor& best_rpn,
    torch::Tensor& best_consts,
    torch::Tensor& best_rmse,
    float tolerance
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("eval_rpn", &run_rpn_cuda, "RPN Evaluation Kernel (CUDA)");
    m.def("eval_rpn_fused", &launch_rpn_eval_fused, "Fused RPN Eval+RMSE (block-per-individual, zero warp divergence)",
        py::arg("population"), py::arg("x"), py::arg("constants"), py::arg("y_target"), py::arg("out_rmse"),
        py::arg("PAD_ID"), py::arg("id_x_start"),
        py::arg("id_C"), py::arg("id_pi"), py::arg("id_e"),
        py::arg("id_0"), py::arg("id_1"), py::arg("id_2"), py::arg("id_3"), py::arg("id_4"), py::arg("id_5"), py::arg("id_6"), py::arg("id_10"),
        py::arg("op_add"), py::arg("op_sub"), py::arg("op_mul"), py::arg("op_div"), py::arg("op_pow"), py::arg("op_mod"),
        py::arg("op_sin"), py::arg("op_cos"), py::arg("op_tan"), py::arg("op_log"), py::arg("op_exp"),
        py::arg("op_sqrt"), py::arg("op_abs"), py::arg("op_neg"),
        py::arg("op_fact"), py::arg("op_floor"), py::arg("op_ceil"), py::arg("op_sign"),
        py::arg("op_gamma"), py::arg("op_lgamma"),
        py::arg("op_asin"), py::arg("op_acos"), py::arg("op_atan"),
        py::arg("pi_val"), py::arg("e_val"),
        py::arg("strict_mode") = 0);
    m.def("decode_rpn", &decode_rpn, "RPN Decoder (C++)",
        py::arg("population"), py::arg("constants"), py::arg("vocab"), py::arg("arities"), py::arg("PAD_ID"), py::arg("precision") = 4);
    
    // Phase 2
    m.def("find_subtree_ranges", &launch_find_subtree_ranges, "Find Subtree Ranges (CUDA)");
    m.def("mutate_population", &launch_mutation_kernel, "Mutation Kernel (CUDA)");
    m.def("crossover_splicing", &launch_crossover_splicing, "Crossover Splicing Kernel (CUDA)");
    m.def("validate_rpn_batch", &launch_validate_rpn_batch, "Validate RPN Batch Kernel (CUDA)");
    
    // Phase 3
    m.def("tournament_selection", &launch_tournament_selection, "Tournament Selection (CUDA)",
        py::arg("fitness"), py::arg("errors"), py::arg("rand_idx"), py::arg("rand_cases"), py::arg("selected_idx"), py::arg("lengths"), py::arg("mad_eps") = torch::empty({0}, torch::kFloat32));
    m.def("pso_update", &launch_pso_update, "PSO Update (CUDA)");
    m.def("pso_update_bests", &launch_pso_update_bests, "PSO Update Bests (CUDA)");

    // Phase 4
    m.def("evolve_generation", &evolve_generation, "Full Evolution Generation (C++)",
        py::arg("population"), py::arg("constants"), py::arg("fitness"), py::arg("abs_errors"),
        py::arg("X"), py::arg("Y_target"), py::arg("lengths"), // Added lengths
        py::arg("token_arities"), py::arg("arity_0_ids"), py::arg("arity_1_ids"), py::arg("arity_2_ids"),
        py::arg("mutation_bank"), py::arg("mad_eps"),
        py::arg("mutation_rate"), py::arg("crossover_rate"),
        py::arg("tournament_size"),
        py::arg("pso_steps"), py::arg("pso_particles"),
        py::arg("pso_w"), py::arg("pso_c1"), py::arg("pso_c2"),
        py::arg("PAD_ID"),
        py::arg("id_x_start"), 
        py::arg("id_C"), py::arg("id_pi"), py::arg("id_e"),
        py::arg("id_0"), py::arg("id_1"), py::arg("id_2"), py::arg("id_3"), py::arg("id_4"), py::arg("id_5"), py::arg("id_6"), py::arg("id_10"),
        py::arg("op_add"), py::arg("op_sub"), py::arg("op_mul"), py::arg("op_div"), py::arg("op_pow"), py::arg("op_mod"),
        py::arg("op_sin"), py::arg("op_cos"), py::arg("op_tan"),
        py::arg("op_log"), py::arg("op_exp"),
        py::arg("op_sqrt"), py::arg("op_abs"), py::arg("op_neg"),
        py::arg("op_fact"), py::arg("op_floor"), py::arg("op_ceil"), py::arg("op_sign"),
        py::arg("op_gamma"), py::arg("op_lgamma"),
        py::arg("op_asin"), py::arg("op_acos"), py::arg("op_atan"),
        py::arg("pi_val"), py::arg("e_val"),
        py::arg("n_islands") = 1
    );

    // Phase 5: Simplifier + Generator Kernels
    m.def("simplify_batch", &launch_simplify_batch, "Batch Simplification (CUDA)");
    m.def("precompute_subtree_starts", &launch_precompute_subtree_starts, "Precompute Subtree Starts (CUDA)");
    m.def("generate_random_rpn", &launch_generate_random_rpn, "Random RPN Generation (CUDA)");

    // Phase 6: Fused PSO and Autograd
    m.def("fused_pso", &launch_fused_pso, "Fused PSO (Eval+PSO in single kernel)");
    m.def("eval_rpn_backward", &run_rpn_backward_cuda, "RPN Backward Pass (CUDA)");

    // Diversity Kernels: Structural Hash & Deduplication
    m.def("compute_population_hashes", &launch_compute_population_hashes, 
        "Compute structural hashes for population (CUDA)",
        py::arg("population"), py::arg("hashes"), py::arg("var_presence"),
        py::arg("PAD_ID"), py::arg("id_x_start"), py::arg("num_vars"));
    
    m.def("structural_dedup", &launch_structural_dedup,
        "Find duplicate formulas via structural hash (CUDA)",
        py::arg("hashes"), py::arg("hash_table"), py::arg("duplicate_mask"), py::arg("original_index"));
    
    m.def("count_unique", &launch_count_unique,
        "Count unique formulas (CUDA)",
        py::arg("duplicate_mask"));
    
    m.def("get_replacement_positions", &launch_get_replacement_positions,
        "Get positions of duplicates for replacement (CUDA)",
        py::arg("duplicate_mask"), py::arg("replacement_positions"), py::arg("n_replacements"));
    
    m.def("compute_var_presence", &launch_compute_var_presence,
        "Compute variable presence bitmask for population (CUDA)",
        py::arg("population"), py::arg("var_presence"),
        py::arg("PAD_ID"), py::arg("id_x_start"), py::arg("num_vars"));

    // L-BFGS-B Optimizer (CUDA)
    m.def("lbfgs_optimize", &launch_lbfgs_optimize,
        "L-BFGS-B constant optimization (CUDA)",
        py::arg("population"), py::arg("constants"), py::arg("x"), py::arg("y_target"),
        py::arg("out_rmse"),
        py::arg("max_iter") = 20,
        py::arg("history_size") = 10,
        py::arg("gtol") = 1e-7f,
        py::arg("const_min") = -25.0f,
        py::arg("const_max") = 25.0f,
        py::arg("PAD_ID"),
        py::arg("id_x_start"),
        py::arg("id_C"), py::arg("id_pi"), py::arg("id_e"),
        py::arg("id_0"), py::arg("id_1"), py::arg("id_2"), py::arg("id_3"), py::arg("id_4"), py::arg("id_5"), py::arg("id_6"), py::arg("id_10"),
        py::arg("op_add"), py::arg("op_sub"), py::arg("op_mul"), py::arg("op_div"), py::arg("op_pow"), py::arg("op_mod"),
        py::arg("op_sin"), py::arg("op_cos"), py::arg("op_tan"),
        py::arg("op_log"), py::arg("op_exp"),
        py::arg("op_sqrt"), py::arg("op_abs"), py::arg("op_neg"),
        py::arg("op_fact"), py::arg("op_floor"), py::arg("op_ceil"), py::arg("op_sign"),
        py::arg("op_gamma"), py::arg("op_lgamma"),
        py::arg("op_asin"), py::arg("op_acos"), py::arg("op_atan"),
        py::arg("pi_val"), py::arg("e_val"));

    // Best Tracker Kernels (CUDA)
    m.def("update_best", &launch_update_best,
        "Update best individual tracker (CUDA)",
        py::arg("population"), py::arg("constants"), py::arg("fitness"),
        py::arg("tracker_rpn"), py::arg("tracker_consts"), py::arg("tracker_rmse"),
        py::arg("tracker_idx"), py::arg("tracker_gen"), py::arg("tracker_len"), py::arg("tracker_updated"),
        py::arg("current_generation"), py::arg("tolerance") = 1e-9f);
    
    m.def("check_improvement", &launch_check_improvement,
        "Check if best improved without sync (CUDA)",
        py::arg("fitness"), py::arg("tracked_rmse"), py::arg("improved"), py::arg("tolerance") = 1e-9f);
    
    m.def("batch_update_best", &launch_batch_update_best,
        "Batch update best individual (CUDA)",
        py::arg("population"), py::arg("constants"), py::arg("fitness"),
        py::arg("best_rpn"), py::arg("best_consts"), py::arg("best_rmse"),
        py::arg("tolerance") = 1e-9f);
}