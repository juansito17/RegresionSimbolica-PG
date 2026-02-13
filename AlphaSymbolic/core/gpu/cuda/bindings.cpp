
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
    double pi_val, double e_val
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
    double pi_val, double e_val
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
        pi_val, e_val
    );
}

// Forward declaration for decoder
std::vector<std::string> decode_rpn(
    torch::Tensor population, 
    torch::Tensor constants, 
    const std::vector<std::string>& vocab,
    const std::vector<int>& arities,
    int PAD_ID
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
    torch::Tensor& selected_idx
);

void launch_pso_update(
    torch::Tensor& pos,
    torch::Tensor& vel,
    const torch::Tensor& pbest,
    const torch::Tensor& gbest,
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
    torch::Tensor token_arities,   // [VocabSize] int32
    torch::Tensor arity_0_ids,     // [n0] int64
    torch::Tensor arity_1_ids,     // [n1] int64
    torch::Tensor arity_2_ids,     // [n2] int64
    torch::Tensor mutation_bank,   // [BankSize, L] or Empty
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
    double pi_val, double e_val
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
    uint64_t seed
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("eval_rpn", &run_rpn_cuda, "RPN Evaluation Kernel (CUDA)");
    m.def("decode_rpn", &decode_rpn, "RPN Decoder (C++)");
    
    // Phase 2
    m.def("find_subtree_ranges", &launch_find_subtree_ranges, "Find Subtree Ranges (CUDA)");
    m.def("mutate_population", &launch_mutation_kernel, "Mutation Kernel (CUDA)");
    m.def("crossover_splicing", &launch_crossover_splicing, "Crossover Splicing Kernel (CUDA)");
    
    // Phase 3
    m.def("tournament_selection", &launch_tournament_selection, "Tournament Selection (CUDA)");
    m.def("pso_update", &launch_pso_update, "PSO Update (CUDA)");
    m.def("pso_update_bests", &launch_pso_update_bests, "PSO Update Bests (CUDA)");

    // Phase 4
    m.def("evolve_generation", &evolve_generation, "Full Evolution Generation (C++)");

    // Phase 5: Simplifier + Generator Kernels
    m.def("simplify_batch", &launch_simplify_batch, "Batch Simplification (CUDA)");
    m.def("precompute_subtree_starts", &launch_precompute_subtree_starts, "Precompute Subtree Starts (CUDA)");
    m.def("generate_random_rpn", &launch_generate_random_rpn, "Random RPN Generation (CUDA)");

    // Phase 6: Fused PSO
    m.def("fused_pso", &launch_fused_pso, "Fused PSO (Eval+PSO in single kernel)");
}


