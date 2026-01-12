
#include <torch/extension.h>

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
    int id_1, int id_2, int id_3, int id_5,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_gamma,
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
    int id_1, int id_2, int id_3, int id_5,
    int op_add, int op_sub, int op_mul, int op_div, int op_pow, int op_mod,
    int op_sin, int op_cos, int op_tan,
    int op_log, int op_exp,
    int op_sqrt, int op_abs, int op_neg,
    int op_fact, int op_floor, int op_gamma,
    int op_asin, int op_acos, int op_atan,
    double pi_val, double e_val
) {
    launch_rpn_kernel(
        population, x, constants, out_preds, out_sp, out_error,
        PAD_ID, id_x_start, 
        id_C, id_pi, id_e,
        id_1, id_2, id_3, id_5,
        op_add, op_sub, op_mul, op_div, op_pow, op_mod,
        op_sin, op_cos, op_tan,
        op_log, op_exp,
        op_sqrt, op_abs, op_neg,
        op_fact, op_floor, op_gamma,
        op_asin, op_acos, op_atan,
        pi_val, e_val
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("eval_rpn", &run_rpn_cuda, "RPN Evaluation Kernel (CUDA)");
}
