#ifndef FP8_ADAMW_CONPAND
#define FP8_ADAMW_CONPAND

#include <torch/types.h>

void FP8_AdamW_expand(torch::Tensor params,   // parameter
                      torch::Tensor grads,    // gradient
                      torch::Tensor exp_avg,  // first order momentum
                      torch::Tensor scale_exp_avg, torch::Tensor expand_exp_avg,
                      torch::Tensor sqrtminmax_exp_avg,
                      torch::Tensor exp_avg_sq,  // second order momentum
                      torch::Tensor scale_exp_avg_sq,
                      torch::Tensor expand_exp_avg_sq,
                      torch::Tensor sqrtminmax_exp_avg_sq, float beta1,
                      float beta2, float lr, float wd, float eps, int step,
                      int qgroup_size, int expand_min);

#endif  // FP8_ADAMW_CONPAND
