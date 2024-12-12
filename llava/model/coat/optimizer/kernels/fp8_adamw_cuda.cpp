#include <torch/extension.h>
#include <torch/torch.h>

void FP8_AdamW_cuda(torch::Tensor params,   // parameter
                    torch::Tensor grads,    // gradient
                    torch::Tensor exp_avg,  // first order momentum
                    torch::Tensor scale_exp_avg,
                    torch::Tensor exp_avg_sq,  // second order momentum
                    torch::Tensor scale_exp_avg_sq, float beta1, float beta2,
                    float lr, float wd, float eps, int step,
                    int qgroup_size  // other parameters
);

void FP8_AdamW(torch::Tensor params,   // parameter
               torch::Tensor grads,    // gradient
               torch::Tensor exp_avg,  // first order momentum
               torch::Tensor scale_exp_avg,
               torch::Tensor exp_avg_sq,  // second order momentum
               torch::Tensor scale_exp_avg_sq, float beta1, float beta2,
               float lr, float wd, float eps, int step,
               int qgroup_size) {  // other parameters

  FP8_AdamW_cuda(params, grads, exp_avg, scale_exp_avg, exp_avg_sq,
                 scale_exp_avg_sq, beta1, beta2, lr, wd, eps, step,
                 qgroup_size);
}
