#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <torch/extension.h>

#define QUANT_MIN_VAL 1e-20

namespace cg = cooperative_groups;
#define WARPSIZE 32

template <typename scalar_t>
__global__ void fp8_adamw_cuda_expand_kernel(
    scalar_t* __restrict__ params, scalar_t* __restrict__ grads,
    __nv_fp8_e4m3* __restrict__ exp_avg, float* __restrict__ scale_exp_avg,
    float* __restrict__ expand_exp_avg, float* __restrict__ sqrtminmax_exp_avg,
    __nv_fp8_e4m3* __restrict__ exp_avg_sq,
    float* __restrict__ scale_exp_avg_sq, float* __restrict__ expand_exp_avg_sq,
    float* __restrict__ sqrtminmax_exp_avg_sq, float beta1, float beta2,
    float lr, float wd, float eps, int step, int qgroup_size, int expand_min,
    int total_elements, int total_scale_elements) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int scale_idx = blockIdx.x;

  float float_exp_avg, float_exp_avg_sq;
  float correction1, correction2_sqrt;
  float denom, update;

  if (idx < total_elements) {
    // dequantize the optimizer states
    float_exp_avg = float(exp_avg[idx]) * scale_exp_avg[scale_idx];
    int sign_exp_avg = 1 - 2 * signbit(float_exp_avg);
    float_exp_avg = sign_exp_avg *
                    powf(fabsf(float_exp_avg), 1 / expand_exp_avg[scale_idx]) *
                    sqrtminmax_exp_avg[scale_idx];
    float_exp_avg_sq = float(exp_avg_sq[idx]) * scale_exp_avg_sq[scale_idx];
    float_exp_avg_sq =
        powf(float_exp_avg_sq, 1 / expand_exp_avg_sq[scale_idx]) *
        sqrtminmax_exp_avg_sq[scale_idx];

    // calculation of optimizer.step()
    float_exp_avg = beta1 * float_exp_avg + (1 - beta1) * grads[idx];
    float_exp_avg_sq =
        beta2 * float_exp_avg_sq + (1 - beta2) * grads[idx] * grads[idx];

    correction1 = 1.0f - powf(beta1, step);
    correction2_sqrt = sqrtf(1.0f - powf(beta2, step));

    denom = (sqrtf(float_exp_avg_sq) / correction2_sqrt + eps) * correction1;
    update = (float_exp_avg / denom) + (wd * params[idx]);
    params[idx] = params[idx] - (lr * update);
  } else {
    float_exp_avg = 0.0f;
    float_exp_avg_sq = 0.0f;
  }

  //// quantize the first-order and second-order momentum
  int wid = threadIdx.x / WARPSIZE;

  // reduction within a warp
  __shared__ float sharedFirstMaxVal[32];
  __shared__ float sharedFirstMinVal[32];
  __shared__ float sharedSecondMaxVal[32];
  __shared__ float sharedSecondMinVal[32];
  cg::thread_block_tile<32> warpTile =
      cg::tiled_partition<32>(cg::this_thread_block());
  float firstMaxVal = fabsf(float_exp_avg);
  float firstMinVal = fabsf(float_exp_avg);
  float secondMaxVal = fabsf(float_exp_avg_sq);
  float secondMinVal = fabsf(float_exp_avg_sq);
  // Special Handel
  if (idx >= total_elements) {
    firstMinVal = __int_as_float(0x7f7fffff);
    secondMinVal = __int_as_float(0x7f7fffff);
  }

  for (int i = warpTile.size() / 2; i > 0; i /= 2) {
    float reduceFirstMaxVal = warpTile.shfl_down(firstMaxVal, i);
    float reduceFirstMinVal = warpTile.shfl_down(firstMinVal, i);
    float reduceSecondMaxVal = warpTile.shfl_down(secondMaxVal, i);
    float reduceSecondMinVal = warpTile.shfl_down(secondMinVal, i);
    firstMaxVal = fmax(firstMaxVal, fabsf(reduceFirstMaxVal));
    firstMinVal = fmin(firstMinVal, fabsf(reduceFirstMinVal));
    secondMaxVal = fmax(secondMaxVal, fabsf(reduceSecondMaxVal));
    secondMinVal = fmin(secondMinVal, fabsf(reduceSecondMinVal));
    // printf("First Max: %f\n", reduceFirstMaxVal);
  }
  int lane = warpTile.thread_rank();
  if (lane == 0) {
    sharedFirstMaxVal[wid] = firstMaxVal;
    sharedFirstMinVal[wid] = firstMinVal;
    sharedSecondMaxVal[wid] = secondMaxVal;
    sharedSecondMinVal[wid] = secondMinVal;
  }

  __syncthreads();

  // reduction within a block
  __shared__ float shared_absmax_exp_avg;
  __shared__ float shared_absmin_exp_avg;
  __shared__ float shared_absmax_exp_avg_sq;
  __shared__ float shared_absmin_exp_avg_sq;
  firstMaxVal =
      (threadIdx.x < blockDim.x / warpSize) ? sharedFirstMaxVal[lane] : 0;
  firstMinVal =
      (threadIdx.x < blockDim.x / warpSize) ? sharedFirstMinVal[lane] : 1e9;
  secondMaxVal =
      (threadIdx.x < blockDim.x / warpSize) ? sharedSecondMaxVal[lane] : 0;
  secondMinVal =
      (threadIdx.x < blockDim.x / warpSize) ? sharedSecondMinVal[lane] : 1e9;
  if (wid == 0) {
    for (int offset = WARPSIZE / 2; offset > 0; offset /= 2) {
      float reduceFirstMaxVal =
          __shfl_down_sync(0xFFFFFFFF, firstMaxVal, offset);
      float reduceFirstMinVal =
          __shfl_down_sync(0xFFFFFFFF, firstMinVal, offset);
      float reduceSecondMaxVal =
          __shfl_down_sync(0xFFFFFFFF, secondMaxVal, offset);
      float reduceSecondMinVal =
          __shfl_down_sync(0xFFFFFFFF, secondMinVal, offset);
      firstMaxVal = fmax(firstMaxVal, fabsf(reduceFirstMaxVal));
      firstMinVal = fmin(firstMinVal, fabsf(reduceFirstMinVal));
      secondMaxVal = fmax(secondMaxVal, fabsf(reduceSecondMaxVal));
      secondMinVal = fmin(secondMinVal, fabsf(reduceSecondMinVal));
    }
    if (lane == 0) {
      shared_absmax_exp_avg = firstMaxVal;
      shared_absmin_exp_avg = firstMinVal;
      shared_absmax_exp_avg_sq = secondMaxVal;
      shared_absmin_exp_avg_sq = secondMinVal;
    }
  }

  __syncthreads();

  if (idx < total_elements) {
    // float fp8MaxVal = fp8_dtype_max<__nv_fp8_e4m3>(exp_avg[idx]);
    // scaling factor before expanding
    float fp8MaxVal = 448;

    // dynamic exponent quantization part
    firstMaxVal = shared_absmax_exp_avg + QUANT_MIN_VAL;
    firstMinVal = shared_absmin_exp_avg + QUANT_MIN_VAL;
    secondMaxVal = shared_absmax_exp_avg_sq + QUANT_MIN_VAL;
    secondMinVal = shared_absmin_exp_avg_sq + QUANT_MIN_VAL;

    // calculate the ratio and make the scale to center
    float firstRatio = firstMaxVal / firstMinVal;
    float secondRatio = secondMaxVal / secondMinVal;
    float firstSqrtMinMax = sqrt(firstMaxVal * firstMinVal);
    float secondSqrtMinMax = sqrt(secondMaxVal * secondMinVal);

    // printf("Max %f, Min %f, Origin %f \n", firstMaxVal, firstMinVal,
    // float_exp_avg);

    // since we use x^k expander, calculate the optimal expanding factor
    float ratioUpperBound = fp8MaxVal * fp8MaxVal / 2;
    float firstExp =
        floor((log2f(ratioUpperBound) / log2f(firstRatio)) * expand_min) /
        expand_min;  // expand_min is set to 8 for example, then the firstExp is
                     // the multiple of 1/8
    float secondExp =
        floor((log2f(ratioUpperBound) / log2f(secondRatio)) * expand_min) /
        expand_min;

    int sign_exp_avg = 1 - 2 * signbit(float_exp_avg);
    float_exp_avg =
        sign_exp_avg * powf(fabsf(float_exp_avg) / firstSqrtMinMax, firstExp);
    float_exp_avg_sq = powf(float_exp_avg_sq / secondSqrtMinMax, secondExp);

    // correspondingly, change the scaling factor
    float new_scale_exp_avg =
        powf(firstMaxVal / firstSqrtMinMax, firstExp) / fp8MaxVal;
    float new_scale_exp_avg_sq =
        powf(secondMaxVal / secondSqrtMinMax, secondExp) / fp8MaxVal;

    // quantize the optimizer states
    __nv_fp8_e4m3 exp_avg_new =
        static_cast<__nv_fp8_e4m3>(float_exp_avg / new_scale_exp_avg);
    __nv_fp8_e4m3 exp_avg_sq_new =
        static_cast<__nv_fp8_e4m3>(float_exp_avg_sq / new_scale_exp_avg_sq);
    // __half exp_avg_new = static_cast<__half>(float_exp_avg /
    // new_scale_exp_avg);
    // __half exp_avg_sq_new = static_cast<__half>(float_exp_avg_sq /
    // new_scale_exp_avg_sq);

    // printf("idx: %d, float: %f, quantize: %f\n", idx, float_exp_avg,
    // (float)exp_avg_new * new_scale_exp_avg);

    // store the output
    exp_avg[idx] = exp_avg_new;
    exp_avg_sq[idx] = exp_avg_sq_new;
    scale_exp_avg[scale_idx] = new_scale_exp_avg;
    scale_exp_avg_sq[scale_idx] = new_scale_exp_avg_sq;
    expand_exp_avg[scale_idx] = firstExp;
    expand_exp_avg_sq[scale_idx] = secondExp;
    sqrtminmax_exp_avg[scale_idx] = firstSqrtMinMax;
    sqrtminmax_exp_avg_sq[scale_idx] = secondSqrtMinMax;
  }
}

void FP8_AdamW_expand_cuda(torch::Tensor params,   // parameter
                           torch::Tensor grads,    // gradient
                           torch::Tensor exp_avg,  // first order momentum
                           torch::Tensor scale_exp_avg,
                           torch::Tensor expand_exp_avg,
                           torch::Tensor sqrtminmax_exp_avg,
                           torch::Tensor exp_avg_sq,  // second order momentum
                           torch::Tensor scale_exp_avg_sq,
                           torch::Tensor expand_exp_avg_sq,
                           torch::Tensor sqrtminmax_exp_avg_sq, float beta1,
                           float beta2, float lr, float wd, float eps, int step,
                           int qgroup_size,
                           int expand_min) {  // other parameters

  // CUDA Blocks
  int total_elements = params.numel();
  int total_scale_elements = scale_exp_avg.numel();
  AT_ASSERTM(qgroup_size == 128,
             "Only Support 128 per-group quantization currently");
  const int block_dim = 128;  // This should equal to the qgroup_size
  int grid_dim = (total_elements + qgroup_size - 1) / block_dim;
  AT_ASSERTM(grid_dim == scale_exp_avg.numel());
  AT_ASSERTM(grid_dim == scale_exp_avg_sq.numel());
  const dim3 blocks(grid_dim);

  // Execution
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kBFloat16, at::kHalf, params.scalar_type(), "fp8_adamw", ([&] {
        fp8_adamw_cuda_expand_kernel<scalar_t><<<blocks, block_dim>>>(
            params.data_ptr<scalar_t>(), grads.data_ptr<scalar_t>(),
            (__nv_fp8_e4m3*)exp_avg.data_ptr<at::Float8_e4m3fn>(),
            scale_exp_avg.data_ptr<float>(), expand_exp_avg.data_ptr<float>(),
            sqrtminmax_exp_avg.data_ptr<float>(),
            (__nv_fp8_e4m3*)exp_avg_sq.data_ptr<at::Float8_e4m3fn>(),
            scale_exp_avg_sq.data_ptr<float>(),
            expand_exp_avg_sq.data_ptr<float>(),
            sqrtminmax_exp_avg_sq.data_ptr<float>(), beta1, beta2, lr, wd, eps,
            step, qgroup_size, expand_min, total_elements,
            total_scale_elements);
      }));
}
