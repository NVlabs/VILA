#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace cg = cooperative_groups;
#define WARPSIZE 32
#define QGROUPSIZE 128
#define QUANT_MIN_VAL 1e-20

template <typename T>
inline float fp8_dtype_max(const T &variable) {
  if (std::is_same<T, __nv_fp8_e4m3>::value) {
    return 448;
  } else if (std::is_same<T, __nv_fp8_e5m2>::value) {
    return 57344;
  } else {
    throw "Unsupported data format";
  }
}

typedef enum { fp8_adamw } myCsrcKernels;

void fp8_adamw_cpu(float *params, float *grads, float *fp_exp_avg,
                   float *fp_exp_avg_sq, float beta1, float beta2, float lr,
                   float wd, float eps, int step, int qgroup_size, int M,
                   int N) {
  for (int idx = 0; idx < M * N; idx++) {
    fp_exp_avg[idx] = beta1 * fp_exp_avg[idx] + (1 - beta1) * grads[idx];
    fp_exp_avg_sq[idx] =
        beta2 * fp_exp_avg_sq[idx] + (1 - beta2) * grads[idx] * grads[idx];

    const float correction1 = 1.0f - powf(beta1, step);
    const float correction2_sqrt = sqrtf(1.0f - powf(beta2, step));

    float denom =
        (sqrtf(fp_exp_avg_sq[idx]) / correction2_sqrt + eps) * correction1;
    float update = (fp_exp_avg[idx] / denom) + (wd * params[idx]);
    params[idx] = params[idx] - (lr * update);
  }
}

template <typename T>
void printFloatArrayToFile(T *array, int M, int N,
                           const std::string &outputFileName) {
  std::ofstream outputFile(outputFileName);
  if (!outputFile.is_open()) {
    std::cout << "Failed to open the file." << std::endl;
    return;
  }

  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      int index = i * N + j;
      outputFile << std::setw(10) << std::right << std::fixed
                 << std::setprecision(6) << (float)array[index] << " ";
      if (j == N - 1) {
        outputFile << "\n";
      }
    }
  }
}

template <typename scalar_t>
__global__ void fp8_adamw_csrc(
    scalar_t *__restrict__ params, scalar_t *__restrict__ grads,
    __nv_fp8_e4m3 *__restrict__ exp_avg, float *__restrict__ scale_exp_avg,
    float *__restrict__ expand_exp_avg, float *__restrict__ sqrtminmax_exp_avg,
    __nv_fp8_e4m3 *__restrict__ exp_avg_sq,
    float *__restrict__ scale_exp_avg_sq, float *__restrict__ expand_exp_avg_sq,
    float *__restrict__ sqrtminmax_exp_avg_sq, float beta1, float beta2,
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

template <myCsrcKernels algo>
void myKernelLauncher(float *params, float *grads, __nv_fp8_e4m3 *exp_avg,
                      float *scale_exp_avg, float *expand_exp_avg,
                      float *sqrtminmax_exp_avg, __nv_fp8_e4m3 *exp_avg_sq,
                      float *scale_exp_avg_sq, float *expand_exp_avg_sq,
                      float *sqrtminmax_exp_avg_sq, float beta1, float beta2,
                      float lr, float wd, float eps, int step, int qgroup_size,
                      int expand_min, int M, int N) {
  if (algo == fp8_adamw) {
    const int block_dim = 128;
    int grid_dim = (M * N + qgroup_size - 1) / block_dim;
    const dim3 gridDim(grid_dim);
    const dim3 blockDim(block_dim);
    printf("Yes!\n");
    fp8_adamw_csrc<float><<<gridDim, blockDim>>>(
        params, grads, exp_avg, scale_exp_avg, expand_exp_avg,
        sqrtminmax_exp_avg, exp_avg_sq, scale_exp_avg_sq, expand_exp_avg_sq,
        sqrtminmax_exp_avg_sq, beta1, beta2, lr, wd, eps, step, qgroup_size,
        expand_min, M * N, int(floor(M * N / 128.)));
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
      std::cout << "CUDA error occurred in kernel launch: "
                << cudaGetErrorString(error) << std::endl;
      return;
    }
    printf("Finish!\n");
  }
}

float testMaxError(void (*myGPUKernel)(float *, float *, __nv_fp8_e4m3 *,
                                       float *, float *, float *,
                                       __nv_fp8_e4m3 *, float *, float *,
                                       float *,  // tensor input
                                       float, float, float, float, float, int,
                                       int, int,   // float and int input
                                       int, int),  // M and N
                   int M, int N) {
  size_t size_param = M * N * sizeof(float);
  size_t size_optim = M * N * sizeof(__nv_fp8_e4m3);
  size_t size_scale = int(ceil(M * N / 128.)) * sizeof(float);

  // host tensor
  float *h_p, *h_g;
  __nv_fp8_e4m3 *h_m, *h_v;
  float *h_sm, *h_sv;
  float *h_fp_m, *h_fp_v;
  float *h_cpd_m, *h_cpd_v;
  float *h_sqrtmm_m, *h_sqrtmm_v;

  // device tensor
  float *d_p, *d_g;
  __nv_fp8_e4m3 *d_m, *d_v;
  float *d_sm, *d_sv;
  float *d_cpd_m, *d_cpd_v;
  float *d_sqrtmm_m, *d_sqrtmm_v;

  // device tensor transfer to host
  float *hd_p, *hd_g;
  __nv_fp8_e4m3 *hd_m, *hd_v;
  float *hd_sm, *hd_sv;
  float *hd_fp_m, *hd_fp_v;
  float *hd_cpd_m, *hd_cpd_v;
  float *hd_sqrtmm_m, *hd_sqrtmm_v;

  h_p = (float *)malloc(size_param);
  h_g = (float *)malloc(size_param);
  h_m = (__nv_fp8_e4m3 *)malloc(size_optim);
  h_v = (__nv_fp8_e4m3 *)malloc(size_optim);
  h_sm = (float *)malloc(size_scale);
  h_sv = (float *)malloc(size_scale);
  h_cpd_m = (float *)malloc(size_scale);
  h_cpd_v = (float *)malloc(size_scale);
  h_sqrtmm_m = (float *)malloc(size_scale);
  h_sqrtmm_v = (float *)malloc(size_scale);
  h_sv = (float *)malloc(size_scale);
  h_fp_m = (float *)malloc(size_param);
  h_fp_v = (float *)malloc(size_param);
  cudaMalloc(&d_p, size_param);
  cudaMalloc(&d_g, size_param);
  cudaMalloc(&d_m, size_optim);
  cudaMalloc(&d_v, size_optim);
  cudaMalloc(&d_sm, size_scale);
  cudaMalloc(&d_sv, size_scale);
  cudaMalloc(&d_cpd_m, size_scale);
  cudaMalloc(&d_cpd_v, size_scale);
  cudaMalloc(&d_sqrtmm_m, size_scale);
  cudaMalloc(&d_sqrtmm_v, size_scale);
  hd_p = (float *)malloc(size_param);
  hd_g = (float *)malloc(size_param);
  hd_m = (__nv_fp8_e4m3 *)malloc(size_optim);
  hd_v = (__nv_fp8_e4m3 *)malloc(size_optim);
  hd_sm = (float *)malloc(size_scale);
  hd_sv = (float *)malloc(size_scale);
  hd_fp_m = (float *)malloc(size_param);
  hd_fp_v = (float *)malloc(size_param);
  hd_cpd_m = (float *)malloc(size_scale);
  hd_cpd_v = (float *)malloc(size_scale);
  hd_sqrtmm_m = (float *)malloc(size_scale);
  hd_sqrtmm_v = (float *)malloc(size_scale);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cout << "CUDA error occurred in data copy: "
              << cudaGetErrorString(error) << std::endl;
    return 0.;
  }

  srand(0);
  // random initialization for CPU tensor
  for (int i = 0; i < M * N; i++) {
    h_p[i] = (float)(rand() / (float(RAND_MAX) / 10));
    h_g[i] = (float)(rand() / (float(RAND_MAX) / 10));
    h_m[i] = (__nv_fp8_e4m3)(rand() / (float(RAND_MAX) / 10));
    h_v[i] = (__nv_fp8_e4m3)(rand() / (float(RAND_MAX) / 10));
  }
  for (int i = 0; i < int(ceilf(M * N / 128.)); i++) {
    h_sm[i] = (float)(rand() / (float(RAND_MAX) / 10));
    h_sv[i] = (float)(rand() / (float(RAND_MAX) / 10));
    h_cpd_m[i] = 2;
    h_cpd_v[i] = 3. / 8.;
    h_sqrtmm_m[i] = (float)(rand() / (float(RAND_MAX) / 10));
    h_sqrtmm_v[i] = (float)(rand() / (float(RAND_MAX) / 10));

    printf("scale is %f\n", h_sm[i]);
  }
  for (int i = 0; i < M * N; i++) {
    h_fp_m[i] = (float)h_m[i] * h_sm[int(floor(i / 128.))];
    h_fp_v[i] = (float)h_v[i] * h_sv[int(floor(i / 128.))];
  }
  float beta1 = 0.9, beta2 = 0.95, lr = 4e-4, wd = 0.1, eps = 1e-8;
  int step = 100, qgroup_size = 128, expand_min = 16;

  printFloatArrayToFile(h_p, M, N, "Past_CPU_param.txt");
  printFloatArrayToFile(h_g, M, N, "Past_CPU_grad.txt");
  printFloatArrayToFile(h_m, M, N, "Past_CPU_m1.txt");
  printFloatArrayToFile(h_sm, 1, int(ceilf(M * N / 128.)), "Past_CPU_ms.txt");
  printFloatArrayToFile(h_fp_m, M, N, "Past_CPU_mf.txt");
  printFloatArrayToFile(h_v, M, N, "Past_CPU_v2.txt");
  printFloatArrayToFile(h_sv, 1, int(ceilf(M * N / 128.)), "Past_CPU_vs.txt");
  printFloatArrayToFile(h_fp_v, M, N, "Past_CPU_vf.txt");

  cudaMemcpy(d_p, h_p, size_param, cudaMemcpyHostToDevice);
  cudaMemcpy(d_g, h_g, size_param, cudaMemcpyHostToDevice);
  cudaMemcpy(d_m, h_m, size_optim, cudaMemcpyHostToDevice);
  cudaMemcpy(d_v, h_v, size_optim, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sm, h_sm, size_scale, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sv, h_sv, size_scale, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cpd_m, h_cpd_m, size_scale, cudaMemcpyHostToDevice);
  cudaMemcpy(d_cpd_v, h_cpd_v, size_scale, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sqrtmm_m, h_sqrtmm_m, size_scale, cudaMemcpyHostToDevice);
  cudaMemcpy(d_sqrtmm_v, h_sqrtmm_v, size_scale, cudaMemcpyHostToDevice);

  fp8_adamw_cpu(h_p, h_g, h_fp_m, h_fp_v, beta1, beta2, lr, wd, eps, step,
                qgroup_size, M, N);

  if (error != cudaSuccess) {
    std::cout << "CUDA error occurred in data initialization: "
              << cudaGetErrorString(error) << std::endl;
    return 0.;
  }

  myGPUKernel(d_p, d_g, d_m, d_sm, d_cpd_m, d_sqrtmm_m, d_v, d_sv, d_cpd_v,
              d_sqrtmm_v, beta1, beta2, lr, wd, eps, step, qgroup_size,
              expand_min, M, N);

  cudaMemcpy(hd_p, d_p, size_param, cudaMemcpyDeviceToHost);
  cudaMemcpy(hd_g, d_g, size_param, cudaMemcpyDeviceToHost);
  cudaMemcpy(hd_m, d_m, size_optim, cudaMemcpyDeviceToHost);
  cudaMemcpy(hd_v, d_v, size_optim, cudaMemcpyDeviceToHost);
  cudaMemcpy(hd_sm, d_sm, size_scale, cudaMemcpyDeviceToHost);
  cudaMemcpy(hd_sv, d_sv, size_scale, cudaMemcpyDeviceToHost);
  cudaMemcpy(hd_cpd_m, d_cpd_m, size_scale, cudaMemcpyDeviceToHost);
  cudaMemcpy(hd_cpd_v, d_cpd_v, size_scale, cudaMemcpyDeviceToHost);
  cudaMemcpy(hd_sqrtmm_m, d_sqrtmm_m, size_scale, cudaMemcpyDeviceToHost);
  cudaMemcpy(hd_sqrtmm_v, d_sqrtmm_v, size_scale, cudaMemcpyDeviceToHost);

  for (int i = 0; i < M * N; i++) {
    hd_fp_m[i] = pow((float)hd_m[i] * hd_sm[int(floor(i / 128.))],
                     1 / hd_cpd_m[int(floor(i / 128.))]) *
                 hd_sqrtmm_m[int(floor(i / 128.))];
    hd_fp_v[i] = pow((float)hd_v[i] * hd_sv[int(floor(i / 128.))],
                     1 / hd_cpd_v[int(floor(i / 128.))]) *
                 hd_sqrtmm_v[int(floor(i / 128.))];
  }
  printFloatArrayToFile(h_p, M, N, "CPU_param.txt");
  printFloatArrayToFile(hd_p, M, N, "GPU_param.txt");
  printFloatArrayToFile(h_g, M, N, "CPU_grad.txt");
  printFloatArrayToFile(hd_g, M, N, "GPU_grad.txt");
  printFloatArrayToFile(h_m, M, N, "CPU_m1.txt");
  printFloatArrayToFile(h_sm, 1, int(ceilf(M * N / 128.)), "CPU_ms.txt");
  printFloatArrayToFile(h_fp_m, M, N, "CPU_mf.txt");
  printFloatArrayToFile(hd_m, M, N, "GPU_m1.txt");
  printFloatArrayToFile(hd_sm, 1, int(ceilf(M * N / 128.)), "GPU_ms.txt");
  printFloatArrayToFile(hd_fp_m, M, N, "GPU_mf.txt");
  printFloatArrayToFile(h_v, M, N, "CPU_v2.txt");
  printFloatArrayToFile(h_sv, 1, int(ceilf(M * N / 128.)), "CPU_vs.txt");
  printFloatArrayToFile(h_fp_v, M, N, "CPU_vf.txt");
  printFloatArrayToFile(hd_v, M, N, "GPU_v2.txt");
  printFloatArrayToFile(hd_sv, 1, int(ceilf(M * N / 128.)), "GPU_vs.txt");
  printFloatArrayToFile(hd_fp_v, M, N, "GPU_vf.txt");

  printFloatArrayToFile(hd_cpd_m, 1, int(ceilf(M * N / 128.)), "GPU_cpd_m.txt");
  printFloatArrayToFile(hd_cpd_v, 1, int(ceilf(M * N / 128.)), "GPU_cpd_v.txt");
  printFloatArrayToFile(hd_sqrtmm_m, 1, int(ceilf(M * N / 128.)),
                        "GPU_sqrtmm_m.txt");
  printFloatArrayToFile(hd_sqrtmm_v, 1, int(ceilf(M * N / 128.)),
                        "GPU_sqrtmm_v.txt");

  return 0.;
}

int main() {
  const int M = 1, N = 7;
  float max_error = testMaxError(myKernelLauncher<fp8_adamw>, M, N);
}
