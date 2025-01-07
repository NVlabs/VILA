# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

try:
    from ._division import _stochastic_rounding
    from .common import FP8_MAX_VALUE, SCALE_MIN_THRES, convert_fp8_to_embit, convert_str_to_fp8
except:
    from common import SCALE_MIN_THRES, FP8_MAX_VALUE, convert_str_to_fp8, convert_fp8_to_embit
    from COAT.coat.activation.real_quantization._division import _stochastic_rounding

import os
import time

"""Linear Layer Forward + Backward"""
"""Input uses per-tensor quantization"""
"""Output is full-precision/BF16 (for FlashAttention) or 1 * 16 quantization (for the rest)"""
"""The input can be 2D or 3D, but the calculation is performed in 2D"""


def get_configs_io_block():
    configs = []
    for nstages in [3]:
        for block_m in [128, 256]:
            for block_n in [128, 256]:
                for block_k in [128, 256]:
                    for nwarps in [8]:
                        configs.append(
                            triton.Config(
                                {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_K": block_k},
                                num_stages=nstages,
                                num_warps=nwarps,
                            )
                        )
    return configs


# @triton.autotune(
#     configs=get_configs_io_block(),
#     key=["M", "N", "K"],
# )
@triton.jit
def _fp8matmul_kernel(
    A,
    B,
    C,
    noise_ptr,  # noise for stochastic
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,  #
    stride_am,
    stride_ak,  #
    stride_bk,
    stride_bn,  #
    stride_cm,
    stride_cn,  ##
    Scale_A,
    Scale_B,
    Scale_C,
    stride_scm,
    stride_scn,
    output_quantize: tl.constexpr,
    QB: tl.constexpr,  # default to use 1 * 16 quantization
    BIAS,
    fp8_max: tl.constexpr,
    e_bit: tl.constexpr,
    m_bit: tl.constexpr,
    SCALE_MIN_THRES: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # matrix multiplication
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.0)
        b = tl.load(B, mask=rk[:, None] < k_remaining, other=0.0)

        acc = tl.dot(a, b, acc)

        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    scale_a = tl.load(Scale_A)
    scale_b = tl.load(Scale_B)
    scale_ab = scale_a.to(tl.float32) * scale_b.to(tl.float32)
    # fp8 dequantize
    acc = acc * scale_ab

    if BIAS:
        bias = tl.load(BIAS + rbn)
        acc = acc + bias

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]

    if output_quantize:
        acc = tl.reshape(acc, (BLOCK_M, BLOCK_N // QB, QB))
        abs_acc = tl.abs(acc)
        acc_max = tl.max(abs_acc, axis=2) + SCALE_MIN_THRES
        # tl.device_print("acc_max", acc_max)
        acc_scale = acc_max / fp8_max
        # tl.device_print("acc_scale", acc_scale)
        acc_scale = tl.reshape(acc_scale, (BLOCK_M, BLOCK_N // QB, 1))
        acc = tl.fdiv(acc, acc_scale)
        acc = tl.reshape(acc, (BLOCK_M, BLOCK_N))

        if STOCHASTIC:
            noise_block_ptr = noise_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
            noise = tl.load(noise_block_ptr, boundary_check=(0, 1))
            acc = _stochastic_rounding(acc, noise, e_bit, m_bit)

        acc_scale = tl.reshape(acc_scale, (BLOCK_M, BLOCK_N // QB))
        acc_scale = acc_scale.to(Scale_C.type.element_ty)
        acc = acc.to(C.dtype.element_ty)

        rsm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        rsn = pid_n * BLOCK_N // QB + tl.arange(0, BLOCK_N // QB)
        Scale_C = Scale_C + (rsm[:, None] * stride_scm + rsn[None, :] * stride_scn)

        tl.store(C, acc, mask=mask)
        tl.store(Scale_C, acc_scale)

    else:
        # handles write-back with reduction-splitting
        acc = acc.to(C.dtype.element_ty)
        tl.store(C, acc, mask=mask)


def fp8matmul(a, b, output_quantize, scale_a, scale_b, QB, bias=None, stochastic=False):
    # Deal with batched input
    if len(a.shape) == 3:
        BS, batched = a.shape[0], True
        a = a.reshape(-1, a.shape[2])
    else:
        batched = False

    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    fp8MaxValue = FP8_MAX_VALUE[a.dtype]  # E4M3 and E5M2 have different max value
    e_bit, m_bit = convert_fp8_to_embit[a.dtype]

    # Allocates output.
    if output_quantize:
        c = torch.empty((M, N), device=a.device, dtype=a.dtype)
        # c = torch.empty((M, N), device=a.device, dtype=torch.float32)
        scale_c = torch.empty((M, N // QB), device=a.device, dtype=torch.bfloat16)
    else:
        c = torch.empty((M, N), device=a.device, dtype=torch.bfloat16)
        scale_c = torch.empty(
            (1, 1), device=a.device, dtype=torch.bfloat16
        )  # This line is useless, equivalent to scale_c = None

    if stochastic:
        noise = torch.empty_like(c, dtype=torch.float32).uniform_(-0.5, 0.5)
    else:
        noise = None

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    _fp8matmul_kernel[grid](
        a,
        b,
        c,
        noise,  #
        M,
        N,
        K,  #
        a.stride(0),
        a.stride(1),  #
        b.stride(0),
        b.stride(1),  #
        c.stride(0),
        c.stride(1),  #
        scale_a,
        scale_b,
        scale_c,
        scale_c.stride(0),
        scale_c.stride(1),
        output_quantize=output_quantize,
        QB=QB,
        BIAS=bias,
        fp8_max=fp8MaxValue,
        e_bit=e_bit,
        m_bit=m_bit,
        SCALE_MIN_THRES=SCALE_MIN_THRES,
        STOCHASTIC=stochastic,
        BLOCK_M=128,
        BLOCK_N=256,
        BLOCK_K=128,
        GROUP_M=8,
        num_stages=3,
        num_warps=8,
    )
    # Reshape output to batch
    if batched:
        c = c.reshape(BS, -1, N)
        if output_quantize:
            scale_c = scale_c.reshape(BS, -1, N // QB)
            return c, scale_c
    else:
        if output_quantize:
            scale_c = scale_c.reshape(M, N // QB)
            return c, scale_c
    return c


def fp8_linear_forward(x, s, w, s_w, output_quantize, QB, bias=None):
    assert s.numel() == 1, f"X uses per-tensor quantization in linear forward, but the scale shape is {s.shape}"
    assert s_w.numel() == 1, f"W uses per-tensor quantization in linear forward, but the scale shape is {s_w.shape}"

    w_t = w.t()
    return fp8matmul(x, w_t, output_quantize, s, s_w, QB, bias)


# def fp8_linear_forward(x, s, w, s_w, output_quantize, QB):
#     print("you are using the wrong linear function. ")
#     w_t = w.t()
#     if output_quantize:
#         return fp8matmul(x, w_t, True, s, s_w, QB)
#     else:
#         y = fp8matmul(x, w_t, False, s, s_w, QB)

#         return y


def fp8_linear_backward(
    x_t, s, g, s_g, g_t, w_t, s_w, QB, bias=None, stochastic=False, dgrad_quantize=False
):  # dgrad_quantize=True for backward before flashattention
    assert s.numel() == 1, f"X uses per-tensor quantization in linear backward, but the scale shape is {s.shape}"
    assert s_g.numel() == 1, f"G uses per-tensor quantization in linear backward, but the scale shape is {s.shape}"
    assert s_w.numel() == 1, f"W uses per-tensor quantization in linear backward, but the scale shape is {s_w.shape}"

    batched = False
    if len(g.shape) == 3:  # others must be of 2D!
        batched = True
        BS = g.shape[0]
        g = g.reshape(-1, g.shape[-1])

    w_t_t = w_t.t()
    x_t_t = x_t.t()
    if dgrad_quantize:
        y, s_y = fp8matmul(g, w_t_t, True, s_g, s_w, QB, stochastic=stochastic)
    else:
        y = fp8matmul(g, w_t_t, False, s_g, s_w, QB)

    w_g = fp8matmul(g_t, x_t_t, False, s_g, s, QB)

    if batched:
        y = y.reshape(BS, -1, y.shape[-1])
        if dgrad_quantize:
            if s_y.numel() > 1:
                s_y = s_y.reshape(BS, -1, s_y.shape[-1])
    if dgrad_quantize:
        return y, s_y, w_g
    else:
        return y, w_g


