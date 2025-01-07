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

"""
Layer Normalization
====================
In this tutorial, you will write a high-performance layer normalization
kernel that runs faster than the PyTorch implementation.

In doing so, you will learn about:

* Implementing backward pass in Triton.

* Implementing parallel reduction in Triton.

"""

import torch
import triton
import triton.language as tl

from ._division import _stochastic_rounding
from ._division_transpose import fp8_division_transpose
from .common import FP8_MAX_VALUE, SCALE_MIN_THRES, convert_fp8_to_embit

"""FP8 LayerNorm. Forward + Backward"""
"""Forward: input uses 1 * 16 quantization"""
"""Forward: output use per-tensor quantization."""
"""Backward: input uses full-precision/BF16."""
"""Backward: output uses full-precision/BF16"""
"""Support 3D input, but need to first flatten to 2D to perform calculation."""


@triton.heuristics(
    {
        "BLOCK_SN": lambda args: args["N"] // args["QB"],
        "BLOCK_SN2": lambda args: args["N2"] // args["QB"],
    }
)
@triton.jit
def _layer_norm_fwd_fused(
    X,  # pointer to the input
    SX,  # pointer to the scale of input
    Y,  # pointer to the output
    SY,  # pointer to the scale of output
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    scale_stride,  # how much to increase the pointer when moving by 1 row
    N: tl.constexpr,  # number of columns in X,
    N2: tl.constexpr,  # number of columns in X,
    SN: tl.constexpr,
    SN2: tl.constexpr,
    QB: tl.constexpr,
    eps,  # epsilon to avoid division by zero
    fp8_max,
    SCALE_MIN_THRES: tl.constexpr,
    BLOCK_SN: tl.constexpr,
    BLOCK_SN2: tl.constexpr,
):
    # Map the program id to the row of X and Y it should compute.
    row = tl.program_id(0)
    Y += row * stride
    X += row * stride
    SX += row * scale_stride

    mean = 0
    cols = tl.arange(0, N2)
    scale_cols = tl.arange(0, SN2)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    scale_x = tl.load(SX + scale_cols, mask=scale_cols < SN, other=0.0).to(tl.float32)

    # Dequantize and swish calculation
    scale_x = tl.reshape(scale_x, (BLOCK_SN2, 1))
    x = tl.reshape(x, (BLOCK_SN2, QB))
    x = x * scale_x
    x = tl.reshape(x, (N2,))

    # Compute mean and Variance
    mean = tl.sum(x, axis=0) / N
    # Compute variance
    _var = (x - mean) * (x - mean)
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    # Write mean / rstd
    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)
    # Normalize and apply linear transformation
    x_hat = (x - mean) * rstd

    # Scale calculation
    abs_x_hat = tl.abs(x_hat)
    max_val = tl.max(abs_x_hat, axis=0) + SCALE_MIN_THRES
    scale_output = max_val / fp8_max
    scale_output = scale_output.to(SY.type.element_ty)

    tl.store(SY + row, scale_output)

    # Write output
    tl.store(Y + cols, x_hat, mask=cols < N)


@triton.heuristics(
    {
        "BLOCK_SN": lambda args: args["N"] // args["QB"],
        "BLOCK_SN2": lambda args: args["N2"] // args["QB"],
    }
)
@triton.jit
def _layer_norm_bwd_dx_fused(
    DX,  # pointer to the input gradient
    DY,  # pointer to the output gradient
    X,  # pointer to the input
    SX,  # pointer to the input
    noise_ptr,  # noise for stochastic
    Mean,  # pointer to the mean
    Rstd,  # pointer to the 1/std
    stride,  # how much to increase the pointer when moving by 1 row
    scale_stride,  # how much to increase the pointer when moving by 1 row
    N: tl.constexpr,  # number of columns in X
    N2: tl.constexpr,  # number of columns in X
    SN: tl.constexpr,
    SN2: tl.constexpr,
    QB: tl.constexpr,
    SCALE_MIN_THRES: tl.constexpr,
    STOCHASTIC: tl.constexpr,
    BLOCK_SN: tl.constexpr,
    BLOCK_SN2: tl.constexpr,
):
    # Map the program id to the elements of X, DX, and DY it should compute.
    row = tl.program_id(0)
    cols = tl.arange(0, N2)
    scale_cols = tl.arange(0, SN2)
    mask = cols < N
    X += row * stride
    DY += row * stride
    DX += row * stride
    SX += row * scale_stride
    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    scale_x = tl.load(SX + scale_cols, mask=scale_cols < SN, other=0.0).to(tl.float32)
    scale_x = tl.reshape(scale_x, (BLOCK_SN2, 1))
    x = tl.reshape(x, (BLOCK_SN2, QB))
    x = x * scale_x
    x = tl.reshape(x, N2)

    dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.load(Mean + row)
    rstd = tl.load(Rstd + row)
    # Compute dx
    xhat = (x - mean) * rstd
    xhat = tl.where(mask, xhat, 0.0)
    dy = tl.where(mask, dy, 0.0)
    c1 = tl.sum(xhat * dy, axis=0) / N
    c2 = tl.sum(dy, axis=0) / N
    dx = (dy - (xhat * c1 + c2)) * rstd

    if STOCHASTIC:
        # noise_ptr += row * stride
        # noise_block_ptr = noise_ptr + cols
        # noise = tl.load(noise_block_ptr, mask=mask, other=0.)

        noise_offset = row * stride + cols
        noise = tl.rand(0, noise_offset)

        dx = _stochastic_rounding(dx, noise, e_bit, m_bit)

    dx = dx.to(DX.type.element_ty)

    # Write dx
    tl.store(DX + cols, dx, mask=mask)


def fp8_layernorm_noparam_forward(x, s_x, QB, eps, transpose_output_2d=False):
    # Change batched 3D input to 2D
    batched = False
    if len(x.shape) == 3:
        assert len(s_x.shape) == 3
        batched = True
        BS = x.shape[0]
        x = x.reshape(-1, x.shape[-1])
        s_x = s_x.reshape(-1, s_x.shape[-1])

    # allocate output
    M, N = x.shape
    _, SN = s_x.shape
    y = torch.empty_like(x, dtype=torch.float32)
    s_y = torch.empty(
        (M,), dtype=torch.bfloat16, device=x.device
    )  # We do this because we apply per-tensor quantization for it afterwards
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    # heuristics for number of warps
    num_warps = 8
    fp8MaxValue = FP8_MAX_VALUE[x.dtype]

    N2 = triton.next_power_of_2(N)
    SN2 = N2 // QB
    # enqueue kernel
    _layer_norm_fwd_fused[(M,)](  #
        x,
        s_x,
        y,
        s_y,
        mean,
        rstd,  #
        x.stride(0),
        s_x.stride(0),
        N,
        N2,
        SN,
        SN2,
        QB,
        eps,
        fp8MaxValue,
        SCALE_MIN_THRES=SCALE_MIN_THRES,  #
        num_warps=num_warps,
        num_ctas=1,
    )
    # reduction
    s_y_max = s_y.max()
    qy, s_y_max, qy_t = fp8_division_transpose(y, QB, x.dtype, s_y_max)

    # Recover 2D to 3D
    if batched:
        y = y.reshape(BS, -1, y.shape[-1])
        qy = qy.reshape(BS, -1, y.shape[-1])
        if not transpose_output_2d:
            qy_t = qy_t.reshape(BS, -1, y.shape[-1])

    return qy, s_y_max, qy_t, (mean, rstd, num_warps)


def fp8_layernorm_noparam_backward(x, s_x, g, QB, m, v, num_warps, stochastic=False):
    # Change batched 3D input to 2D
    batched = False
    if len(x.shape) == 3:
        assert len(s_x.shape) == 3
        batched = True
        BS = x.shape[0]
        x = x.reshape(-1, x.shape[-1])
        s_x = s_x.reshape(-1, s_x.shape[-1])
        g = g.reshape(-1, g.shape[-1])

    if stochastic:
        # noise = torch.empty_like(g, dtype=torch.float32).uniform_(-0.5, 0.5)
        noise = None
    else:
        noise = None

    # heuristics for amount of parallel reduction stream for DW/DB
    dx = torch.empty_like(g, dtype=torch.bfloat16)
    # enqueue kernel using forward pass heuristics
    # also compute partial sums for DW and DB
    M, N = g.shape
    _, SN = s_x.shape

    N2 = triton.next_power_of_2(N)
    SN2 = triton.next_power_of_2(SN)
    _layer_norm_bwd_dx_fused[(M,)](  #
        dx,
        g,
        x,
        s_x,
        noise,
        m,
        v,  #
        x.stride(0),
        s_x.stride(0),
        N,
        N2,
        SN,
        SN2,
        QB,
        SCALE_MIN_THRES=SCALE_MIN_THRES,
        STOCHASTIC=stochastic,
        num_warps=num_warps,
    )

    if batched:
        dx = dx.reshape(BS, -1, dx.shape[-1])

    return dx


