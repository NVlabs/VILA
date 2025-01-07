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

from ._division_transpose import fp8_division_transpose
from .common import FP8_MAX_VALUE, SCALE_MIN_THRES

"""RMSNorm Forward + Backward"""
"""Forward: Input uses 1 * 16 group quantization"""
"""Forward: Output uses per-tensor quantization"""
"""Backward: Input uses full-precision/BF16."""
"""Backward: Output uses full-precision/BF16."""

"""The input can be 2D or 3D, but the calculation is performed in 2D"""


@triton.heuristics(
    {
        "BLOCK_SN": lambda args: args["N"] // args["QB"],
        "BLOCK_SN2": lambda args: args["N2"] // args["QB"],
    }
)
@triton.jit
def _rms_norm_fwd_fused(
    X,  # pointer to the input
    SX,  # pointer to the scale of input
    Y,  # pointer to the output
    SY,  # pointer to the scale of output
    W,  # Weight
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

    cols = tl.arange(0, N2)
    scale_cols = tl.arange(0, SN2)
    x = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    scale_x = tl.load(SX + scale_cols, mask=scale_cols < SN, other=0.0).to(tl.float32)

    # Dequantize and swish calculation
    scale_x = tl.reshape(scale_x, (BLOCK_SN2, 1))
    x = tl.reshape(x, (BLOCK_SN2, QB))
    x = x * scale_x
    x = tl.reshape(x, N2)

    # Compute variance
    _var = x * x
    var = tl.sum(_var, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)

    # Write mean / rstd
    tl.store(Rstd + row, rstd)

    # Normalize and apply linear transformation
    w = tl.load(W + cols, mask=cols < N, other=0.0)
    x_hat = x * rstd
    y = x_hat * w

    # Scale calculation
    abs_y = tl.abs(y)
    max_val = tl.max(abs_y) + SCALE_MIN_THRES
    scale_output = max_val / fp8_max
    tl.store(SY + row, scale_output)

    # Write output
    tl.store(Y + cols, y, mask=cols < N)


@triton.heuristics(
    {
        "BLOCK_SN": lambda args: args["N"] // args["QB"],
        "BLOCK_SN2": lambda args: args["N2"] // args["QB"],
    }
)
@triton.jit
def _rms_norm_bwd_dx_fused(
    DX,  # pointer to the input gradient
    DY,  # pointer to the output gradient
    DW,
    X,  # pointer to the input
    SX,  # pointer to the input
    W,  # weight
    Rstd,  # pointer to the 1/std
    Lock,
    stride,  # how much to increase the pointer when moving by 1 row
    scale_stride,  # how much to increase the pointer when moving by 1 row
    N: tl.constexpr,  # number of columns in X,
    N2: tl.constexpr,  # number of columns in X,
    SN: tl.constexpr,
    SN2: tl.constexpr,
    QB: tl.constexpr,
    SCALE_MIN_THRES: tl.constexpr,
    BLOCK_SN: tl.constexpr,
    BLOCK_SN2: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
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

    # Offset locks and weights/biases gradient pointer for parallel reduction
    lock_id = row % GROUP_SIZE_M
    Lock += lock_id
    Count = Lock + GROUP_SIZE_M
    DW = DW + lock_id * N + cols

    # Load data to SRAM
    x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
    scale_x = tl.load(SX + scale_cols, mask=scale_cols < SN, other=0.0).to(tl.float32)
    scale_x = tl.reshape(scale_x, (BLOCK_SN2, 1))
    x = tl.reshape(x, (BLOCK_SN2, QB))
    x = x * scale_x
    x = tl.reshape(x, N2)
    dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)

    # Load weight
    w = tl.load(W + cols, mask=cols < N, other=0.0).to(tl.float32)
    rstd = tl.load(Rstd + row).to(tl.float32)

    # Compute dx
    xhat = x * rstd
    wdy = w * dy
    xhat = tl.where(mask, xhat, 0.0)
    wdy = tl.where(mask, wdy, 0.0)
    c1 = tl.sum(xhat * wdy, axis=0) / N
    dx = (wdy - (xhat * c1)) * rstd  # layer norm have c2 term, rmsnorm do not

    dx = dx.to(DX.type.element_ty)

    # Write dx
    tl.store(DX + cols, dx, mask=mask)

    # Accumulate partial sums for dw/db
    partial_dw = (dy * xhat).to(w.dtype)
    while tl.atomic_cas(Lock, 0, 1) == 1:
        pass
    count = tl.load(Count)
    # First store doesn't accumulate
    if count == 0:
        tl.atomic_xchg(Count, 1)
    else:
        partial_dw += tl.load(DW, mask=mask)
    tl.store(DW, partial_dw, mask=mask)
    # Release the lock
    tl.atomic_xchg(Lock, 0)


@triton.jit
def _rms_norm_bwd_dwdb(
    DW,  # pointer to the partial sum of weights gradient
    FINAL_DW,  # pointer to the weights gradient
    M,  # GROUP_SIZE_M
    N,  # number of columns
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Map the program id to the elements of DW and DB it should compute.
    pid = tl.program_id(0)
    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Iterate through the rows of DW and DB to sum the partial sums.
    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        mask = (rows[:, None] < M) & (cols[None, :] < N)
        offs = rows[:, None] * N + cols[None, :]
        dw += tl.load(DW + offs, mask=mask, other=0.0)
    # Write the final sum to the output.
    sum_dw = tl.sum(dw, axis=0)
    tl.store(FINAL_DW + cols, sum_dw, mask=cols < N)


def fp8_rmsnorm_forward(x, s_x, w, QB, eps, transpose_output_2d=False):
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
    y = torch.empty_like(x, dtype=torch.bfloat16)
    s_y = torch.empty((M,), dtype=torch.bfloat16, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    # heuristics for number of warps
    num_warps = 8
    fp8MaxValue = FP8_MAX_VALUE[x.dtype]

    N2 = triton.next_power_of_2(N)
    SN2 = N2 // QB

    # import os
    # if int(os.environ.get("LOCAL_RANK")) == 7:
    #     print(x.device, x.shape, x.dtype, s_x.shape, s_x.dtype, y.shape, y.dtype, s_y.shape, s_y.dtype, w.shape, w.dtype, rstd.shape, rstd.dtype, x.stride(0), s_x.stride(0), N, N2, SN, SN2, QB, "\n")
    # enqueue kernel
    _rms_norm_fwd_fused[(M,)](  #
        x,
        s_x,
        y,
        s_y,
        w,
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

    return qy, s_y_max, qy_t, (w.clone(), rstd, num_warps)


def fp8_rmsnorm_backward(x, s_x, g, w, v, QB, num_warps):
    # Change batched 3D input to 2D
    batched = False
    if len(x.shape) == 3:
        assert len(s_x.shape) == 3
        batched = True
        BS = x.shape[0]
        x = x.reshape(-1, x.shape[-1])
        s_x = s_x.reshape(-1, s_x.shape[-1])
        g = g.reshape(-1, g.shape[-1])

    # enqueue kernel using forward pass heuristics
    # also compute partial sums for DW and DB
    M, N = g.shape
    _, SN = s_x.shape

    GROUP_SIZE_M = 128
    # heuristics for amount of parallel reduction stream for DW/DB
    locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device=w.device)
    _dw = torch.zeros((GROUP_SIZE_M, N), dtype=w.dtype, device=w.device)
    dw = torch.empty((N,), dtype=w.dtype, device=w.device)

    dx = torch.empty_like(g, dtype=torch.bfloat16)

    N2 = triton.next_power_of_2(N)
    SN2 = triton.next_power_of_2(SN)
    _rms_norm_bwd_dx_fused[(M,)](  #
        dx,
        g,
        _dw,
        x,
        s_x,
        w,
        v,
        locks,  #
        x.stride(0),
        s_x.stride(0),
        N,
        N2,
        SN,
        SN2,
        QB,
        SCALE_MIN_THRES=SCALE_MIN_THRES,
        num_warps=num_warps,
        GROUP_SIZE_M=GROUP_SIZE_M,
    )

    if batched:
        dx = dx.reshape(BS, -1, dx.shape[-1])

    grid = lambda meta: [triton.cdiv(N, meta["BLOCK_SIZE_N"])]
    # accumulate partial sums in separate kernel
    _rms_norm_bwd_dwdb[grid](_dw, dw, min(GROUP_SIZE_M, M), N, BLOCK_SIZE_M=32, BLOCK_SIZE_N=128, num_ctas=1)  #  #

    return dx, dw


