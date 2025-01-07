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

# 4 block
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from ._division import fp8_division
from .common import FP8_MAX_VALUE, SCALE_MIN_THRES, convert_str_to_fp8, get_configs_io_block

"""Element-wise Add, useful for backward"""
"""Input1 (Residual) uses full-precision/BF16"""
"""Input2 (Backbone) uses full-precision/BF16"""
"""Output1 uses full-precision/BF16"""
"""Output2 uses per-tensor quantization"""
"""The input can be 2D or 3D, but the calculation is performed in 2D"""


@triton.autotune(
    configs=[] + get_configs_io_block(),
    key=[
        "N",
    ],
)
@triton.heuristics(
    {
        "BLOCK_SN": lambda args: args["BLOCK_N"] // args["QB"],
    }
)
@triton.jit
def _fp8_add_Ifp_Ifp_Ofp_Opt_kernel(
    output1_ptr,  # output
    output2_scale_ptr,
    input1_ptr,  # input
    input2_ptr,  # input
    M,
    N,
    SN,
    QB: tl.constexpr,
    fp8_max,  # shape
    input1_stride_0,
    input1_stride_1,  # input1 stride
    input2_stride_0,
    input2_stride_1,  # input2 stride
    output1_stride_0,
    output1_stride_1,  # output stride
    s_output2_stride_0,
    s_output2_stride_1,  # scale of output stride
    SCALE_MIN_THRES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SN: tl.constexpr,
):  # CUDA block size

    # Block PID
    pid = tl.program_id(0)
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    pid_dim0 = pid // NUM_BLOCK_N
    pid_dim1 = pid % NUM_BLOCK_N

    # --- The first input ---
    input1_block_ptr = tl.make_block_ptr(
        base=input1_ptr,
        shape=(M, N),
        strides=(input1_stride_0, input1_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    input1 = tl.load(input1_block_ptr)
    input1 = input1.to(tl.float32)
    input1 = tl.reshape(input1, (BLOCK_M, BLOCK_SN, QB))

    # --- The second input ---
    input2_block_ptr = tl.make_block_ptr(
        base=input2_ptr,
        shape=(M, N),
        strides=(input2_stride_0, input2_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    input2 = tl.load(input2_block_ptr)
    input2 = input2.to(tl.float32)
    input2 = tl.reshape(input2, (BLOCK_M, BLOCK_SN, QB))

    # Actual Calculation of Add
    add_output = input1 + input2

    # Quantize the grad 1 - Scale calculation
    abs_add_output = tl.abs(add_output)
    max_val = tl.max(abs_add_output, axis=2) + SCALE_MIN_THRES
    scale_output2 = max_val / fp8_max
    scale_output2 = tl.reshape(scale_output2, (BLOCK_M, BLOCK_SN, 1))

    # save the fp add output
    fp_add_output = add_output.to(output1_ptr.type.element_ty)
    fp_add_output = tl.reshape(fp_add_output, (BLOCK_M, BLOCK_N))

    # pointers
    output1_block_ptr = tl.make_block_ptr(
        base=output1_ptr,
        shape=(M, N),
        strides=(output1_stride_0, output1_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    tl.store(output1_block_ptr, fp_add_output, boundary_check=(0, 1))

    # Quantize
    scale_output2 = scale_output2.to(output2_scale_ptr.type.element_ty)
    scale_output2 = tl.reshape(scale_output2, (BLOCK_M, BLOCK_SN))

    # pointers
    scale_output2_ptr = tl.make_block_ptr(
        base=output2_scale_ptr,
        shape=(M, SN),
        strides=(s_output2_stride_0, s_output2_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_M, BLOCK_SN),
        order=(1, 0),
    )
    tl.store(scale_output2_ptr, scale_output2, boundary_check=(0, 1))


def fp8_add_Ifp_Ifp_Ofp_Opt(x1, x2, QB, fp8type, stochastic=False):  # suppose x1 is full precision or BF16
    # Change batched 3D input to 2D
    batched = False
    if len(x1.shape) == 3:
        assert len(x2.shape) == 3
        batched = True
        BS = x1.shape[0]
        x1 = x1.reshape(-1, x1.shape[-1])
        x2 = x2.reshape(-1, x2.shape[-1])

    # defining the input and output tensor
    M, N = x1.shape
    SN = N // QB
    assert x1.shape == x2.shape

    if isinstance(fp8type, str):
        fp8type = convert_str_to_fp8[fp8type]
    y1 = torch.empty_like(x1, dtype=torch.bfloat16)
    s_y2 = torch.empty((M, SN), dtype=torch.bfloat16, device=x2.device)
    fp8MaxValue = FP8_MAX_VALUE[fp8type]  # E4M3 and E5M2 have different max value

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _fp8_add_Ifp_Ifp_Ofp_Opt_kernel[grid](
        y1,
        s_y2,
        x1,
        x2,
        M,
        N,
        SN,
        QB,
        fp8MaxValue,
        x1.stride(0),
        x1.stride(1),
        x2.stride(0),
        x2.stride(1),
        y1.stride(0),
        y1.stride(1),
        s_y2.stride(0),
        s_y2.stride(1),
        SCALE_MIN_THRES=SCALE_MIN_THRES,
    )

    s_y2_max = s_y2.max()
    qy2, s_y2_max = fp8_division(y1, QB, fp8type, s_y2_max, stochastic=stochastic)  # reuse the floating point output y1

    # Recover 2D to 3D
    if batched:
        y1 = y1.reshape(BS, -1, y1.shape[-1])
        qy2 = qy2.reshape(BS, -1, qy2.shape[-1])
        s_y2 = s_y2.reshape(BS, -1, s_y2.shape[-1])

    return y1, (qy2, s_y2_max, s_y2)


