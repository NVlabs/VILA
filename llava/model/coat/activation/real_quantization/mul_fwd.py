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

from ._division_transpose import fp8_division_transpose
from .common import FP8_MAX_VALUE, SCALE_MIN_THRES, get_configs_io_block

"""Element-wise Multiplication Forward"""
"""Input1 (Gate) uses 1 * 16 group quantization"""
"""Input2 (Up) uses 1 * 16 group quantization"""
"""Output uses per-tensor quantization"""
"""The input can be 2D or 3D, but the calculation is performed in 2D"""

fp8_max_value = {
    torch.float8_e4m3fn: 448,
    torch.float8_e5m2: 57344,
}


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
def fp8_mul_forward_kernel(
    output_ptr,
    output_scale_ptr,  # output
    input1_ptr,
    input1_scale_ptr,  # input
    input2_ptr,
    input2_scale_ptr,  # input
    M,
    N,
    SN,
    QB: tl.constexpr,
    fp8_max,  # shape
    input1_stride_0,
    input1_stride_1,  # input1 stride
    s_input1_stride_0,
    s_input1_stride_1,  # scale of input1 stride
    input2_stride_0,
    input2_stride_1,  # input2 stride
    s_input2_stride_0,
    s_input2_stride_1,  # scale of input2 stride
    output_stride_0,
    output_stride_1,  # output stride
    s_output_stride_0,
    s_output_stride_1,  # scale of output stride
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

    # input ptr
    scale_input1_ptr = tl.make_block_ptr(
        base=input1_scale_ptr,
        shape=(M, SN),
        strides=(s_input1_stride_0, s_input1_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_M, BLOCK_SN),
        order=(1, 0),
    )

    input1 = tl.load(input1_block_ptr)
    scale_input1 = tl.load(scale_input1_ptr)

    input1 = input1.to(tl.float32)
    scale_input1 = scale_input1.to(tl.float32)

    # Dequantize and mul calculation
    scale_input1 = tl.reshape(scale_input1, (BLOCK_M, BLOCK_SN, 1))
    input1 = tl.reshape(input1, (BLOCK_M, BLOCK_SN, QB))
    input1 = input1 * scale_input1

    # --- The second input ---
    input2_block_ptr = tl.make_block_ptr(
        base=input2_ptr,
        shape=(M, N),
        strides=(input2_stride_0, input2_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    # input ptr
    scale_input2_ptr = tl.make_block_ptr(
        base=input2_scale_ptr,
        shape=(M, SN),
        strides=(s_input2_stride_0, s_input2_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_M, BLOCK_SN),
        order=(1, 0),
    )

    input2 = tl.load(input2_block_ptr)
    scale_input2 = tl.load(scale_input2_ptr)

    input2 = input2.to(tl.float32)
    scale_input2 = scale_input2.to(tl.float32)

    # Dequantize and mul calculation
    scale_input2 = tl.reshape(scale_input2, (BLOCK_M, BLOCK_SN, 1))
    input2 = tl.reshape(input2, (BLOCK_M, BLOCK_SN, QB))
    input2 = input2 * scale_input2

    # Actual Calculation of SiLU
    mul_output = input1 * input2

    # Quantize Scale calculation
    abs_output = tl.abs(mul_output)
    max_val = tl.max(abs_output, axis=2) + SCALE_MIN_THRES
    scale_output = max_val / fp8_max
    scale_output = tl.reshape(scale_output, (BLOCK_M, BLOCK_SN, 1))

    # Quantize
    # mul_output = tl.fdiv(mul_output, scale_output) # do not quantize the output since it should use per-tensor quantization afterwards
    mul_output = mul_output.to(output_ptr.type.element_ty)
    scale_output = scale_output.to(output_scale_ptr.type.element_ty)
    scale_output = tl.reshape(scale_output, (BLOCK_M, BLOCK_SN))
    mul_output = tl.reshape(mul_output, (BLOCK_M, BLOCK_N))

    # debug
    # mul_output = input
    # scale_output = scale_input

    # pointers
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(M, N),
        strides=(output_stride_0, output_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    scale_output_ptr = tl.make_block_ptr(
        base=output_scale_ptr,
        shape=(M, SN),
        strides=(s_output_stride_0, s_output_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_M, BLOCK_SN),
        order=(1, 0),
    )

    tl.store(output_block_ptr, mul_output, boundary_check=(0, 1))
    tl.store(scale_output_ptr, scale_output, boundary_check=(0, 1))


def fp8_mul_forward(x1, s_x1, x2, s_x2, QB, transpose_output_2d=False):
    # Change batched 3D input to 2D
    batched = False
    if len(x1.shape) == 3:
        assert len(s_x1.shape) == 3
        batched = True
        BS = x1.shape[0]
        x1 = x1.reshape(-1, x1.shape[-1])
        s_x1 = s_x1.reshape(-1, s_x1.shape[-1])
        x2 = x2.reshape(-1, x2.shape[-1])
        s_x2 = s_x2.reshape(-1, s_x2.shape[-1])

    # defining the input and output tensor
    M, N = x1.shape
    _, SN = s_x1.shape  # assume the shape of quantization block size is always 1 * G
    assert x1.shape == x2.shape
    assert s_x1.shape == s_x2.shape

    y = torch.empty_like(x1, dtype=torch.bfloat16)
    s_y = torch.empty_like(s_x1, dtype=s_x1.dtype)
    fp8MaxValue = fp8_max_value[x1.dtype]  # E4M3 and E5M2 have different max value

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    fp8_mul_forward_kernel[grid](
        y,
        s_y,
        x1,
        s_x1,
        x2,
        s_x2,
        M,
        N,
        SN,
        QB,
        fp8MaxValue,
        x1.stride(0),
        x1.stride(1),
        s_x1.stride(0),
        s_x1.stride(1),
        x2.stride(0),
        x2.stride(1),
        s_x2.stride(0),
        s_x2.stride(1),
        y.stride(0),
        y.stride(1),
        s_y.stride(0),
        s_y.stride(1),
        SCALE_MIN_THRES=SCALE_MIN_THRES,
    )

    s_y_max = s_y.max()
    qy, s_y_max, qy_t = fp8_division_transpose(y, QB, x2.dtype, s_y_max)
    qy = qy.to(x2.dtype)
    qy_t = qy_t.to(x2.dtype)

    # Recover 2D to 3D
    if batched:
        y = y.reshape(BS, -1, y.shape[-1])
        qy = qy.reshape(BS, -1, qy.shape[-1])
        if not transpose_output_2d:
            qy_t = qy_t.reshape(BS, -1, qy.shape[-1])

    return qy, s_y_max, qy_t


