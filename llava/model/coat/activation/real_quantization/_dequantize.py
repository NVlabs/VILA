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

from .common import FP8_MAX_VALUE, SCALE_MIN_THRES, get_configs_io_block

"""Quantize Operator"""
"""Input uses 1 * 16 group quantization"""
"""Output uses 1 * 16 group quantization"""
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
def _fp8_dequantize_kernel(
    output_ptr,  # output
    input_ptr,
    input_scale_ptr,  # input
    M,
    N,
    SN,
    QB: tl.constexpr,  # shape
    input_stride_0,
    input_stride_1,  # input stride
    s_input_stride_0,
    s_input_stride_1,  # scale of output stride
    output_stride_0,
    output_stride_1,  # output stride
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_SN: tl.constexpr,
):  # CUDA block size

    # Block PID
    pid = tl.program_id(0)
    NUM_BLOCK_N = tl.cdiv(N, BLOCK_N)
    pid_dim0 = pid // NUM_BLOCK_N
    pid_dim1 = pid % NUM_BLOCK_N

    # pointers
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(M, N),
        strides=(input_stride_0, input_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    # input ptr
    scale_input_ptr = tl.make_block_ptr(
        base=input_scale_ptr,
        shape=(M, SN),
        strides=(s_input_stride_0, s_input_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_M, BLOCK_SN),
        order=(1, 0),
    )

    input = tl.load(input_block_ptr)
    scale_input = tl.load(scale_input_ptr)

    input = input.to(tl.float32)
    scale_input = scale_input.to(tl.float32)

    # Dequantize and gelu calculation
    scale_input = tl.reshape(scale_input, (BLOCK_M, BLOCK_SN, 1))
    input = tl.reshape(input, (BLOCK_M, BLOCK_SN, QB))
    output = input * scale_input

    output = tl.reshape(output, (BLOCK_M, BLOCK_N))
    output = output.to(output_ptr.dtype.element_ty)

    # debug
    # gelu_output = input
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

    tl.store(output_block_ptr, output, boundary_check=(0, 1))


def fp8_dequantize(x, s_x, QB):
    # Change batched 3D input to 2D
    batched = False
    if len(x.shape) == 3:
        batched = True
        BS = x.shape[0]
        x = x.reshape(-1, x.shape[-1])
        s_x = s_x.reshape(-1, s_x.shape[-1])

    # defining the input and output tensor
    M, N = x.shape
    SN = N // QB

    y = torch.empty_like(x, dtype=torch.bfloat16)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _fp8_dequantize_kernel[grid](
        y,
        x,
        s_x,
        M,
        N,
        SN,
        QB,
        x.stride(0),
        x.stride(1),
        s_x.stride(0),
        s_x.stride(1),
        y.stride(0),
        y.stride(1),
    )

    # Recover 2D to 3D
    if batched:
        y = y.reshape(BS, -1, y.shape[-1])

    return y


