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
from .common import FP8_MAX_VALUE, SCALE_MIN_THRES, convert_fp8_to_embit, convert_str_to_fp8, get_configs_io_block

"""Element-wise Multiplication Backward"""
"""Input1 (Gate) uses 1 * 16 group quantization"""
"""Input2 (Up) uses 1 * 16 group quantization"""
"""Grad (Down) uses full-precision/BF16"""
"""Output1 (Gate) uses full-precision/BF16"""
"""Output2 (Up) uses per-tensor quantization, we can choose whether it should be quantized inside this function"""
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
def _fp8_mul_backward_kernel(
    output1_ptr,  # output
    output2_ptr,
    output2_scale_ptr,  # output
    input1_ptr,
    input1_scale_ptr,  # input
    input2_ptr,
    input2_scale_ptr,  # input
    grad_ptr,  # input
    M,
    N,
    SN,
    QB: tl.constexpr,
    fp8_max,
    e_bit,
    m_bit,  # shape
    input1_stride_0,
    input1_stride_1,  # input1 stride
    s_input1_stride_0,
    s_input1_stride_1,  # scale of input1 stride
    input2_stride_0,
    input2_stride_1,  # input2 stride
    s_input2_stride_0,
    s_input2_stride_1,  # scale of input2 stride
    grad_stride_0,
    grad_stride_1,  # input stride
    output1_stride_0,
    output1_stride_1,  # output stride
    output2_stride_0,
    output2_stride_1,  # output stride
    s_output2_stride_0,
    s_output2_stride_1,  # scale of output stride
    SCALE_MIN_THRES: tl.constexpr,
    STOCHASTIC: tl.constexpr,
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

    # pointers of gradient
    grad_block_ptr = tl.make_block_ptr(
        base=grad_ptr,
        shape=(M, N),
        strides=(grad_stride_0, grad_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    grad = tl.load(grad_block_ptr)
    grad = grad.to(tl.float32)
    grad = tl.reshape(grad, (BLOCK_M, BLOCK_SN, QB))

    # Actual Calculation of Mul Backward
    grad1 = grad * input2
    grad1 = tl.reshape(grad1, (BLOCK_M, BLOCK_N))
    grad1 = grad1.to(output1_ptr.type.element_ty)

    # pointers
    output1_block_ptr = tl.make_block_ptr(
        base=output1_ptr,
        shape=(M, N),
        strides=(output1_stride_0, output1_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    tl.store(output1_block_ptr, grad1, boundary_check=(0, 1))

    # Actual Calculation of Mul Backward
    grad2 = grad * input1
    # Quantize the grad 1 - Scale calculation
    abs_grad2 = tl.abs(grad2)
    max_val = tl.max(abs_grad2, axis=2) + SCALE_MIN_THRES
    scale_grad2 = max_val / fp8_max
    scale_grad2 = tl.reshape(scale_grad2, (BLOCK_M, BLOCK_SN, 1))
    # Quantize
    # grad1 = tl.fdiv(grad1, scale_output) # do not quantize the output due to the data flow
    grad2 = grad2.to(output2_ptr.type.element_ty)
    scale_grad2 = scale_grad2.to(output2_scale_ptr.type.element_ty)
    scale_grad2 = tl.reshape(scale_grad2, (BLOCK_M, BLOCK_SN))
    grad2 = tl.reshape(grad2, (BLOCK_M, BLOCK_N))

    # pointers
    output2_block_ptr = tl.make_block_ptr(
        base=output2_ptr,
        shape=(M, N),
        strides=(output2_stride_0, output2_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    scale_output2_ptr = tl.make_block_ptr(
        base=output2_scale_ptr,
        shape=(M, SN),
        strides=(s_output2_stride_0, s_output2_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_M, BLOCK_SN),
        order=(1, 0),
    )
    tl.store(output2_block_ptr, grad2, boundary_check=(0, 1))
    tl.store(scale_output2_ptr, scale_grad2, boundary_check=(0, 1))


def fp8_mul_backward(
    x1, s_x1, x2, s_x2, g, QB, fp8type, stochastic=False, output_quantized_transpose=False
):  # Stochastic Rounding is left outside this function
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
        g = g.reshape(-1, g.shape[-1])

    if stochastic:
        noise = torch.empty_like(g, dtype=torch.float32).uniform_(-0.5, 0.5)
    else:
        noise = None

    # defining the input and output tensor
    M, N = x1.shape
    _, SN = s_x1.shape  # assume the shape of quantization block size is always 1 * G
    assert x1.shape == x2.shape
    assert s_x1.shape == s_x2.shape

    if isinstance(fp8type, str):
        fp8type = convert_str_to_fp8[fp8type]
    y1 = torch.empty_like(g, dtype=torch.bfloat16)
    y2 = torch.empty_like(g, dtype=torch.bfloat16)
    s_y2 = torch.empty_like(s_x1, dtype=torch.bfloat16)
    fp8MaxValue = FP8_MAX_VALUE[fp8type]  # E4M3 and E5M2 have different max value
    e_bit, m_bit = convert_fp8_to_embit[fp8type]

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _fp8_mul_backward_kernel[grid](
        y1,
        y2,
        s_y2,
        x1,
        s_x1,
        x2,
        s_x2,
        g,
        M,
        N,
        SN,
        QB,
        fp8MaxValue,
        e_bit,
        m_bit,
        x1.stride(0),
        x1.stride(1),
        s_x1.stride(0),
        s_x1.stride(1),
        x2.stride(0),
        x2.stride(1),
        s_x2.stride(0),
        s_x2.stride(1),
        g.stride(0),
        g.stride(1),
        y1.stride(0),
        y1.stride(1),
        y2.stride(0),
        y2.stride(1),
        s_y2.stride(0),
        s_y2.stride(1),
        SCALE_MIN_THRES=SCALE_MIN_THRES,
        STOCHASTIC=stochastic,
    )

    if not output_quantized_transpose:
        # Recover 2D to 3D
        if batched:
            y1 = y1.reshape(BS, -1, y1.shape[-1])

            y2 = y2.reshape(BS, -1, y2.shape[-1])
            s_y2 = s_y2.reshape(BS, -1, s_y2.shape[-1])

        return y1, (y2, s_y2)
    else:
        # Per-tensor quantization
        s_y2_max = s_y2.max()
        qy2, s_y2_max, qy2_t = fp8_division_transpose(y2, QB, fp8type, s_y2_max)

        # Recover 2D to 3D
        if batched:
            y1 = y1.reshape(BS, -1, y1.shape[-1])

            y2 = y2.reshape(BS, -1, y2.shape[-1])
            qy2 = qy2.reshape(BS, -1, qy2.shape[-1])
            s_y2 = s_y2.reshape(BS, -1, s_y2.shape[-1])

        return y1, (qy2, s_y2_max, qy2_t)


