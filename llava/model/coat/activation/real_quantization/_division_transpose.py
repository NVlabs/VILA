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

from ._division import _stochastic_rounding
from .common import FP8_MAX_VALUE, SCALE_MIN_THRES, convert_fp8_to_embit, convert_str_to_fp8, get_configs_io_block

"""Division and Transpose Operator"""
"""Input uses full-precision/BF16"""
"""Output uses per tensor quantization"""
"""Output_t uses per tensor quantization and is transposed, but is flattened to 2D"""
"""The input can be 2D or 3D, but the calculation is performed in 2D"""


@triton.autotune(
    configs=[] + get_configs_io_block(),  # triton.Config({'BLOCK_M': 1, 'BLOCK_N': 16}, num_stages=4, num_warps=1,)
    # configs=[triton.Config({'BLOCK_M': 1, 'BLOCK_N': 16}, num_stages=4, num_warps=1,)], #
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
def _fp8_division_transpose_kernel(
    output_ptr,
    output_t_ptr,  # output
    input_ptr,
    input_scale_ptr,  # input
    noise_ptr,  # noise for stochastic
    M,
    N,
    SN,
    QB: tl.constexpr,
    fp8_max,
    e_bit,
    m_bit,  # shape
    input_stride_0,
    input_stride_1,  # input stride
    output_stride_0,
    output_stride_1,  # output stride
    output_t_stride_0,
    output_t_stride_1,  # output stride
    SCALE_MIN_THRES: tl.constexpr,  # We do not use it since we believe SCALE_MIN_THRES should be used in previous kernel when calculating scaling factor
    STOCHASTIC: tl.constexpr,
    ONLY_TRANSPOSED: tl.constexpr,
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

    input = tl.load(input_block_ptr)
    input = input.to(tl.float32)
    scale_output = tl.load(input_scale_ptr)
    scale_output = scale_output.to(tl.float32)

    output = tl.reshape(input, (BLOCK_M, BLOCK_SN, QB))

    # Quantize Scale calculation
    # Quantize
    output = tl.fdiv(output, scale_output)
    output = tl.reshape(output, (BLOCK_M, BLOCK_N))

    if STOCHASTIC:
        # noise_block_ptr = tl.make_block_ptr(
        #     base=noise_ptr,
        #     shape=(M, N),
        #     strides=(input_stride_0, input_stride_1),
        #     offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        #     block_shape=(BLOCK_M, BLOCK_N),
        #     order=(1, 0)
        # )
        # noise = tl.load(noise_block_ptr)

        offs_m = pid_dim0 * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_dim1 * BLOCK_N + tl.arange(0, BLOCK_N)
        noise_offset = offs_m[:, None] * input_stride_0 + offs_n[None, :] * input_stride_1
        noise = tl.rand(0, noise_offset)

        output = _stochastic_rounding(output, noise, e_bit, m_bit)

    output = output.to(output_ptr.type.element_ty)
    # tl.device_print("3: ", output)
    output_t = tl.trans(output)

    # pointers
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(M, N),
        strides=(output_stride_0, output_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )
    output_t_block_ptr = tl.make_block_ptr(
        base=output_t_ptr,
        shape=(N, M),
        strides=(output_t_stride_0, output_t_stride_1),
        offsets=(pid_dim1 * BLOCK_N, pid_dim0 * BLOCK_M),
        block_shape=(BLOCK_N, BLOCK_M),
        order=(1, 0),
    )
    if not ONLY_TRANSPOSED:
        tl.store(output_block_ptr, output, boundary_check=(0, 1))
    tl.store(output_t_block_ptr, output_t, boundary_check=(0, 1))


def fp8_division_transpose(x, QB, fp8type, s_y=None, stochastic=False, only_transposed=False):
    # Change batched 3D input to 2D
    batched = False
    if len(x.shape) == 3:
        batched = True
        BS = x.shape[0]
        x = x.reshape(-1, x.shape[-1])

    if stochastic:
        # noise = torch.empty_like(x, dtype=torch.float32).uniform_(-0.5, 0.5)
        noise = None
    else:
        noise = None

    # defining the input and output tensor
    M, N = x.shape
    SN = N // QB

    if isinstance(fp8type, str):
        fp8type = convert_str_to_fp8[fp8type]

    y = torch.empty_like(x, dtype=fp8type)
    y_t = torch.empty((N, M), dtype=fp8type, device=x.device)
    fp8MaxValue = FP8_MAX_VALUE[fp8type]  # E4M3 and E5M2 have different max value
    e_bit, m_bit = convert_fp8_to_embit[fp8type]

    if s_y is None:
        # print("Warning: do not specify s_y in fp8_division_transpose")
        s_y = (x.abs().max() + SCALE_MIN_THRES) / fp8MaxValue

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _fp8_division_transpose_kernel[grid](
        y,
        y_t,
        x,
        s_y,
        noise,
        M,
        N,
        SN,
        QB,
        fp8MaxValue,
        e_bit,
        m_bit,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        y_t.stride(0),
        y_t.stride(1),
        SCALE_MIN_THRES=SCALE_MIN_THRES,
        STOCHASTIC=stochastic,
        ONLY_TRANSPOSED=only_transposed,
    )

    if not only_transposed:
        # Recover 2D to 3D
        if batched:
            y = y.reshape(BS, -1, y.shape[-1])

        return y, s_y, y_t  # y_t is expected to be 2D tensor
    else:
        return y_t, s_y


