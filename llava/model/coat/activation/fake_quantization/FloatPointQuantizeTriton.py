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

import math
import struct

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


def floatExMy_quantize_triton(x, e_bit, m_bit, stochastic):
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    y = torch.zeros_like(x)

    if x.dtype in [torch.bfloat16, torch.float32]:
        if stochastic:
            noise = x.new(x.shape).uniform_(-0.5, 0.5)
            _floatExMy_stochastic_quantize_kernel[grid](x, noise, y, n_elements, e_bit, m_bit)
        else:
            _floatExMy_quantize_kernel[grid](x, y, n_elements, e_bit, m_bit)
    else:
        raise NotImplementedError(f"Other data format {x.dtype} for float quantization triton")

    return y


@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE': 4,}, num_warps=4),
        triton.Config(
            {
                "BLOCK_SIZE": 1024,
            },
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE": 2048,
            },
            num_stages=1,
        ),
    ],
    key=["n_elements"],
)
@triton.jit
def _floatExMy_quantize_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    e_bit,
    m_bit,
    BLOCK_SIZE: tl.constexpr,
):
    if isinstance(e_bit, tl.constexpr):
        ebit = e_bit.value
    else:
        ebit = e_bit

    if isinstance(m_bit, tl.constexpr):
        mbit = m_bit.value
    else:
        mbit = m_bit

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    x = x.to(tl.float32)
    sign = 1 - 2 * libdevice.signbit(x)
    x_abs = tl.abs(x)
    Elow = -tl.exp2((ebit - 1).to(tl.float32)) + 2
    Ehigh = tl.exp2((ebit - 1).to(tl.float32))
    Mhigh = tl.exp2(mbit.to(tl.float32))
    expo = tl.floor(tl.log2(x_abs))
    expo = tl.clamp(expo, min=Elow, max=Ehigh)
    mant = x_abs / tl.exp2(expo)

    mant_int = tl.floor(mant)
    mant_frac = mant - mant_int
    mant_frac = mant_frac * Mhigh
    # mant_frac = mant_frac + noise
    mant_frac = libdevice.round(mant_frac)

    mant_q = mant_int + mant_frac / Mhigh
    y = sign * tl.exp2(expo) * mant_q
    y = y.to(x_ptr.dtype.element_ty)

    tl.store(output_ptr + offsets, y, mask=mask)


@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE': 4,}, num_warps=4),
        triton.Config(
            {
                "BLOCK_SIZE": 1024,
            },
            num_warps=4,
        ),
        triton.Config(
            {
                "BLOCK_SIZE": 2048,
            },
            num_stages=1,
        ),
    ],
    key=["n_elements"],
)
@triton.jit
def _floatExMy_stochastic_quantize_kernel(
    x_ptr,
    noise_ptr,
    output_ptr,
    n_elements,
    e_bit,
    m_bit,
    BLOCK_SIZE: tl.constexpr,
):
    if isinstance(e_bit, tl.constexpr):
        ebit = e_bit.value
    else:
        ebit = e_bit

    if isinstance(m_bit, tl.constexpr):
        mbit = m_bit.value
    else:
        mbit = m_bit

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    noise = tl.load(noise_ptr + offsets, mask=mask)

    x = x.to(tl.float32)
    sign = 1 - 2 * libdevice.signbit(x)
    x_abs = tl.abs(x)
    Elow = -tl.exp2((ebit - 1).to(tl.float32)) + 2
    Ehigh = tl.exp2((ebit - 1).to(tl.float32))
    Mhigh = tl.exp2(mbit.to(tl.float32))
    expo = tl.floor(tl.log2(x_abs))
    expo = tl.clamp(expo, min=Elow, max=Ehigh)
    mant = x_abs / tl.exp2(expo)

    mant_int = tl.floor(mant)
    mant_frac = mant - mant_int
    mant_frac = mant_frac * Mhigh
    mant_frac = mant_frac + noise
    mant_frac = libdevice.round(mant_frac)

    mant_q = mant_int + mant_frac / Mhigh
    y = sign * tl.exp2(expo) * mant_q
    y = y.to(x_ptr.dtype.element_ty)

    tl.store(output_ptr + offsets, y, mask=mask)


