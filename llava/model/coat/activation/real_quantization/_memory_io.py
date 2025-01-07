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

CONST_BLOCK = 32

# The kernel with 1 load operation and 4 store operation
def get_configs_io_block():
    configs = []
    for nstages in [3, 4, 5, 6]:
        for block_m in [32, 64, 128]:
            for block_n in [32, 64, 128]:
                for nwarps in [4, 8, 16, 32]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n},
                            num_stages=nstages,
                            num_warps=nwarps,
                        )
                    )
    return configs


@triton.autotune(
    configs=[] + get_configs_io_block(),
    key=[
        "N",
    ],
)
@triton.jit
def bench_memory_io_kernel_forward(
    output_ptr,
    input_ptr,
    M,
    N,
    B: tl.constexpr,
    input_stride_0,
    input_stride_1,
    output_stride_0,
    output_stride_1,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):

    # Block PID
    pid = tl.program_id(0)
    NUM_BLOCK_M = tl.cdiv(M, BLOCK_M)
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

    output = input * 2

    # pointers
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(M, N),
        strides=(output_stride_0, output_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    output = output.to(output_ptr.type.element_ty)
    tl.store(output_block_ptr, output, boundary_check=(0, 1))


def bench_memory_io_forward(x, B):
    # defining the input and output tensor
    M, N = x.shape

    y = torch.empty_like(x, dtype=x.dtype)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    bench_memory_io_kernel_forward[grid](
        y,
        x,
        M,
        N,
        B,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
    )
    return y


configs = []
for SL in [8192]:
    configs.append(
        triton.testing.Benchmark(  # test different matrix size influence
            x_names=["CDIM"],
            x_vals=[1024, 2048, 4096, 8192],
            line_arg="dtype",
            line_vals=[torch.int8, torch.float16, torch.float32],
            line_names=["float8", "float16", "float32"],
            styles=[("blue", "-"), ("green", "-"), ("red", "-")],
            ylabel="time-cost",
            plot_name=f"INT8GELU<BLSZ={CONST_BLOCK}><SL={SL}>",
            args={"SL": SL, "B": CONST_BLOCK, "provider": "triton", "mode": "time-consuming"},
        )
    )


@triton.testing.perf_report(configs)
def bench_load_store(
    SL, CDIM, B, provider, dtype, mode="forward"
):  # I only use triton as the provider, and mode when benchmarking
    # create data
    x = torch.randn(SL, CDIM, dtype=torch.float32).cuda()
    x = x.to(dtype)

    quantiles = [0.5, 0.2, 0.8]
    # utility functions
    if provider == "triton":

        def y_fwd():
            bench_memory_io_forward(x, B)

    if provider == "torch":
        torch_gelu = torch.nn.GELU()

        def y_fwd():
            return torch_gelu(x)

    # forward pass
    if mode == "time-consuming":
        convert_func = lambda ms: ms
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=100)
    # backward pass
    if mode == "gbps":
        convert_func = lambda ms: 2 * x.numel() * x.element_size() / ms * 1e-6
        ms, min_ms, max_ms = triton.testing.do_bench(y_fwd, quantiles=quantiles, rep=100)
    return convert_func(ms), convert_func(max_ms), convert_func(min_ms)


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(precision=8, linewidth=1600, sci_mode=False, edgeitems=3)
    bench_load_store.run(print_data=True)


