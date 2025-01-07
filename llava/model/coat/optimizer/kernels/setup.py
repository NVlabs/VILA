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

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="qoptim_cuda",
    ext_modules=[
        CUDAExtension(
            name="qoptim_cuda",
            sources=[
                "fp8_adamw_cuda.cpp",
                "fp8_adamw_cuda_kernel.cu",
                "fp8_adamw_expand_cuda.cpp",
                "fp8_adamw_expand_cuda_kernel.cu",
                "bindings.cpp",
            ],
            # include_dirs=[
            #     'include'
            # ],
            extra_compile_args={
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "-gencode=arch=compute_90,code=compute_90",
                    "-DTORCH_USE_CUDA_DSA",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                ]
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)


