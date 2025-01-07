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

from copy import deepcopy

import torch
import torch.nn as nn

from ._quantize import fp8_quantize
from ._quantize_pertensor import fp8_quantize_pertensor


class Coat_quantize_bgn(nn.Module):
    def __init__(self, args=None, layer_type=""):
        super().__init__()
        self.args = deepcopy(args)
        self.fp8type = self.args.fabit
        self.layer_type = layer_type

    def forward(self, input):
        if self.training:
            return Coat_quantize_bgn_func.apply(input, self.args.group_size, self.fp8type)
        else:
            return input, None, None


class Coat_quantize_bgn_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, group_size, fp8type):
        """
        (Qoutput, Oscale) uses 1 * 16 quantization
        """
        Qoutput, Oscale = fp8_quantize(input, group_size, fp8type)
        # For autograd
        Qoutput = Qoutput.view(torch.float8_e4m3fn)
        ctx.saved = group_size
        return input, Qoutput, Oscale

    @staticmethod
    def backward(ctx, grad_output, Qgrad_output, Gscale):
        """
        (Qgrad_output, Gscale) uses 1 * 16 quantization
        """
        return grad_output, None, None


class Coat_quantize_end(nn.Module):
    def __init__(self, args=None, layer_type=""):
        super().__init__()
        self.args = deepcopy(args)
        self.fp8type = self.args.babit
        self.layer_type = layer_type

    def forward(self, input, Qinput, Iscale):
        if self.training:
            return Coat_quantize_end_func.apply(input, Qinput, Iscale, self.args.group_size, self.fp8type)
        else:
            return input


class Coat_quantize_end_func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, Qinput, Iscale, group_size, fp8type):
        """
        (Qinput, Iscale) uses 1 * 16 quantization
        """
        ctx.saved = group_size, fp8type

        return input

    @staticmethod
    def backward(ctx, grad_output):
        """
        (Qgrad_output, Gscale) uses per-tensor quantization
        """

        group_size, fp8type = ctx.saved
        Qgrad_output, Gscale, Gscale_g16 = fp8_quantize_pertensor(grad_output, group_size, fp8type, stochastic=False)

        # For autograd
        Qgrad_output = Qgrad_output.view(torch.float8_e4m3fn)

        return grad_output, Qgrad_output, Gscale_g16, None, None


