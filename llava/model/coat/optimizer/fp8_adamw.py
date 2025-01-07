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

from collections import OrderedDict, defaultdict
from copy import deepcopy
from itertools import chain
from typing import Any, DefaultDict, Dict, Hashable, Iterable, List, Optional, Tuple, Union

import qoptim_cuda
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from typing_extensions import ParamSpec, Self, TypeAlias

StateDict: TypeAlias = Dict[str, Any]

convert_str_to_fp8 = {"E4M3": torch.float8_e4m3fn, "E5M2": torch.float8_e5m2}


class CoatAdamW(Optimizer):
    def __init__(
        self,
        qargs,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
        *,
        fused: Optional[bool] = None,
    ):
        self.qargs = qargs
        assert self.qargs.first_order_expansion == self.qargs.second_order_expansion
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            fused=fused,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            fused = group.setdefault("fused", None)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = torch.tensor(step_val, dtype=torch.float32)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        amsgrad,
        use_expansion,
        exp_avgs,
        scale_exp_avgs,
        expand_exp_avgs,
        sqrt_minmax_exp_avgs,
        exp_avg_sqs,
        scale_exp_avg_sqs,
        expand_exp_avg_sqs,
        sqrt_minmax_exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    ):
        for p in group["params"]:
            if p.grad is None:
                continue
            params_with_grad.append(p)
            if p.grad.is_sparse:
                raise RuntimeError("AdamW does not support sparse gradients")
            grads.append(p.grad)

            state = self.state[p]

            # print(f'Param shape: {p.shape}', file=open('debug.txt', 'a'))
            # print(f'Param shape: {p.shape}, {p.device}')

            # State initialization
            if len(state) == 0:
                # This is because kernel launches are costly on CUDA and XLA.
                state["step"] = torch.tensor(0.0)

                # Should be torch.float8_e4m3fn
                first_order_dtype = convert_str_to_fp8[self.qargs.first_order_bit]
                second_order_dtype = convert_str_to_fp8[self.qargs.second_order_bit]
                scale_shape = (p.numel() + self.qargs.qgroup_size - 1) // self.qargs.qgroup_size

                # Exponential moving average of gradient values
                state["exp_avg"] = torch.zeros_like(p, dtype=first_order_dtype, memory_format=torch.preserve_format)
                state["scale_exp_avg"] = torch.zeros(scale_shape, device=p.device, dtype=p.dtype)
                if use_expansion:
                    state["expand_exp_avg"] = torch.ones(scale_shape, device=p.device, dtype=p.dtype)
                    state["sqrt_minmax_exp_avg"] = torch.ones(scale_shape, device=p.device, dtype=p.dtype)
                # Exponential moving average of squared gradient values
                state["exp_avg_sq"] = torch.zeros_like(p, dtype=second_order_dtype, memory_format=torch.preserve_format)
                state["scale_exp_avg_sq"] = torch.zeros(scale_shape, device=p.device, dtype=p.dtype)
                if use_expansion:
                    state["expand_exp_avg_sq"] = torch.ones(scale_shape, device=p.device, dtype=p.dtype)
                    state["sqrt_minmax_exp_avg_sq"] = torch.ones(scale_shape, device=p.device, dtype=p.dtype)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    state["max_exp_avg_sq"] = torch.zeros(p, memory_format=torch.preserve_format)

            exp_avgs.append(state["exp_avg"])
            scale_exp_avgs.append(state["scale_exp_avg"])
            if use_expansion:
                expand_exp_avgs.append(state["expand_exp_avg"])
                sqrt_minmax_exp_avgs.append(state["sqrt_minmax_exp_avg"])
            exp_avg_sqs.append(state["exp_avg_sq"])
            scale_exp_avg_sqs.append(state["scale_exp_avg_sq"])
            if use_expansion:
                expand_exp_avg_sqs.append(state["expand_exp_avg_sq"])
                sqrt_minmax_exp_avg_sqs.append(state["sqrt_minmax_exp_avg_sq"])

            if group["amsgrad"]:
                max_exp_avg_sqs.append(state["max_exp_avg_sq"])

            state_steps.append(state["step"])

    @torch._disable_dynamo
    def load_state_dict(self, state_dict: StateDict) -> None:
        r"""Loads the optimizer state.

        Args:
            state_dict (dict): optimizer state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # shallow copy, to be consistent with module API
        state_dict = state_dict.copy()

        for pre_hook in self._optimizer_load_state_dict_pre_hooks.values():
            hook_result = pre_hook(self, state_dict)
            if hook_result is not None:
                state_dict = hook_result

        # Validate the state_dict
        groups = self.param_groups

        # Deepcopy as we write into saved_groups later to update state
        saved_groups = deepcopy(state_dict["param_groups"])

        if len(groups) != len(saved_groups):
            raise ValueError("loaded state dict has a different number of " "parameter groups")
        param_lens = (len(g["params"]) for g in groups)
        saved_lens = (len(g["params"]) for g in saved_groups)
        if any(p_len != s_len for p_len, s_len in zip(param_lens, saved_lens)):
            raise ValueError(
                "loaded state dict contains a parameter group " "that doesn't match the size of optimizer's group"
            )

        # Update the state
        id_map = dict(
            zip(
                chain.from_iterable(g["params"] for g in saved_groups), chain.from_iterable(g["params"] for g in groups)
            )
        )

        def _cast(param, value, param_id=None, param_groups=None, key=None):
            r"""Make a deep copy of value, casting all tensors to device of param."""
            if isinstance(value, torch.Tensor):
                return CoatAdamW._process_value_according_to_param_policy(param, value, param_id, param_groups, key)
            elif isinstance(value, dict):
                return {
                    k: _cast(param, v, param_id=param_id, param_groups=param_groups, key=k) for k, v in value.items()
                }
            elif isinstance(value, Iterable):
                return type(value)(_cast(param, v, param_id=param_id, param_groups=param_groups) for v in value)  # type: ignore[call-arg]
            else:
                return value

        # Copy state assigned to params (and cast tensors to appropriate types).
        # State that is not assigned to params is copied as is (needed for
        # backward compatibility).
        state: DefaultDict[torch.Tensor, Dict[Any, Any]] = defaultdict(dict)
        for k, v in state_dict["state"].items():
            if k in id_map:
                param = id_map[k]
                state[param] = _cast(param, v, param_id=k, param_groups=state_dict["param_groups"])
            else:
                state[k] = v

        # Update parameter groups, setting their 'params' value
        def update_group(group: Dict[str, Any], new_group: Dict[str, Any]) -> Dict[str, Any]:
            new_group["params"] = group["params"]
            return new_group

        param_groups = [update_group(g, ng) for g, ng in zip(groups, saved_groups)]
        self.__setstate__({"state": state, "param_groups": param_groups})

        for post_hook in self._optimizer_load_state_dict_post_hooks.values():
            post_hook(self)

    @staticmethod
    def _process_value_according_to_param_policy(
        param: torch.Tensor,
        value: torch.Tensor,
        param_id: int,
        param_groups: List[Dict[Any, Any]],
        key: Hashable = None,
    ) -> torch.Tensor:
        # Floating-point types are a bit special here. They are the only ones
        # that are assumed to always match the type of params.
        # Make sure state['step'] is not casted https://github.com/pytorch/pytorch/issues/74424
        # UNLESS fused or capturable, see note [special device hosting for step]
        fused = False
        capturable = False
        assert param_groups is not None
        for pg in param_groups:
            if param_id in pg["params"]:
                fused = pg["fused"] if "fused" in pg else False
                capturable = pg["capturable"] if "capturable" in pg else False
                break
        if key == "step":
            if capturable or fused:
                return value.to(dtype=torch.float32, device=param.device)
            else:
                return value
        else:
            assert value.dtype in [torch.float8_e4m3fn, torch.float8_e5m2, torch.float32]
            return value.to(device=param.device)  # do not cast optimizer states
            # if param.is_floating_point():
            #     return value.to(dtype=param.dtype, device=param.device)
            # else:
            #     return value.to(device=param.device)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            scale_exp_avgs = []
            expand_exp_avgs = []
            sqrt_minmax_exp_avgs = []
            exp_avg_sqs = []
            scale_exp_avg_sqs = []
            expand_exp_avg_sqs = []
            sqrt_minmax_exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group["amsgrad"]
            use_expansion = self.qargs.first_order_expansion in ["expansion", "true"]
            beta1, beta2 = group["betas"]

            self._init_group(
                group,
                params_with_grad,
                grads,
                amsgrad,
                use_expansion,
                exp_avgs,
                scale_exp_avgs,
                expand_exp_avgs,
                sqrt_minmax_exp_avgs,
                exp_avg_sqs,
                scale_exp_avg_sqs,
                expand_exp_avg_sqs,
                sqrt_minmax_exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            Coatadamw(
                self.qargs,
                params_with_grad,
                grads,
                exp_avgs,
                scale_exp_avgs,
                expand_exp_avgs,
                sqrt_minmax_exp_avgs,
                exp_avg_sqs,
                scale_exp_avg_sqs,
                expand_exp_avg_sqs,
                sqrt_minmax_exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
                use_expansion=use_expansion,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                qgroup_size=self.qargs.qgroup_size,
                expand_min=self.qargs.expand_min,
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        return loss


def Coatadamw(
    qargs,
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    scale_exp_avgs: List[Tensor],
    expand_exp_avgs: List[Tensor],
    sqrt_minmax_exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    scale_exp_avg_sqs: List[Tensor],
    expand_exp_avg_sqs: List[Tensor],
    sqrt_minmax_exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    *,
    amsgrad: bool,
    use_expansion: bool,
    beta1: float,
    beta2: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    qgroup_size: int,
    expand_min: int,
):
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """
    if not torch._utils.is_compiling() and not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError("API has changed, `state_steps` argument must contain a list of singleton tensors")

    func = _single_tensor_Coatadamw

    func(
        qargs,
        params,
        grads,
        exp_avgs,
        scale_exp_avgs,
        expand_exp_avgs,
        sqrt_minmax_exp_avgs,
        exp_avg_sqs,
        scale_exp_avg_sqs,
        expand_exp_avg_sqs,
        sqrt_minmax_exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        use_expansion=use_expansion,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        qgroup_size=qgroup_size,
        expand_min=expand_min,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


def _dispatch_sqrt(x: float):  # float annotation is needed because of torchscript type inference
    if not torch.jit.is_scripting() and isinstance(x, torch.Tensor):
        return x.sqrt()
    else:
        return sqrt(x)


def _single_tensor_Coatadamw(
    qargs,
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    scale_exp_avgs: List[Tensor],
    expand_exp_avgs: List[Tensor],
    sqrt_minmax_exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    scale_exp_avg_sqs: List[Tensor],
    expand_exp_avg_sqs: List[Tensor],
    sqrt_minmax_exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    use_expansion: bool,
    beta1: float,
    beta2: float,
    lr: Union[Tensor, float],
    weight_decay: float,
    eps: float,
    qgroup_size: int,
    expand_min: int,
):

    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)

    for i, param in enumerate(params):
        grad = grads[i]
        # First order
        exp_avg = exp_avgs[i]
        scale_exp_avg = scale_exp_avgs[i]
        # Second order
        exp_avg_sq = exp_avg_sqs[i]
        scale_exp_avg_sq = scale_exp_avg_sqs[i]
        step_t = state_steps[i]

        # print(len(exp_avg.unique()), len(exp_avg_sq.unique()))
        # print(f"{param.shape}, {grad.shape}, {exp_avg.shape}, {exp_avg_sq.shape}", file=open('debug.txt', 'a'))

        # update step
        step_t += 1
        step = int(step_t.item())

        # Perform Optimizer Step
        if use_expansion:
            expand_exp_avg = expand_exp_avgs[i]
            sqrt_minmax_exp_avg = sqrt_minmax_exp_avgs[i]
            expand_exp_avg_sq = expand_exp_avg_sqs[i]
            sqrt_minmax_exp_avg_sq = sqrt_minmax_exp_avg_sqs[i]

            qoptim_cuda.fp8_adamw_expand_step(
                param,
                grad,
                exp_avg,
                scale_exp_avg,
                expand_exp_avg,
                sqrt_minmax_exp_avg,
                exp_avg_sq,
                scale_exp_avg_sq,
                expand_exp_avg_sq,
                sqrt_minmax_exp_avg_sq,
                beta1,
                beta2,
                lr,
                weight_decay,
                eps,
                step,
                qgroup_size,
                expand_min,
            )

        else:
            qoptim_cuda.fp8_adamw_step(
                param,
                grad,
                exp_avg,
                scale_exp_avg,
                exp_avg_sq,
                scale_exp_avg_sq,
                beta1,
                beta2,
                lr,
                weight_decay,
                eps,
                step,
                qgroup_size,
            )


