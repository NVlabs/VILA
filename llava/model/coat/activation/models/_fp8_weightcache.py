import torch.nn as nn

from ..real_quantization import fp8_division_transpose


class FP8CacheWeightModule(nn.Module):
    def __init__(self, config, qargs, layer_id):
        super().__init__()
        self.config = config
        self.qargs = qargs
        self.layer_id = layer_id

    def prepare_weight(self, weight, weight_name, is_first_microbatch):
        if is_first_microbatch:
            if self.qargs.weight_memory_efficient:
                # print(f"{weight_name} uses first microbatch")
                weight_fp8, weight_s, weight_fp8_t = fp8_division_transpose(
                    weight, self.qargs.group_size, self.fwobits["fwbit"]
                )
                setattr(self, f"{weight_name}_fp8_scale", weight_s)
                return weight_fp8, weight_fp8_t, weight_s
            else:
                # print(f"{weight_name} uses first microbatch")
                weight_fp8, weight_s, weight_fp8_t = fp8_division_transpose(
                    weight, self.qargs.group_size, self.fwobits["fwbit"]
                )
                setattr(self, f"{weight_name}_fp8", weight_fp8)
                setattr(self, f"{weight_name}_fp8_t", weight_fp8_t)
                setattr(self, f"{weight_name}_fp8_scale", weight_s)
                return weight_fp8, weight_fp8_t, weight_s
        else:
            if self.qargs.weight_memory_efficient:
                return getattr(self, f"{weight_name}_fp8_scale")
            else:
                return (
                    getattr(self, f"{weight_name}_fp8"),
                    getattr(self, f"{weight_name}_fp8_t"),
                    getattr(self, f"{weight_name}_fp8_scale"),
                )

    def forward(self, x):
        pass


