import torch

SCALE_MIN_THRES = 1e-10

FP8_MAX_VALUE = {
    torch.float8_e4m3fn: 448,
    torch.float8_e5m2: 57344,
}

convert_str_to_fp8 = {"E4M3": torch.float8_e4m3fn, "E5M2": torch.float8_e5m2}
convert_fp8_to_embit = {
    torch.float8_e4m3fn: (4.0, 3.0),
    torch.float8_e5m2: (5.0, 2.0),
}

# from .common import SCALE_MIN_THRES, FP8_MAX_VALUE
#                     SCALE_MIN_THRES: tl.constexpr,
#  + SCALE_MIN_THRES
# SCALE_MIN_THRES=SCALE_MIN_THRES,


