import torch

# 4 block
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

try:
    from .common import FP8_MAX_VALUE, SCALE_MIN_THRES, convert_fp8_to_embit, convert_str_to_fp8
except:
    from common import FP8_MAX_VALUE, SCALE_MIN_THRES, convert_fp8_to_embit, convert_str_to_fp8

"""Quantize Operator"""
"""Input uses 1 * 16 group quantization"""
"""Output uses full-precision/BF16"""
"""The input can be 2D or 3D, but the calculation is performed in 2D"""

# The kernel with 1 load operation and 4 store operation
def get_configs_io_block():
    configs = []
    for nstages in [3, 4, 5]:
        for block_m in [32, 64, 128]:
            for block_n in [32, 64, 128]:
                for nwarps in [4, 8, 16]:
                    if block_m == 64 and block_n == 64:
                        continue
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
@triton.heuristics(
    {
        "BLOCK_SN": lambda args: args["BLOCK_N"] // args["QB"],
    }
)
@triton.jit
def _fp8_division_kernel(
    output_ptr,  # output
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
    SCALE_MIN_THRES: tl.constexpr,  # We do not use it since we believe SCALE_MIN_THRES should be used in previous kernel when calculating scaling factor
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

    # pointers
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(M, N),
        strides=(input_stride_0, input_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N),
        order=(1, 0),
    )

    input = tl.load(input_block_ptr, boundary_check=(0, 1))
    input = input.to(tl.float32)
    scale_output = tl.load(input_scale_ptr)
    scale_output = scale_output.to(tl.float32)

    output = tl.reshape(input, (BLOCK_M, BLOCK_SN, QB))

    # Quantize Scale calculation
    # Quantize
    output = tl.div_rn(output, scale_output)
    output = tl.reshape(output, (BLOCK_M, BLOCK_N))

    if STOCHASTIC:
        noise_block_ptr = tl.make_block_ptr(
            base=noise_ptr,
            shape=(M, N),
            strides=(input_stride_0, input_stride_1),
            offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_N),
            block_shape=(BLOCK_M, BLOCK_N),
            order=(1, 0),
        )
        noise = tl.load(noise_block_ptr, boundary_check=(0, 1))
        output = _stochastic_rounding(output, noise, e_bit, m_bit)

    output = output.to(output_ptr.type.element_ty)

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


@triton.jit
def _stochastic_rounding(output, noise, e_bit, m_bit):
    subnormal_min = tl.exp2(2 - tl.exp2(e_bit - 1) - m_bit)
    # subnormal_should_be = tl.exp2(2 - tl.exp2(e_bit) - 1)

    output_int32 = tl.cast(output, tl.int32, bitcast=True)
    output_int32 = output_int32 & 0x7F800000
    output_float32 = tl.cast(output_int32, tl.float32, bitcast=True)
    output_exp = tl.maximum(output_float32, subnormal_min)

    noise_rescale = tl.exp2(m_bit) + (output_exp == subnormal_min) * (
        1 - tl.exp2(m_bit)
    )  # 2^m_bit for normal, 1 for subnormal

    noise = output_exp * noise / noise_rescale
    sign = 1 - 2 * libdevice.signbit(output)
    output = tl.abs(output) + noise

    # tl.device_print("out", output)
    # tl.device_print("noise", noise)

    minmax_ratio = 2 + (output_exp == subnormal_min) * (tl.exp2(m_bit) - 2)  # 2 for normal, and 2^M for subnormal
    output = sign * tl.clamp(output, min=output_exp, max=minmax_ratio * output_exp)

    return output


def fp8_division(x, QB, fp8type, s_y=None, stochastic=False):
    # Change batched 3D input to 2D
    batched = False
    if len(x.shape) == 3:
        batched = True
        BS = x.shape[0]
        x = x.reshape(-1, x.shape[-1])

    if stochastic:
        noise = torch.empty_like(x, dtype=torch.float32).uniform_(-0.5, 0.5)
    else:
        noise = None

    # defining the input and output tensor
    M, N = x.shape
    SN = N // QB

    if isinstance(fp8type, str):
        fp8type = convert_str_to_fp8[fp8type]

    y = torch.empty_like(x, dtype=fp8type)
    fp8MaxValue = FP8_MAX_VALUE[fp8type]  # E4M3 and E5M2 have different max value
    e_bit, m_bit = convert_fp8_to_embit[fp8type]

    if s_y is None:
        s_y = (x.abs().max() + SCALE_MIN_THRES) / fp8MaxValue

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _fp8_division_kernel[grid](
        y,
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
        SCALE_MIN_THRES=SCALE_MIN_THRES,
        STOCHASTIC=stochastic,
    )

    # Recover 2D to 3D
    if batched:
        y = y.reshape(BS, -1, y.shape[-1])

    return y, s_y  # y_t is expected to be 2D tensor


# I change the dtype of both the input tensor and the output tensor. I use torch.float32, torch.float16, and torch.fp8

configs = []
for SL in [8192]:
    configs.append(
        triton.testing.Benchmark(  # test different matrix size influence
            x_names=["CDIM"],
            x_vals=[1024, 2048, 4096, 8192],
            line_arg="provider",
            line_vals=["triton", "torch"],
            line_names=["triton", "torch"],
            styles=[("blue", "-"), ("green", "-")],
            ylabel="time-cost",
            plot_name=f"FP8gelu<SL={SL}>",
            args={"BS": 4, "SL": SL, "QB": 16, "fp8type": torch.float8_e4m3fn, "mode": "time-consuming"},
        )
    )


@triton.testing.perf_report(configs)
def bench_load_store(
    BS, SL, CDIM, QB, fp8type, provider, mode="forward"
):  # I only use triton as the provider, and mode when benchmarking
    # create data
    x = torch.randn(BS, SL, CDIM).cuda()
    _qx = x.reshape(BS, SL, CDIM // QB, QB)
    sx = _qx.abs().amax(dim=(3)) / FP8_MAX_VALUE[fp8type]
    sx = sx.to(torch.bfloat16)
    _qx = (_qx / sx.unsqueeze(3)).to(fp8type)
    qx = _qx.reshape(BS, SL, CDIM)

    quantiles = [0.5, 0.2, 0.8]
    # utility functions
    if provider == "triton":

        def y_fwd():
            fp8_division(qx, sx, QB)

    if provider == "torch":
        torch_gelu = torch.nn.SiLU()

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


def validity_check(BS, SL, CDIM, QB, fp8type=torch.float8_e4m3fn):
    # create data
    x = torch.randn(BS * SL, CDIM).cuda()

    # torch result
    avg_output_triton = torch.zeros_like(x)

    # triton result
    for _ in range(100):
        x_triton, s_triton, x_triton_t = fp8_division(x, QB, "E4M3", stochastic=False)
        output_triton = x_triton.float() * s_triton

        avg_output_triton = avg_output_triton + output_triton
    avg_output_triton /= 100

    import IPython

    IPython.embed()


