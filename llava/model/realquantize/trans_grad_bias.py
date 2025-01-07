import torch

# 4 block
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .common import FP8_MAX_VALUE, SCALE_MIN_THRES

"""Calculate the gradient of bias Operator"""
"""Input uses per-tensor quantization, and should be transposed"""
"""Output uses similar to the bias shape"""
"""The input can be 2D or 3D, but the calculation is performed in 2D"""

# The kernel with 1 load operation and 4 store operation
def get_configs_io_block():
    configs = []
    for nstages in [3, 4, 5]:
        for block_m in [32, 64, 128]:
            for block_n in [32, 64, 128]:
                for nwarps in [4, 8, 16]:
                    configs.append(
                        triton.Config(
                            {"BLOCK_M": block_m, "BLOCK_N": block_n},
                            num_stages=nstages,
                            num_warps=nwarps,
                        )
                    )
    return configs


convert_str_to_fp8 = {"E4M3": torch.float8_e4m3fn, "E5M2": torch.float8_e5m2}


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
def _fp8_trans_grad_bias_kernel(
    output_scale_ptr,  # output
    input_t_ptr,  # input
    M,
    N,
    SN,
    QB: tl.constexpr,
    fp8_max,  # shape
    input_stride_0,
    input_stride_1,  # input stride
    s_output_stride_0,
    s_output_stride_1,  # scale of output stride
    SCALE_MIN_THRES: tl.constexpr,
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

    output = tl.reshape(input, (BLOCK_M, BLOCK_SN, QB))

    # Quantize Scale calculation
    abs_output = tl.abs(output)
    max_val = tl.max(abs_output, axis=2) + SCALE_MIN_THRES
    scale_output = max_val / fp8_max
    scale_output = tl.reshape(scale_output, (BLOCK_M, BLOCK_SN, 1))

    scale_output = scale_output.to(output_scale_ptr.type.element_ty)
    scale_output = tl.reshape(scale_output, (BLOCK_M, BLOCK_SN))

    scale_output_ptr = tl.make_block_ptr(
        base=output_scale_ptr,
        shape=(M, SN),
        strides=(s_output_stride_0, s_output_stride_1),
        offsets=(pid_dim0 * BLOCK_M, pid_dim1 * BLOCK_SN),
        block_shape=(BLOCK_M, BLOCK_SN),
        order=(1, 0),
    )

    tl.store(scale_output_ptr, scale_output, boundary_check=(0, 1))


def fp8_quantize_and_transpose(x, QB, fp8type, transpose_output_2d=False, stochastic=False):
    # Change batched 3D input to 2D
    batched = False
    if len(x.shape) == 3:
        batched = True
        BS = x.shape[0]
        x = x.reshape(-1, x.shape[-1])

    # defining the input and output tensor
    M, N = x.shape
    SN = N // QB

    fp8type = convert_str_to_fp8[fp8type]
    s_y = torch.empty((M, SN), dtype=torch.float32, device=x.device)
    fp8MaxValue = FP8_MAX_VALUE[fp8type]  # E4M3 and E5M2 have different max value

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)

    _fp8_quantize_and_transpose_kernel[grid](
        s_y,
        x,
        M,
        N,
        SN,
        QB,
        fp8MaxValue,
        x.stride(0),
        x.stride(1),
        s_y.stride(0),
        s_y.stride(1),
        SCALE_MIN_THRES=SCALE_MIN_THRES,
    )

    s_y_max = s_y.max()
    qy, s_y_max, qy_t = fp8_division_transpose(
        x, QB, fp8type, s_y_max, stochastic=stochastic
    )  # Stochastic Rounding happens here

    # Recover 2D to 3D
    if batched:
        qy = qy.reshape(BS, -1, qy.shape[-1])
        if not transpose_output_2d:
            qy_t = qy_t.reshape(BS, -1, qy_t.shape[-1])

    return qy, s_y_max, qy_t  # y_t is expected to be 2D tensor


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
            fp8_quantize_and_transpose(qx, sx, QB)

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

    # triton result
    x_triton, s_triton, x_triton_t = fp8_quantize_and_transpose(x, QB, "E4M3")

    _x_triton = x_triton.reshape(BS * SL, CDIM // QB, QB)
    _x_triton = _x_triton.to(torch.float32)
    s_triton = s_triton.unsqueeze(2)
    output_triton = (_x_triton * s_triton).reshape(BS * SL, CDIM)

    import IPython

    IPython.embed()


if __name__ == "__main__":
    torch.manual_seed(0)
    torch.set_printoptions(precision=8, linewidth=1600, sci_mode=False, edgeitems=3)
    validity_check(BS=4, SL=256, CDIM=512, QB=16, fp8type=torch.float8_e4m3fn)
    bench_load_store.run(save_path=f"result/time/multi_quantize_block_quantize/BLSZ=64", print_data=True)


