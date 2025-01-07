import re

import torch

try:
    from .FloatPointQuantizeTorch import *
    from .FloatPointQuantizeTriton import *
except:
    from FloatPointQuantizeTorch import *
    from FloatPointQuantizeTriton import *


def block_cut(input, row_block, column_block, pad_block=False):
    # print(input.shape)
    original_shape = input.shape
    # input tensor shape is M * N
    if len(input.shape) > 2:
        input = input.reshape(-1, input.shape[2])
    elif len(input.shape) == 2:
        pass
    else:
        raise ValueError(f"input shape {input.shape} does not match for block cut, {input}")
    M, N = input.shape[0], input.shape[1]

    if row_block == -1:
        row_block = M
    if column_block == -1:
        column_block = N

    if pad_block:
        row_remainder, col_remainder = M % row_block, N % column_block
        if row_remainder:
            row_pad = row_block - row_remainder
        else:
            row_pad = 0
        if col_remainder:
            col_pad = column_block - col_remainder
        else:
            col_pad = 0

        input = torch.nn.functional.pad(
            input, (0, col_pad, 0, row_pad), "constant", 0
        )  # refer to torch's doc to see why
        M, N = input.shape[0], input.shape[1]
        row_num, column_num = M // row_block, N // column_block
    else:
        row_num, column_num = M // row_block, N // column_block

    assert row_num * row_block == M, f"{row_num}, {row_block}, {M}, {original_shape}"
    assert column_num * column_block == N, f"{column_num}, {column_block}, {N}, {original_shape}"
    # print(input.shape)
    input = (
        input.reshape(row_num, row_block, column_num, column_block)
        .permute(0, 2, 1, 3)
        .reshape(row_num * column_num, row_block, column_block)
    )
    # print(input.shape)
    return input


def block_reshape(input, origin_input, row_block, column_block, pad_block=False):
    if len(origin_input.shape) > 2:
        flatten_input = origin_input.reshape(-1, origin_input.shape[2])
    elif len(origin_input.shape) == 2:
        flatten_input = origin_input
    else:
        raise ValueError(f"input shape {input.shape} does not match for block cut")

    M, N = flatten_input.shape[0], flatten_input.shape[1]

    if row_block == -1:
        row_block = M
    if column_block == -1:
        column_block = N

    if pad_block:
        row_remainder, col_remainder = M % row_block, N % column_block
        if row_remainder:
            row_pad = row_block - row_remainder
        else:
            row_pad = 0
        if col_remainder:
            col_pad = column_block - col_remainder
        else:
            col_pad = 0

        pad_origin_input = torch.nn.functional.pad(origin_input, (0, col_pad, 0, row_pad), "constant", 0)
        M, N = pad_origin_input.shape[0], pad_origin_input.shape[1]
        row_num, column_num = M // row_block, N // column_block
    else:
        row_num, column_num = M // row_block, N // column_block

    input = (
        input.reshape(row_num, column_num, row_block, column_block)
        .permute(0, 2, 1, 3)
        .reshape(row_num * row_block, column_num * column_block)
    )

    M, N = flatten_input.shape[0], flatten_input.shape[1]
    input = input[:M, :N]

    if len(origin_input.shape) > 2:
        input = input.reshape(origin_input.shape)
    elif len(origin_input.shape) == 2:
        pass
    else:
        raise ValueError(f"input shape {input.shape} does not match for block reshape")

    return input


def block_verify_int8(input, row_block, column_block, layer_type, necessary=True):
    Binput = block_cut(input, row_block, column_block)
    Binput = Binput.to(torch.float32)

    for n in range(Binput.shape[0]):
        unique_values = len(torch.unique(Binput[n, :, :]))
        if unique_values > 256:
            if necessary:
                raise ValueError(f"{layer_type} contains more than 256 unique values.")
            else:
                return False
    return True


def block_quant(input, symm, bits, stochastic, epsilon, apply_quantize, layer_name):
    Quant_fn = SymmQuantizer
    return Quant_fn.apply(input, symm, bits, stochastic, epsilon, apply_quantize, layer_name)


def extract_bit(string):
    match = re.match(r"INT(\d+)", string)
    if match:
        return "integer", int(match.group(1)), None
    match = re.match(r"E(\d+)M(\d+)", string)
    if match:
        Ebit, Mbit = int(match.group(1)), int(match.group(2))
        if Ebit == 1:
            return "integer", Mbit + 1, None
        if Mbit == 0:
            return "floatExM0", int(match.group(1)), 0
        return "floatExMy", int(match.group(1)), int(match.group(2))
    match = re.match(r"DE(\d+)", string)
    if match:
        return "Dynamic", int(match.group(1)), None
    match = re.match(r"ZeroD(\d+)", string)
    if match:
        return "ZeroDynamic", int(match.group(1)), None
    raise ValueError(f"{string} data format is not supported")


class SymmQuantizer(torch.autograd.function.InplaceFunction):
    @staticmethod
    def forward(ctx, input, symm, bits, stochastic, epsilon, apply_quantize=True, layer_name=None):
        with torch.no_grad():
            absmax_per_block = input.abs().amax(dim=(1, 2)).unsqueeze(1).unsqueeze(2) + epsilon

            if bits == "100" or not apply_quantize:
                return input, input, torch.ones_like(absmax_per_block)
            elif bits == "FP32":
                return input.to(torch.float32), input.to(torch.float32), torch.ones_like(absmax_per_block)
            elif bits == "FP16":
                return input.to(torch.float16), input.to(torch.float16), torch.ones_like(absmax_per_block)
            elif bits == "BF16":
                return input.to(torch.bfloat16), input.to(torch.bfloat16), torch.ones_like(absmax_per_block)
            else:
                QuantType, bit1, bit2 = extract_bit(bits)
                if not symm:
                    bit1 = bit1 + 1  # pretend to be asymmtric

                if QuantType == "integer":
                    Qn, Qp = -(2 ** (bit1 - 1) - 1), 2 ** (bit1 - 1) - 1
                elif QuantType == "floatExMy":
                    Qn, Qp = -(2 - 2 ** (-bit2)) * (2 ** (2 ** (bit1 - 1))), (2 - 2 ** (-bit2)) * (
                        2 ** (2 ** (bit1 - 1))
                    )
                    if bit1 == 4 and bit2 == 3:
                        Qn, Qp = -448, 448
                    if bit1 == 5 and bit2 == 2:
                        Qn, Qp = -57344, 57344
                elif QuantType == "floatExM0":
                    Qn, Qp = -(2 ** (2 ** (bit1 - 1))) + 1, 2 ** (2 ** (bit1 - 1))
                elif QuantType == "Dynamic":
                    Qn, Qp = -1, 1
                elif QuantType == "ZeroDynamic":
                    Qn, Qp = -1, 1
                else:
                    raise NotImplementedError(f"{bits} is not supported by quantization")
                scale_per_block = (2 * absmax_per_block) / (Qp - Qn)
                scale_per_block = scale_per_block.to(input)

                Qinput = input / scale_per_block

                if QuantType == "integer":
                    if stochastic:
                        noise = Qinput.new(Qinput.shape).uniform_(-0.5, 0.5)
                        Qinput.add_(noise)
                    Qinput.clamp_(Qn, Qp).round_()
                elif QuantType == "floatExMy":
                    # Qinput = floatExMy_quantize_torch(Qinput, bit1, bit2, stochastic)
                    Qinput = floatExMy_quantize_triton(Qinput, bit1, bit2, stochastic)
                elif QuantType == "floatExM0":
                    Qinput = floatExM0_quantize_torch(Qinput, bit1, stochastic)
                elif QuantType == "Dynamic":
                    Qinput = Dynamic_quantize_torch(Qinput, bit1, stochastic)
                elif QuantType == "ZeroDynamic":
                    Qinput = ZeroDynamic_quantize_torch(Qinput, bit1, stochastic)
                else:
                    raise NotImplementedError(f"{bits} is not supported by quantization")

                RQinput = Qinput * scale_per_block
                # print(f'Layer Name: {layer_name}: {input, Qinput, scale_per_block, absmax_per_block}', file=open('debug.txt', 'a'))

                if input.dtype != Qinput.dtype:
                    print(
                        f"Input type is {input.dtype}, Qinput type is {Qinput.dtype}, scale_per_block type is {scale_per_block.dtype}",
                        file=open("debug.txt", "a"),
                    )
                    import IPython

                    IPython.embed()
                return RQinput, Qinput, scale_per_block

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None


if __name__ == "__main__":
    import time

    torch.manual_seed(0)
    X = (torch.rand(2048, 4096).cuda() - 0.5) * 1000

    X = X.to(torch.bfloat16)

    B_X = block_cut(X, -1, -1)
    RQ_X, Q_X, S_X = block_quant(
        B_X, symm=True, bits="E2M0", stochastic=True, epsilon=1e-14, apply_quantize=True, layer_name=""
    )
    RQ_X = block_reshape(RQ_X, X, -1, -1)
    Q_X = block_reshape(Q_X, X, -1, -1)

    # start = time.time()
    # for _ in range(10000):
    #     B_X = block_cut(X, 2, 2)
    #     RQ_X, Q_X, S_X = block_quant(B_X, symm=True, bits="E5M2", stochastic=True, epsilon=1e-14, apply_quantize=True, layer_name='')
    #     RQ_X = block_reshape(RQ_X, X, 2, 2)
    #     Q_X = block_reshape(Q_X, X, 2, 2)
    #     torch.cuda.synchronize()

    # end = time.time()
    # print(f"Time cost: {end - start}")

    print(X.dtype)
    import IPython

    IPython.embed()

    # X = torch.tensor([
    #     [1.1, 1, 1, 2, 2, 2, 3, 3, 3.01],
    #     [1.2, 1, 1, 2, 2, 2, 3, 3, 3.02],
    #     [4.1, 4, 4, 5, 5, 5, 6, 6, 6.01],
    #     [4.2, 4, 4, 5, 5, 5, 6, 6, 6.02],
    #     [7.1, 7, 7, 8, 8, 8, 9, 9, 9.01],
    #     [7.2, 7, 7, 8, 8, 8, 9, 9, 9.02],
    # ])

    # B_X = block_cut(X, 2, 3)
    # print(B_X)
    # print(B_X.shape)
    # Q_X = block_quant(B_X, symm=True, bits=2, stochastic=False)
    # print(Q_X)
    # print(Q_X.shape)


