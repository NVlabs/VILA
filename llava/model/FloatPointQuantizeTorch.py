import math

import torch


def floatExMy_quantize_torch(x, e_bit, m_bit, stochastic):
    sign, x_abs = x.sign(), x.abs()
    Elow, Ehigh, Mhigh = -(2 ** (e_bit - 1)) + 2, 2 ** (e_bit - 1), 2**m_bit
    expo = torch.floor(torch.log2(x_abs))
    expo = torch.clamp(expo, min=Elow, max=Ehigh)
    mant = x_abs / torch.exp2(expo)

    mant_int = torch.floor(mant)
    mant_frac = mant - mant_int
    mant_frac = mant_frac * Mhigh
    if stochastic:
        noise = mant_frac.new(mant_frac.shape).uniform_(-0.5, 0.5)
        mant_frac.add_(noise)
    mant_frac = torch.round(mant_frac)

    mant_q = mant_int + mant_frac / Mhigh
    y = sign * (2**expo) * mant_q
    y = y.to(x)

    return y


def floatExM0_quantize_torch(x, e_bit, stochastic):
    sign, x_abs = x.sign(), x.abs()
    Elow, Ehigh = -(2 ** (e_bit - 1)) + 1, 2 ** (e_bit - 1)
    expo = torch.log2(x_abs)
    if stochastic:
        noise = expo.new(expo.shape).uniform_(-0.5, 0.5)
        expo.add(noise)
        log_bias = math.log2(4 / 3) - 1 / 2
        expo.add(torch.ones_like(expo) * log_bias)
    expo = torch.clamp(expo, min=Elow - 1, max=Ehigh)
    expo = torch.round(expo)

    y = sign * (2**expo) * (expo > Elow)  # When underflow, set the value to 0
    y = y.to(x)

    return y


def Dynamic_quantize_torch(x, bit, stochastic):
    if stochastic:
        raise NotImplementedError("Dynamic Tree quantization does not support stochastic")
    sign, x_abs = x.sign(), x.abs()
    expo = torch.ceil(torch.log10(x_abs))
    expo = torch.clamp(expo, min=2 - bit)
    mant = (10 * x_abs / torch.pow(10, expo) - 1) / 9  # Range from 0 - 1

    mant_frac = mant * 2 ** (bit - 2 - expo.abs())
    mant_frac = torch.round(mant_frac)
    mant_frac = mant_frac / (2 ** (bit - 2 - expo.abs())) * 9 + 1
    y = sign * (10**expo) * mant_frac / 10

    zero_mask = y.abs() > 1.01 * 10 ** (1 - bit)
    y = y * zero_mask
    y = y.to(x)
    return y


def ZeroDynamic_quantize_torch(x, bit, stochastic):
    if stochastic:
        raise NotImplementedError("Dynamic Tree quantization does not support stochastic")
    sign, x_abs = x.sign(), x.abs()
    expo = torch.ceil(torch.log10(x_abs))
    expo = torch.clamp(expo, min=2 - bit)
    mant = (10 * x_abs / torch.pow(10, expo) - 1) / 9  # Range from 0 - 1

    mant_frac = mant * 2 ** (bit - 2 - expo.abs())
    mant_frac = torch.round(mant_frac)
    mant_frac = mant_frac / (2 ** (bit - 2 - expo.abs())) * 9 + 1
    y = sign * (10**expo) * mant_frac / 10

    y = y.to(x)
    return y


