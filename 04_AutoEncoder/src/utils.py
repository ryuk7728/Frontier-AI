import math


def psnr(mse):
    return -10 * math.log10(mse)
