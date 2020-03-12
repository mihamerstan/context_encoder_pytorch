from __future__ import division
from past.utils import old_div
import numpy
import math
import scipy.misc


def psnr(img1, img2):
    mse = numpy.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(old_div(PIXEL_MAX, math.sqrt(mse)))

# real = scipy.misc.imread('dataset/test/test/006_im.png').astype(numpy.float32)[32:32+64,32:32+64,:]
# recon = scipy.misc.imread('out.png').astype(numpy.float32)[32:32+64,32:32+64,:]


# print(psnr(real,recon))

