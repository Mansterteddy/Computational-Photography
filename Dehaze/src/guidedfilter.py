#encoding: utf-8
"""
Implementation for Guided Image Filtering
"""

from itertools import combinations_with_replacement
from collections import defaultdict

import numpy as np
from numpy.linalg import inv

#index for convenience
R, G, B = 0, 1, 2

def boxfilter(I, r):
    """
    Fast box filter implementation

    Parameters:

    I: a single channel / gray image data normalized to [0.0, 1.0]
    r: window radius

    Return:

    The filtered image data
    """

    M, N = I.shape
    dest = np.zeros((M, N))

    sumY = np.cumsum(I, axis = 0)

    dest[: r + 1] = sumY[r: 2 * r + 1]
    dest[r + 1: M - r] = sumY[2 * r + 1:] - sumY[: M - 2 * r - 1]
    #np.tile 将后面第一个参数，复制成以第二个参数为格式的数组
    dest[-r:] =np.tile(sumY[-1], (r, 1)) - sumY[M - 2 * r - 1: M - r - 1]

    sumX = np.cumsum(dest, axis = 1)

    dest[:, :r + 1] = sumX[:, r: 2 * r + 1]
    dest[:, r + 1 : N - r] = sumX[:, 2 * r + 1:] - sumX[:, :N - 2 * r - 1]
    dest[:, -r:] = np.tile(sumX[:, -1][:, None], (1, r)) - sumX[:, N - 2 * r - 1: N - r - 1]

    return dest


def guided_filter(I, p, r = 40, eps = 1e-3):
    """
    Refine a filter under the guidance of another image

    Parameters:

    I: a M * N * 3 RGB image for guidance
    p: the M * N filter to be guided 
    r: the radius of the guidance
    eps: epsilon for the guided filter

    Return:
    
    The guided filter
    """
    M, N = p.shape 
    base = boxfilter(np.ones((M, N)), r)

    means = [boxfilter(I[:, :, i], r) / base for i in xrange(3)]

    mean_p = boxfilter(p, r) / base 

    means_IP = [boxfilter(I[:, :, i] * p, r) / base for i in xrange(3)]

    covIP = [means_IP[i] - means[i] * mean_p for i in xrange(3)]

    var = defaultdict(dict)
    for i, j in combinations_with_replacement(range(3), 2):
        var[i][j] = boxfilter(I[:, :, i] * I[:, :, j], r) / base - means[i] * means[j]

    a = np.zeros((M, N, 3))

    for y, x in np.ndindex(M, N):
        Sigma = np.array([
        [var[R][R][y, x], var[R][G][y, x], var[R][B][y, x]],
        [var[R][G][y, x], var[G][G][y, x], var[G][B][y, x]],
        [var[R][B][y, x], var[G][B][y, x], var[B][B][y, x]],
        ])
        cov = np.array([c[y, x] for c in covIP])
        a[y, x] = np.dot(cov, inv(Sigma + eps * np.eye(3)))


    b = mean_p - a[:, :, R] * means[R] - a[:, :, G] * means[G] - a[:, :, B] * means[B]

    q = (boxfilter(a[:, :, R], r) * I[:, :, R] + boxfilter(a[:, :, G], r) * I[:, :, G] + boxfilter(a[:, :, B], r) * I[:, :, B] + boxfilter(b, r)) / base

    return q 

"""
Algorithm:

1. 
mean_I = f_mean(I)
mean_p = f_mean(p)
corr_I = f_mean(I.* I)
corr_IP = f_mean(I.* p)

2.
VarI = corrI - mean_I.* mean_I
cov_Ip = corr_Ip - mean_I.* mean_p

3.
a = cov_Ip ./ (var_I + sigma)
b = mean_p - a.* meanI 

4.
mean_a = fmean(a)
mean_b = fmean(b)

5.
q = mean_a.* I + mean_b

"""

print boxfilter(np.ones((4, 4)), 1)
