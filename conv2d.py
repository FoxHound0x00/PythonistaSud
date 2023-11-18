import numpy as np


def conv(ip, kernel, padding=None, dilation=None, stride=None):
    iw, ih = ip.shape
    kw, kh = kernel.shape
    ow, oh = iw - kw + 1, ih - kh + 1
    op = np.zeros((ow, oh))

    if padding >= kh:
        return "Padding size should be smaller than the kernel size"

    for i in range(ow):
        for j in range(oh):
            patch = ip[i : i + kw, j : j + kh]
            op[i, j] += np.dot(patch.flatten(), kernel.flatten())
    return op


arr = np.random.rand(5, 5)
kernel = np.random.rand(2, 2)
print(conv(arr, kernel, padding=3))
