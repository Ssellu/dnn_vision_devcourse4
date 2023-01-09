"""
Coding : Make Convolution

- Make convolution using Numpy 

  1. Sliding window Convolution
  2. IM2COL GEMM Convolution
"""

import numpy as np

from function.convolution import MyConv


def convolution():
    print("Convolution")

    in_w = 3    # input width
    in_h = 3    # input height
    in_c = 1    # input channel
    out_c = 16  # output channel
    batch = 1   # batch size
    k_w = 3     # kernel width
    k_h = 3     # kernel height

    X = np.arange(9, dtype=np.float32).reshape([batch, in_c, in_h, in_w])
    W = np.array(np.random.standard_normal([out_c, in_c, k_h, k_w]))

    print('X : {}'.format(X))
    print('W : {}'.format(W))

    convolution = MyConv(batch=batch,
                  in_c=in_c,
                  out_c=out_c,
                  in_h=in_h,
                  in_w=in_w,
                  k_h=k_h,
                  k_w=k_w,
                  dilation=1,
                  stride=1,
                  pad=0)
    L1 = convolution.conv(X, W)
    print("--- L1 ---")
    print("X Shape : {}".format(X.shape))
    print("W Shape : {}".format(W.shape))
    print("L1 Shape : {}".format(L1.shape))
    print(L1)

if __name__ == '__main__':
    convolution()
