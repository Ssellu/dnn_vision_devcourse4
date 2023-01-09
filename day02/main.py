"""
Coding : Make Convolution

- Make convolution using Numpy 

  1. Sliding window Convolution
  2. IM2COL GEMM Convolution
"""
import time
import numpy as np
import torch
import torch.nn as nn
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

    l1_time = time.time()
    for i in range(5):  # NOTE This loop is to see easier of the time difference between 7-loop and im2col by 5 times of the operation
        L1 = convolution.conv(X, W)
    print("L1 time : ", time.time() - l1_time)
    print("--- L1 ---")
    print("X Shape : {}".format(X.shape))
    print("W Shape : {}".format(W.shape))
    print("L1 Shape : {}".format(L1.shape))
    print(L1)

    l2_time = time.time()
    for i in range(5):  # NOTE This loop is to see easier of the time difference between 7-loop and im2col by 5 times of the operation
        L2 = convolution.gemm(X, W)
    print("L2 time : ", time.time() - l2_time)
    print("--- L2 ---")
    print("X Shape : {}".format(X.shape))
    print("W Shape : {}".format(W.shape))
    print("L2 Shape : {}".format(L1.shape))
    print(L2)

    # pytorch Conv
    torch_conv = nn.Conv2d(in_c, 
                           out_c, 
                           kernel_size=k_h,
                           stride=1, 
                           padding=0, 
                           bias=False, 
                           dtype=torch.float32)
    torch_conv.weight = nn.Parameter(torch.tensor(W, dtype=torch.float32))
    L3 = torch_conv(torch.tensor(X, requires_grad=False, dtype=torch.float32))
    print("--- L3 ---")
    print(L3)


if __name__ == '__main__':
    convolution()
