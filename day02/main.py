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
from function.pooling import Pool
from function.fc import FC

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


def forward_net():
    """_summary_
    'Conv - Pooling - FC' model inference code 
    """
    #define
    batch = 1
    in_c = 3
    in_w = 6
    in_h = 6
    k_h = 3
    k_w = 3
    out_c = 1
    
    X = np.arange(batch*in_c*in_w*in_h, dtype=np.float32).reshape([batch,in_c,in_w,in_h])
    W1 = np.array(np.random.standard_normal([out_c,in_c,k_h,k_w]), dtype=np.float32)
    
    Convolution = MyConv(batch = batch,
                        in_c = in_c,
                        out_c = out_c,
                        in_h = in_h,
                        in_w = in_w,
                        k_h = k_h,
                        k_w = k_w,
                        dilation = 1,
                        stride = 1,
                        pad = 0)
    
    L1 = Convolution.gemm(X,W1)
    
    print("L1 shape : ", L1.shape)
    print(L1)
    
    Pooling = Pool(batch=batch,
                   in_c = 1,
                   out_c = 1,
                   in_h = 4,
                   in_w = 4,
                   kernel=2,
                   dilation=1,
                   stride=2,
                   pad = 0)
    
    L1_MAX = Pooling.pool(L1)
    print("L1_MAX shape : ", L1_MAX.shape)
    print(L1_MAX)
    
    #fully connected layer
    W2 = np.array(np.random.standard_normal([1, L1_MAX.shape[1] * L1_MAX.shape[2] * L1_MAX.shape[3]]), dtype=np.float32)
    Fc = FC(batch = L1_MAX.shape[0],
            in_c = L1_MAX.shape[1],
            out_c = 1,
            in_h = L1_MAX.shape[2],
            in_w = L1_MAX.shape[3])

    L2 = Fc.fc(L1_MAX, W2)
    
    print("L2 shape : ", L2.shape)
    print(L2)

if __name__  == '__main__':
    # convolution()
    forward_net()
