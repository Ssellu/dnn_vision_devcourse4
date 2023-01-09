import numpy as np
import time, sys
import torch
import torch.nn as nn
from function.convolution import Conv
from function.pool import Pool
from function.fc import FC
from function.activation import *

import matplotlib.pyplot as plt

def convolution():
    """_summary_
    naive convolution and Im2Col&gemm convolution
    """    
    print("convolution")
    
    #define the shape of input & weight
    in_w = 6
    in_h = 6
    in_c = 3
    out_c = 16
    batch = 1
    k_w = 3
    k_h = 3
    
    X = np.arange(in_w*in_h*in_c*batch, dtype=np.float32).reshape([batch, in_c, in_h, in_w])
    W = np.array(np.random.standard_normal([out_c,in_c,k_h,k_w]), dtype=np.float32)
    
    print("X")
    print(X)
    
    Convolution = Conv(batch = batch,
                        in_c = in_c,
                        out_c = out_c,
                        in_h = in_h,
                        in_w = in_w,
                        k_h = k_h,
                        k_w = k_w,
                        dilation = 1,
                        stride = 1,
                        pad = 0)
    
    #print("X shape : ", X.shape)
    #print("W shape : ", W.shape)
    
    l1_time = time.time()
    for i in range(5):
        L1 = Convolution.conv(X,W)
    print("L1 time : ", time.time() - l1_time)
    
    # print("L1")
    # print(L1)
    
    l2_time = time.time()
    for i in range(5):
        L2 = Convolution.gemm(X,W)
    print("L2 time : ", time.time() - l2_time)

    print("L2")
    print(L2)
    
    #pytorch conv
    torch_conv = nn.Conv2d(in_c,
                           out_c,
                           kernel_size = k_h,
                           stride = 1,
                           padding = 0,
                           bias = False,
                           dtype = torch.float32)
    torch_conv.weight = torch.nn.Parameter(torch.tensor(W))
    
    L3 = torch_conv(torch.tensor(X, requires_grad=False, dtype=torch.float32))
    print("L3")
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
    
    Convolution = Conv(batch = batch,
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
    
def plot_activation():
    """_summary_
    Plot the activation output of [-10,10] inputs
    activations : relu, leaky_relu, sigmoid, tanh
    """    
    x = np.arange(-10,10,1)
    
    out_relu = relu(x)
    out_leaky = leaky_relu(x)
    out_sigmoid = sigmoid(x)
    out_tanh = tanh(x)

    #print(out_relu, out_leaky, out_sigmoid, out_tanh)
    
    plt.plot(x, out_relu, 'r', label='relu')
    plt.plot(x, out_leaky, 'b', label='leaky')
    plt.plot(x, out_sigmoid, 'g', label='sigmoid')
    plt.plot(x, out_tanh, 'bs', label='tanh')
    plt.ylim([-2,2])
    plt.legend()
    plt.show()
    
def shallow_network():
    """_summary_
    'Conv - MaxPool - FC' shallow model's forward and backward code
    """
    #input [1,1,6,6], 2 iter
    X = [np.array(np.random.standard_normal([1,1,6,6]), dtype=np.float32),
         np.array(np.random.standard_normal([1,1,6,6]), dtype=np.float32)]
    #GT
    Y = np.array([1,1], dtype=np.float32)
    
    #conv1 weights. [1,1,3,3]
    W1 = np.array(np.random.standard_normal([1,1,3,3]), dtype=np.float32)
    #fc weights. [1,4]
    W2 = np.array(np.random.standard_normal([1,4]), dtype=np.float32)
    
    padding = 0 
    stride = 1

    #L1 layer shape w,h
    L1_h = (X[0].shape[2] - W1.shape[2] + 2 * padding) // stride + 1
    L1_w = (X[0].shape[3] - W1.shape[2] + 2 * padding) // stride + 1
    
    print("L1 output : {} {}".format(L1_h, L1_w))
    
    #Conv1
    Convolution = Conv(batch = X[0].shape[0],
                       in_c =X[0].shape[1],
                       out_c = W1.shape[0],
                       in_h = X[0].shape[2],
                       in_w = X[0].shape[3],
                       k_h = W1.shape[2],
                       k_w = W1.shape[3],
                       dilation = 1,
                       stride = stride,
                       pad = padding)
    #conv1 diff
    Conv_diff = Conv(batch = X[0].shape[0],
                     in_c = X[0].shape[1],
                     out_c = W1.shape[0],
                     in_h = X[0].shape[2],
                     in_w = X[0].shape[3],
                     k_h = L1_h,
                     k_w = L1_w,
                     dilation = 1,
                     stride = 1,
                     pad = 0)
    #FC
    Fc = FC(batch = X[0].shape[0],
            in_c = X[1].shape[1],
            out_c = 1,
            in_h = L1_h,
            in_w = L1_w)
    #Max Pooling
    Pooling = Pool(batch = X[0].shape[0],
                   in_c = W1.shape[0],
                   out_c = W1.shape[0],
                   in_h = L1_h,
                   in_w = L1_w,
                   kernel = 2,
                   dilation = 1,
                   stride = 2,
                   pad = 0)
    
    num_epoch = 1000
    for e in range(num_epoch):
        total_loss = 0
        for i in range(len(X)):
            #forward
            L1 = Convolution.gemm(X[i], W1)
            #print("L1")
            #print(L1)
            L1_act = np.array(sigmoid(L1), dtype=np.float32)
            #print("L1_act")
            #print(L1_act)
            
            L1_max = Pooling.pool(L1_act)
            
            #print("L1 max")
            #print(L1_max)
            
            L1_max_flatten = np.reshape(L1_max, (1,-1))
            #print(L1_max_flatten.shape)
            
            L2 = Fc.fc(L1_max_flatten, W2)
            #print("L2")
            #print(L2)
            
            L2_act = np.array(sigmoid(L2), dtype=np.float32)
            
            #print("L2_act")
            #print(L2_act)
            
            #L2 error
            loss = np.square(Y[i] - L2_act) * 0.5
            #print(loss)
            total_loss += loss.item()
            
            
            #Backward (Backpropogation)
            
            #delta E / delta W2
            
            diff_w2_a = L2_act - Y[i]
            
            diff_w2_b = L2_act*(1 - L2_act)
            
            diff_w2_c = L1_max
            
            diff_w2 = diff_w2_a * diff_w2_b * diff_w2_c
            diff_w2 = np.reshape(diff_w2, (1,-1))
            # print("{} / {} / {}".format(diff_w2_a, diff_w2_b, diff_w2_c))
            # print("diff_w2 : ", diff_w2)
            
            #delta E / delta W1
            diff_w1 = 1
            diff_w1_a = diff_w2_a * diff_w2_b
            
            # print("diff_w1_a")
            # print(diff_w1_a)
            
            diff_w1_b = np.reshape(W2,(1,1,2,2))
            
            # print("diff_w1_b")
            # print(diff_w1_b)
            
            diff_w1_b = diff_w1_b.repeat(2, axis=2).repeat(2,axis=3)
            
            # print("L1_act")
            # print(L1_act)
            # print("L1_max")
            # print(L1_max)

            L1_max_repeat = L1_max.repeat(2, axis=2).repeat(2, axis=3)
            #diff maxpool
            diff_w1_c = np.equal(L1_act, L1_max_repeat).astype(int)
            #print(diff_w1_c)
            
            diff_w1_d = diff_w1_c * L1_act * (1 - L1_act)
            
            #print(diff_w1_d)
            
            #diff_w1_e = X[i]
            
            diff_w1 = diff_w1_a * diff_w1_b * diff_w1_c * diff_w1_d
            
            diff_w1 = Conv_diff.gemm(X[i], diff_w1)
            
            # print("diff_w1")
            # print(diff_w1)
            
            
            #update
            W2 = W2 - 0.01 * diff_w2
            W1 = W1 - 0.01 * diff_w1
            
        print("{} epoch loss {}".format(e, total_loss / len(X)))
            

                   
    
    

if __name__ == "__main__":
    #convolution()
    #plot_activation()
    #forward_net()
    shallow_network()