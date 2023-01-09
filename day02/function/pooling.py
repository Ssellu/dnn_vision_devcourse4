import numpy as np

#2D pooling
class Pool:
    def __init__(self, batch, in_c, out_c, in_h, in_w, kernel, dilation, stride, pad):
        self.batch = batch
        self.in_c = in_c
        self.out_c = out_c
        self.in_h = in_h
        self.in_w = in_w
        self.kernel = kernel
        self.dilation = dilation
        self.stride = stride
        self.pad = pad
        self.out_w = (in_w - kernel + 2 * pad) // stride + 1
        self.out_h = (in_h - kernel + 2 * pad) // stride + 1
    
    def pool(self, A):
        C = np.zeros([self.batch, self.out_c, self.out_h, self.out_w], dtype=np.float32)
        for b in range(self.batch):
            for c in range(self.in_c):
                for oh in range(self.out_h):
                    a_j = oh * self.stride - self.pad
                    for ow in range(self.out_w):
                        a_i = ow * self.stride - self.pad
                        max_value = np.amax(A[:, c, a_j:a_j+self.kernel, a_i:a_i+self.kernel])
                        C[b,c,oh,ow] = max_value
        return C