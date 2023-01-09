import numpy as np


class MyConv:
    def __init__(self, batch, in_c, out_c, in_h, in_w, k_h, k_w, dilation, stride, pad) -> None:
        self.batch = batch
        self.in_c = in_c
        self.out_c = out_c
        self.in_h = in_h
        self.in_w = in_w
        self.k_h = k_h
        self.k_w = k_w
        self.dilation = dilation
        self.stride = stride
        self.pad = pad

        self.out_h = (in_h - k_h + 2 * pad) // stride + 1
        self.out_w = (in_w - k_w + 2 * pad) // stride + 1


    def check_range(self, a, b):
        return a > -1 and a < b

    # Naive convolution. Sliding Window metric
    def conv(self, A, B):
        C = np.zeros((self.batch, self.out_c, self.out_h, self.out_w), dtype=np.float32)

        # 7-Loop
        for b in range(self.batch):
            for oc in range(self.out_c):
                # Each channel of output
                for oh in range(self.out_h):
                    for ow in range(self.out_w):
                        # Each pixel of output shape
                        a_j = oh * self.stride - self.pad
                        for kh in range(self.k_h):
                            if not self.check_range(a_j, self.in_h):
                                C[b, oc, oh, ow] += 0
                            else:
                                a_i = ow * self.stride - self.pad
                                for kw in range(self.k_w):
                                    if not self.check_range(a_i, self.in_w):
                                        C[b, oc, oh, ow] += 0
                                    else:
                                        C[b, oc, oh, ow] += np.dot(A[b, :, a_j, a_i], B[oc, :, kh, kw])
                                    a_i += self.stride
                            a_j += self.stride 
        return C
