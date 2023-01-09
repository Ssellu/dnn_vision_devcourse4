import numpy as np

class FC:
    def __init__(self, batch, in_c, out_c, in_h, in_w):
        self.batch = batch
        self.out_c = out_c
        self.in_c = in_c
        self.in_h = in_h
        self.in_w = in_w 
        
    def fc(self, A, W):
        #A shape : [b,in_c, in_h, in_w]
        a_mat = A.reshape([self.batch, -1])
        B = np.dot(a_mat, np.transpose(W, (1,0)))
        return B
    