import numpy as np

def relu(x):
    x_shape = x.shape
    x = np.reshape(x,[-1])
    x = [max(v,0) for v in x]
    x = np.reshape(x, x_shape)
    return x

def leaky_relu(x):
    x_shape = x.shape
    x = np.reshape(x, [-1])
    x = [max(0.1*v,v) for v in x]
    x = np.reshape(x, x_shape)
    return x

def sigmoid(x):
    x_shape = x.shape
    x = np.reshape(x,[-1])
    x = [ 1 / (1 + np.exp(-v)) for v in x]
    x = np.reshape(x, x_shape)
    return x

def tanh(x):
    x_shape = x.shape
    x = np.reshape(x, [-1])
    x = [np.tanh(v) for v in x]
    x = np.reshape(x,x_shape)
    return x