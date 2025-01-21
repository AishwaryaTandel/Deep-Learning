import numpy as np

class Flatten:
    def __init__(self):
        self.trainable=False
        self.optimizer = None
    def forward(self,input_tensor):
        self.shape=input_tensor.shape
        # x=input_tensor.flatten(order='C')
        # x.reshape(x.shape[0], 1)
        return np.ravel(input_tensor).reshape(input_tensor.shape[0],-1)

    def backward(self,error_tensor):
        return error_tensor.reshape(self.shape)