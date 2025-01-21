import numpy as np

class ReLU:
    def __init__(self):
        self.trainable=False
        self.optimizer=None

    def forward(self,input_tensor):
        self.temp=np.maximum(0,input_tensor)
        return self.temp

    def backward(self,error_tensor):

        error_tensor[self.temp<=0]=0#####or we can take input tensor
        return error_tensor
        # y=(error_tensor > 0).astype(error_tensor.dtype)
        # return y*error_tensor



