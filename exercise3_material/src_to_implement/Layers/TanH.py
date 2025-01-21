import numpy as np
from .Base import BaseLayer

class TanH(BaseLayer):
    def __init__(self):
        super(TanH, self).__init__()
        self.trainable = False

    def forward(self, input_tensor):
        self.tan = (np.exp(input_tensor) - np.exp(-input_tensor)) / (np.exp(input_tensor) + np.exp(-input_tensor))
        self.back_x = 1 - np.square(self.tan)
        return self.tan

    def backward(self, error_tensor):
        error_tensor = self.back_x * error_tensor
        return error_tensor

