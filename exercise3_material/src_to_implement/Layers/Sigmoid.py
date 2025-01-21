import numpy as np
from .Base import BaseLayer

class Sigmoid(BaseLayer):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.trainable = False

    def forward(self, input_tensor):
        self.sigmoid = 1 / (1 + np.exp(-input_tensor))
        self.back_x = self.sigmoid * (1 - self.sigmoid)
        return self.sigmoid

    def backward(self, error_tensor):
        error_tensor = error_tensor * self.back_x
        return error_tensor

