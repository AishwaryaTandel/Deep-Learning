from Layers.Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_tensor = None
        self.output_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        max_val = np.max(input_tensor, axis=1, keepdims=True)
        exponent = np.exp(input_tensor - max_val)
        sum_of_all_exp = np.sum(exponent, axis=1, keepdims=True)
        self.output_tensor = exponent / sum_of_all_exp
        return self.output_tensor

    def backward(self, error_tensor):
        sum_error_output = np.sum(error_tensor * self.output_tensor, axis=1)
        term = error_tensor - sum_error_output.reshape(-1, 1)
        return self.output_tensor * term