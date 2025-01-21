from Layers.Base import BaseLayer
import numpy

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return numpy.maximum(input_tensor, 0)

    def backward(self, error_tensor):
        self.mask = self.input_tensor > 0
        return self.mask * error_tensor