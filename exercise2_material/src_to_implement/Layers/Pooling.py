import numpy as np
import math
from Layers.Base import BaseLayer

class Pooling(BaseLayer):

    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_tensor = None
        self.masked_tensor = None
        self.output_tensor = None
        self.trainable = False

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        pooling_x = self.pooling_shape[0]
        pooling_y = self.pooling_shape[1]
        stride_along_x = 1
        stride_along_y = 1
        if len(self.stride_shape) == 1:
            stride_along_x = self.stride_shape
            stride_along_y = self.stride_shape

        else:
            stride_along_x, stride_along_y = self.stride_shape

        x_shape = math.floor((input_tensor.shape[2] - self.pooling_shape[0]) / stride_along_x) + 1
        y_shape = math.floor((input_tensor.shape[3] - self.pooling_shape[1]) / stride_along_y) + 1
        self.output_tensor = np.zeros((input_tensor.shape[0], input_tensor.shape[1], x_shape, y_shape))
        self.masked_tensor = np.zeros(self.output_tensor.shape)

        for batch in range(input_tensor.shape[0]):
            for channel in range(input_tensor.shape[1]):
                for i, a1 in enumerate(range(0, input_tensor.shape[2], stride_along_x)):
                    if a1 + pooling_x > input_tensor.shape[2]:
                        break

                    for j, b1 in enumerate(range(0, input_tensor.shape[3], stride_along_y)):
                        if b1 + pooling_y > input_tensor.shape[3]:
                            break

                        max = np.max(input_tensor[batch, channel, a1: a1 + pooling_x, b1: b1 + pooling_y])
                        self.output_tensor[batch, channel, i, j] = max
                        self.masked_tensor[batch, channel, i, j] = np.argmax(
                            input_tensor[batch, channel, a1: a1 + pooling_x, b1: b1 + pooling_y])
        return self.output_tensor

    def backward(self, error_tensor):
        gradient_output = np.zeros((self.input_tensor.shape))
        pooling_along_x = self.pooling_shape[0]
        pooling_along_y = self.pooling_shape[1]
        stride_along_x = 1
        stride_along_y = 1
        if len(self.stride_shape) == 1:
            stride_along_x = self.stride_shape
            stride_along_y = self.stride_shape
        else:
            stride_along_x, stride_along_y = self.stride_shape

        for batch in range(error_tensor.shape[0]):
            for channel in range(self.input_tensor.shape[1]):
                for i, a1 in enumerate(range(0, self.input_tensor.shape[2], stride_along_x)):
                    if a1 + pooling_along_x > self.input_tensor.shape[2]:
                        break
                    for j, b1 in enumerate(range(0, self.input_tensor.shape[3], stride_along_y)):
                        if b1 + pooling_along_y > self.input_tensor.shape[3]:
                            break

                        error_tensor_value = error_tensor[batch, channel, i, j]
                        max_error_index = np.unravel_index(int(self.masked_tensor[batch, channel, i, j]), gradient_output[batch, channel, a1: a1 + pooling_along_x, b1: b1 + pooling_along_y].shape)
                        gradient_output[batch, channel, a1: a1 + pooling_along_x, b1: b1 + pooling_along_y][max_error_index] += error_tensor_value
        return gradient_output
