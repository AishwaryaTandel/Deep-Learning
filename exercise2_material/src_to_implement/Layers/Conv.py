import numpy as np
import math
from scipy.signal import correlate2d, convolve2d
from .Base import BaseLayer
import copy

class Conv(BaseLayer):

    def __init__(self , stride_shape , convolution_shape , num_kernels):
        super().__init__()

        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_filters = num_kernels
        self.trainable = True

        self.weight_shape_list = list(self.convolution_shape)
        self.weight_shape_list.insert(0 ,  self.num_filters)

        self.weights = np.random.uniform(low = 0.0 , high = 1.0 , size = tuple(self.weight_shape_list))
        self.bias = np.random.uniform(low = 0.0 , high = 1.0 , size = (num_kernels , 1))
        self.input_tensor = None
        self.gradient_weights = None
        self.gradient_bias = None
        self.optimizer = None

    @property
    def gradient_weights(self):
        return self.__gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, val):
        self.__gradient_weights = val

    @property
    def gradient_bias(self):
        return self.__gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, val):
        self.__gradient_bias = val

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, val):
        self.__optimizer = val

    def initialize(self, weights_initializers, bias_initializers):
        fan_in = None
        fan_out = None
        if len(self.convolution_shape) == 2:    #2-Dimensional
            fan_in = self.convolution_shape[0] * self.convolution_shape[1]
            fan_out = self.num_filters * self.convolution_shape[1]

        elif len(self.convolution_shape) == 3:  #3-Dimensional
            fan_in = self.convolution_shape[0] * self.convolution_shape[1] * self.convolution_shape[2]
            fan_out = self.num_filters * self.convolution_shape[1] * self.convolution_shape[2]

        self.weights = weights_initializers.initialize(tuple(self.weight_shape_list), fan_in, fan_out)
        self.bias = bias_initializers.initialize((self.num_filters), fan_in, fan_out)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = None
        xStride = 1
        yStride = 1

        if (len(self.stride_shape) == 1) and (len(input_tensor.shape) > 3):
            xStride = self.stride_shape[0]
            yStride = self.stride_shape[0]

        elif len(self.stride_shape) == 2:
            xStride = self.stride_shape[0]
            yStride = self.stride_shape[1]

        elif len(input_tensor.shape) == 3:
            xStride = self.stride_shape[0]

        if len(input_tensor.shape) == 4:
            output_tensor = np.zeros((input_tensor.shape[0], self.num_filters, input_tensor.shape[2], input_tensor.shape[3]))

        elif len(input_tensor.shape) == 3:
            output_tensor = np.zeros((input_tensor.shape[0], self.num_filters, input_tensor.shape[2]))

        for batch in range(input_tensor.shape[0]):
            for filter_no in range(self.num_filters):
                filterNew = self.weights[filter_no]
                biasNew = self.bias[filter_no]

                if len(input_tensor.shape) == 3:
                    output_tensor[batch, filter_no] = np.sum(correlate2d(input_tensor[batch], filterNew, mode='same')) + biasNew
                else:
                    output_channel = 0
                    for channel_num in range(input_tensor.shape[1]):
                        output_channel = output_channel + correlate2d(input_tensor[batch, channel_num], filterNew[channel_num], mode='same')
                    output_tensor[batch, filter_no, :, :] = output_channel + biasNew

        final_tensor = output_tensor.copy()
        if len(input_tensor.shape) == 3:
            final_tensor = final_tensor[:, :, ::xStride]
        elif len(input_tensor.shape) == 4:
            final_tensor = final_tensor[:, :, ::xStride, ::yStride]
        return final_tensor

    def backward(self, error_tensor):
        if len(self.input_tensor.shape) == 3:
            input_tensor = np.expand_dims(self.input_tensor, axis=2)
            error_tensor = np.expand_dims(error_tensor, axis=2)
            weights = np.expand_dims(self.weights, axis=2)
            convolution_shape = tuple([self.convolution_shape[0], 1, self.convolution_shape[1]])
            xStride, yStride = 1, self.stride_shape[0]

        else:
            input_tensor = self.input_tensor
            weights = self.weights
            convolution_shape = self.convolution_shape
            xStride, yStride = self.stride_shape

        data_num, channel_num, X, Y = input_tensor.shape
        input_gradient = np.zeros(input_tensor.shape)
        new_weights = []

        for i in range(channel_num):
            new_weights.append((weights[:, i]))
        new_weights = np.array(new_weights)
        error_tensor_upsampled = np.zeros((data_num , self.num_filters, X, Y))

        for batch in range(data_num):
            for filter_no in range(self.num_filters):
                for i in range(error_tensor.shape[2]):
                    for j in range(error_tensor.shape[3]):
                        error_tensor_upsampled[batch, filter_no, i * xStride, j * yStride] = error_tensor[batch, filter_no, i,j]

        for batch in range(data_num):
            for channel in range(channel_num):
                for filter_no in range(self.num_filters):
                    input_gradient[batch, channel] += convolve2d(error_tensor_upsampled[batch, filter_no], new_weights[channel, filter_no], mode='same', fillvalue=0.0)

        if input_gradient.shape[2] == 1:
            input_gradient = np.squeeze(input_gradient, axis=2)

        self.gradient_weights = np.zeros(weights.shape)

        conv_x, conv_y = convolution_shape[1:]

        for batch in range(data_num):
            for channel in range(channel_num):
                padding_x = (conv_x - 1)
                padding_y = (conv_y - 1)

                if padding_x % 2 == 0:
                    padding_x_left = padding_x // 2
                    padding_x_right = padding_x // 2
                else:
                    padding_x_left = padding_x // 2
                    padding_x_right = int(math.ceil(padding_x / 2))

                if padding_y % 2 == 0:
                    padding_y_up = padding_y // 2
                    padding_y_down = padding_y // 2
                else:
                    padding_y_up = padding_y // 2
                    padding_y_down = int(math.ceil(padding_y / 2))

                input_padding = np.pad(input_tensor[batch, channel],((padding_x_left, padding_x_right), (padding_y_up, padding_y_down)))

                for filter_no in range(self.num_filters):
                    self.gradient_weights[filter_no, channel] += correlate2d(input_padding, error_tensor_upsampled[batch, filter_no], mode= 'valid')
        self.gradient_bias = np.zeros((self.bias.shape))

        for batch in range(error_tensor.shape[0]):
            for filter_no in range(error_tensor.shape[1]):
                self.gradient_bias[filter_no] += np.sum(error_tensor[batch, filter_no])

        if self.optimizer!= None:
            weights_optimizer = copy.deepcopy(self.optimizer)
            bias_optimizer = copy.deepcopy(self.optimizer)

            self.weights = weights_optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        return input_gradient