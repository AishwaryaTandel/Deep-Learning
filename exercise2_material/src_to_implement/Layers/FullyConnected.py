from Layers.Base import BaseLayer
import numpy

class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size
        self.weights = numpy.random.uniform(0, 1, (self.input_size + 1, self.output_size))
        self.optimizer = None
        self.gradient_weights = None
        self.input_tensor = None

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self.__optimizer = opt

    def forward(self, input_tensor):
        batch_size, inputs = input_tensor.shape
        bias = numpy.ones((batch_size, 1), dtype=int)
        self.input_tensor = numpy.concatenate((input_tensor, bias), axis=1)     ## Append bias layer to input
        output_tensor = numpy.dot(self.input_tensor, self.weights)              ## calculating dot product of input and weights

        return output_tensor

    def backward(self, error_tensor):
        updated_error_tensor = numpy.dot(error_tensor, self.weights.T)[:, :-1]      ## updated error tensor and excluding bias term
        self.gradient_weights = numpy.dot(self.input_tensor.T, error_tensor)        ## weight update
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return updated_error_tensor

    @property
    def gradient_weights(self):
        return self.__gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, weight_value):
        self.__gradient_weights = weight_value

    def initialize(self, weights_initializers, bias_initializers):
        fan_in = self.input_size
        fan_out = self.output_size
        self.weights[:-1] = weights_initializers.initialize((self.input_size, self.output_size), fan_in, fan_out)
        self.weights[-1] = bias_initializers.initialize((1, self.output_size), fan_in, fan_out)