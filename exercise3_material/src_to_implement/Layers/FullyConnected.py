import numpy as np
from .Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        super(FullyConnected, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.trainable = True
        self.weights = np.random.random((self.input_size + 1, self.output_size)) #already has bias
        self.weights1 = np.delete(self.weights, (-1), axis=0)
        self._optimizer = None
        self.input_tensor = None

    def initialize(self, weights_initializer, bias_initializer): # these two going to defined at some place
        weights = weights_initializer.initialize((self.input_size, self.output_size), self.input_size, self.output_size)#xavier.initialize
        bias = bias_initializer.initialize((1, self.output_size), self.input_size, self.output_size)
        # self.weights1 = weights
        self.weights = np.concatenate((weights, bias), axis=0)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    def forward(self, input_tensor):
        bias = np.ones((input_tensor.shape[0], 1))
        self.xt = np.concatenate((input_tensor, bias), axis=1)
        output_tensor = np.dot(self.xt, self.weights)
        return output_tensor

    def backward(self, error_tensor):
        present_error_tensor = error_tensor
        E_n_minus_1 = np.dot(present_error_tensor, self.weights.T)
        self._gradient_weights = np.dot(self.xt.T, present_error_tensor)
        if self._optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self._gradient_weights)
        return E_n_minus_1[:, :-1]

    def calculate_regularization_loss(self):
        self.regulation_loss = 0
        if self.trainable is True:
            if self.optimizer.regularizer is not None:
                self.regulation_loss = self.optimizer.regularizer.norm(self.weights)

        return self.regulation_loss

    # call optimizer.calculate_update
    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

