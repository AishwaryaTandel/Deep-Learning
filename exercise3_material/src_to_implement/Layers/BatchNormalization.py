import copy
from .Base import BaseLayer
from .Helpers import *

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.mu = 0
        self.sigma = 0
        self.weights = np.ones((1, self.channels))
        self.bias = np.zeros((1, self.channels))
        self._optimizer = None
        self._optimizerbias = None

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones((1, self.channels))
        self.bias = np.zeros((1, self.channels))

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if len(input_tensor.shape) == 4:
            self.input_tensor_new = self.reformat(self.input_tensor)
        else:
            self.input_tensor_new = self.input_tensor

        if not self.testing_phase:
            self.mu_b = np.mean(self.input_tensor_new, axis=0)
            self.sigma_b = np.std(self.input_tensor_new, axis=0)
            self.x_estimate = (self.input_tensor_new - self.mu_b) / (self.sigma_b ** 2 + np.finfo(float).eps) ** 0.5
            self.y_estimate = self.weights * self.x_estimate + self.bias
            alpha = 0.8
            self.mu = alpha * self.mu + (1 - alpha) * self.mu_b
            self.sigma = alpha * self.sigma + (1 - alpha) * self.sigma_b
        else:
            self.x_estimate = (self.input_tensor_new - self.mu) / (self.sigma ** 2 + np.finfo(float).eps) ** 0.5
            self.y_estimate = self.weights * self.x_estimate + self.bias

        if len(self.input_tensor.shape) == 4:
            self.y_estimate = self.reformat(self.y_estimate)

        return self.y_estimate

    @staticmethod
    def compute_bn_gradients(error_tensor, input_tensor, weights, mean, var, eps=np.finfo(float).eps):
        normalized_mean = input_tensor - mean
        var_eps = var + eps

        gamma_error = error_tensor * weights
        inv_batch = 1. / error_tensor.shape[0]

        grad_var = np.sum(normalized_mean * gamma_error * -0.5 * (var_eps ** (-3 / 2)), keepdims=True, axis=0)

        sqrt_var = np.sqrt(var_eps)
        first = gamma_error * 1. / sqrt_var

        grad_mu2 = (grad_var * np.sum(-2. * normalized_mean, keepdims=True, axis=0)) * inv_batch
        grad_mu1 = np.sum(gamma_error * -1. / sqrt_var, keepdims=True, axis=0)

        second = grad_var * (2. * normalized_mean) * inv_batch
        grad_mu = grad_mu2 + grad_mu1
        return first + second + inv_batch * grad_mu

    def backward(self, error_tensor):
        if len(error_tensor.shape) == 4:
            self.error_tensor = self.reformat(error_tensor)
        else:
            self.error_tensor = np.reshape(error_tensor, self.x_estimate.shape)

        gradient_wt = np.sum(self.error_tensor * self.x_estimate, axis=0)
        self.gradient_weights = np.reshape(gradient_wt, (1, self.channels))

        gradient_bias = np.sum(self.error_tensor, axis=0)
        self.gradient_bias = np.reshape(gradient_bias, (1, self.channels))
        self.gradient_input = compute_bn_gradients(self.error_tensor, self.input_tensor_new, self.weights, self.mu_b, self.sigma_b ** 2, np.finfo(float).eps)

        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
        if self._optimizerbias is not None:
            self.bias = self._optimizerbias.calculate_update(self.bias, self.gradient_bias)
        if len(error_tensor.shape) == 4:
            self.gradient_input = self.reformat(self.gradient_input)
        return self.gradient_input

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = copy.deepcopy(value)
        self._optimizerbias = copy.deepcopy(value)

    def reformat(self, tensor):
        if len(tensor.shape) == 4:
            batch = tensor.shape[0]
            channel = tensor.shape[1]
            height = tensor.shape[2]
            width = tensor.shape[3]
            tensor2 = np.reshape(tensor, (batch, channel, height * width))
            tensor2 = np.transpose(tensor2, (0, 2, 1))
            tensor2 = np.reshape(tensor2, (batch * height * width, channel))
        else:
            batch = self.input_tensor.shape[0]
            channel = self.input_tensor.shape[1]
            height = self.input_tensor.shape[2]
            width = self.input_tensor.shape[3]
            tensor2 = np.reshape(tensor, (batch, height * width, channel))
            tensor2 = np.transpose(tensor2, (0, 2, 1))
            tensor2 = np.reshape(tensor2, (batch, channel, height, width))
        return tensor2

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self._gradient_weights = value

    @property
    def gradient_bias(self):
        return self._gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, value):
        self._gradient_bias = value

    def calculate_regularization_loss(self):
        self.regulation_loss = 0
        if self.trainable is True:
            if self.optimizer.regularizer is not None:
                self.regulation_loss = self.optimizer.regularizer.norm(self.weights)
        return self.regulation_loss


