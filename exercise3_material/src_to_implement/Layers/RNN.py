from copy import deepcopy
from .FullyConnected import *
from .TanH import *
from .Sigmoid import *

class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.trainable = True
        self.l1 = []
        self.l2 = []
        self.l3 = []
        self.l4 = []
        self.l5 = []
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(hidden_size)
        self._memorize = False
        self._optimizer = None
        self.fullyConnected1 = FullyConnected(self.input_size + self.hidden_size, self.hidden_size)
        self.fullyConnected2 = FullyConnected(self.hidden_size, self.output_size)
        self.Tanh_layer = TanH()
        self.Sigmoid_layer = Sigmoid()

    def initialize(self, weights_initializer, bias_initializer):
        self.fullyConnected1.initialize(weights_initializer, bias_initializer)
        self.fullyConnected2.initialize(weights_initializer, bias_initializer)

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, value):
        self._memorize = value

    @property
    def weights(self):
        return self.fullyConnected1.weights

    @weights.setter
    def weights(self, value):
        self.fullyConnected1.weights = value

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        time_step = input_tensor.shape[0]
        output_tensor = np.zeros([self.input_tensor.shape[0], self.output_size])
        if not self._memorize:
            self.hidden_state = np.zeros(self.hidden_size)
        for time in range(time_step):
            current_input_tensor = input_tensor[time]

            concatenation = np.concatenate([current_input_tensor, self.hidden_state])
            concatenation = np.array([concatenation])
            self.concatenation = concatenation

            self.l1.append(self.concatenation)
            result1 = self.fullyConnected1.forward(concatenation)
            self.result1 = result1

            self.l2.append(self.fullyConnected1.xt)
            result2 = self.Tanh_layer.forward(self.result1)
            self.result2 = result2

            self.l3.append(self.Tanh_layer.back_x)
            self.hidden_state = result2

            result3 = self.fullyConnected2.forward(self.hidden_state)
            self.result3 = result3
            self.l4.append(self.fullyConnected2.xt)
            result4 = self.Sigmoid_layer.forward(self.result3)
            self.result4 = result4
            self.l5.append(self.Sigmoid_layer.back_x)
            self.hidden_state = self.hidden_state.flatten()
            output_tensor[time] = self.result4
        return output_tensor

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = deepcopy(optimizer)
        self._optimizerbias = deepcopy(optimizer)

    @property
    def gradient_weights(self):
        return self.fullyConnected1.gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, value):
        self.fullyConnected1.gradient_weights = value
        self._gradient_weights = value


    def calculate_regularization_loss(self):
        if self.trainable is True:
            if self.optimizer.regularizer is not None:
                self.regulation_loss = self.optimizer.regularizer.norm(self.weights)
        return self.regulation_loss

    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        time_step = error_tensor.shape[0]
        output_tensor = np.zeros([time_step, self.input_size])
        self.gradiant_weights_fcn2 = np.zeros(self.fullyConnected2.weights.shape)
        self.gradiant_weights_fcn1 = np.zeros(self.fullyConnected1.weights.shape)
        hidden_error = 0

        for time in reversed(range(time_step)):
            current_error_tensor = error_tensor[time]
            self.Sigmoid_layer.back_x = self.l5[time]
            self.fullyConnected2.xt = self.l4[time]
            self.Tanh_layer.back_x = self.l3[time]
            self.fullyConnected1.xt = self.l2[time]

            error4 = self.Sigmoid_layer.backward(current_error_tensor)
            error3 = self.fullyConnected2.backward(error4)
            error3 = error3 + hidden_error
            error2 = self.Tanh_layer.backward(error3)
            error1 = self.fullyConnected1.backward(error2)

            self.gradiant_weights_fcn2 += self.fullyConnected2.gradient_weights
            self.gradiant_weights_fcn1 += self.fullyConnected1.gradient_weights
            output_tensor[time] = error1[:, : self.input_size]
            hidden_error = error1[:, self.input_size:]

        self.gradient_weights = self.gradiant_weights_fcn1

        if self._optimizer is not None:
            self.fullyConnected2.weights = self._optimizer.calculate_update(self.fullyConnected2.weights, self.gradiant_weights_fcn2)
            self.fullyConnected1.weights = self._optimizer.calculate_update(self.fullyConnected1.weights, self.gradiant_weights_fcn1)
        return output_tensor