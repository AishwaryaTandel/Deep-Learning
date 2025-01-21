import copy
from Layers import Base

class NeuralNetwork(Base.BaseLayer):
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        super().__init__()
        self.Optimizer = optimizer
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = []
        self.layers = []
        self. data_layer = None
        self.loss_layer = None
        self._testing_phase = False
        self.regularizer = None
        self.regular_loss = 0

    def forward(self):
        self.regular_loss = 0
        self.input_tensor, self.label_tensor = self.data_layer.next()
        for i in self.layers:
            i.testing_phase = False
            self.input_tensor = i.forward(self.input_tensor)
            if i.optimizer is not None:
                if i.optimizer.regularizer is not None:
                    self.regular_loss += i.calculate_regularization_loss()
        return self.loss_layer.forward(self.input_tensor, self.label_tensor) + self.regular_loss

    def backward(self):
        error = self.loss_layer.backward(self.label_tensor)
        temporary_list = self.layers.copy()
        temporary_list.reverse()
        for x in range(len(temporary_list)):
            error = temporary_list[x].backward(error)
        return error

    def append_layer(self, layer):
        if layer.trainable:
            self.layers.append(layer)
            layer.optimizer=copy.deepcopy(self.Optimizer)
            layer.initialize(self.weights_initializer, self.bias_initializer)
            self.weights = layer.initialize(self.weights_initializer, self.bias_initializer)
        else:
            self.layers.append(layer)

    @property
    def phase(self):
        return self._testing_phase

    @phase.setter
    def phase(self, value):
        self._testing_phase = value

    def train(self, iterations):
        for it in range(iterations):
            self.testing_phase = False
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        self.input_tensor = input_tensor
        for layer in self.layers:
            layer.testing_phase = True
            self.input_tensor = layer.forward(self.input_tensor)
        return self.input_tensor