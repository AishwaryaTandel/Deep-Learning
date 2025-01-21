import copy
class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.input_tensor = None
        self.label_tensor = None
        self.forward_output = None

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.input_tensor = input_tensor
        self.label_tensor = label_tensor
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        self.forward_output = self.loss_layer.forward(input_tensor, self.label_tensor)
        return self.forward_output


    def backward(self):
        backward_output = self.loss_layer.backward(self.label_tensor)
        for layer in self.layers[::-1]:
            backward_output = layer.backward(backward_output)


    def append_layer(self, layer):
        if (layer.trainable):
            deep_copy_optimizer = copy.deepcopy(self.optimizer)
            layer.optimizer = deep_copy_optimizer
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            loss_result = self.forward()
            self.loss.append(loss_result)
            self.backward()

    def test(self, input_tensor):
        for layer in self.layers:
            prediction = layer.forward(input_tensor)
            input_tensor = prediction
        return prediction

