from Layers.Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.input_tensor_shape = 0

    def forward(self, input_tensor):
        self.input_tensor_shape = input_tensor.shape
        a,b,c,d = input_tensor.shape
        input_tensor = input_tensor.reshape(a, b*c*d)
        return input_tensor

    def backward(self, output_tensor):
        a,b,c,d = self.input_tensor_shape
        output_tensor = output_tensor.reshape(a, b, c, d)
        return output_tensor