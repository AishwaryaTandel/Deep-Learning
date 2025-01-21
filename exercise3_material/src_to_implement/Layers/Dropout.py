import numpy as np

class Dropout:
    def __init__(self , probability):
        self.probability = probability
        self.D1 = None
        self.trainable = False
        self.testing_phase = False
        self.optimizer = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        if self.testing_phase == False:
            self.D1 = np.random.binomial([np.ones((self.input_tensor.shape[0], self.input_tensor.shape[1]))], self.probability)[0]
            self.input_tensor = np.multiply(input_tensor, self.D1)
            self.input_tensor = self.input_tensor / self.probability
        return self.input_tensor

    def backward(self, error_tensor):
        val = self.D1 / (1 - self.probability)
        error_tensor = np.multiply(val, error_tensor)
        return error_tensor



