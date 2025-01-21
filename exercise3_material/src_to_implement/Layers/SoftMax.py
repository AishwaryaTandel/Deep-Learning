import numpy as np


class SoftMax:
    def __init__(self):
        self.trainable = False
        self.predicted = None
        self.optimizer = None

    def forward(self, input_tensor):
        # self.predicted=np.exp(input_tensor)/np.exp(input_tensor).sum(axis=1)[:None]
        # self.predicted=exp_input_tensor / exp_input_tensor.sum(axis=1, keepdims=True)
        input_tensor = np.exp(input_tensor - np.max(input_tensor))
        self.predicted = input_tensor / input_tensor.sum(axis=1, keepdims=True)
        return self.predicted

    def backward(self, error_tensor):
        x = self.predicted * (
                    error_tensor - np.sum(error_tensor * self.predicted, axis=1).reshape(error_tensor.shape[0], 1))
        return x



