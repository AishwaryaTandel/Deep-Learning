import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.loss = None
        self.prediction_tensor = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        epsilon = np.finfo(float).eps
        mask = (label_tensor == 1)
        loss = -(np.sum(np.log(prediction_tensor[mask] + epsilon)))
        return loss

    def backward(self, label_tensor):
        epsilon = np.finfo(float).eps
        result = - (label_tensor/(self.prediction_tensor + epsilon))
        return result