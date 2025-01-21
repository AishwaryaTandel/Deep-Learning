import numpy as np

class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = (self.momentum_rate * self.v) - (self.learning_rate * gradient_tensor)
        weight_tensor = weight_tensor + self.v
        return weight_tensor

class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.counter = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = (self.mu * self.v) + ((1 - self.mu) * gradient_tensor)
        self.r = (self.rho * self.r) + ((1 - self.rho) * (np.multiply(gradient_tensor, gradient_tensor)))
        v_prime = self.v / (1 - (self.mu ** self.counter))
        r_prime = self.r / (1 - (self.rho ** self.counter))
        weight_tensor = weight_tensor - (self.learning_rate * (v_prime / ((np.sqrt(r_prime) + np.finfo(float).eps))))
        self.counter = self.counter + 1
        return weight_tensor

class Sgd:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        weight_tensor = weight_tensor - (self.learning_rate * gradient_tensor)
        return weight_tensor