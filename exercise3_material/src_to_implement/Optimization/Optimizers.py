import numpy as np

class Optimizer():
    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer

class Sgd(Optimizer):
    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate=learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer!= None:
            weight_tensor = weight_tensor - self.learning_rate * gradient_tensor-(self.learning_rate * self.regularizer.calculate_gradient(weight_tensor))
        else:
            weight_tensor = weight_tensor - self.learning_rate * gradient_tensor
        return weight_tensor

class SgdWithMomentum(Optimizer):
    def __init__(self, learning_rate, momentum_rate):
        super().__init__()
        self.learning_rate=learning_rate
        self.momentum_rate=momentum_rate
        self.velocity =0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.velocity = self.velocity*self.momentum_rate-self.learning_rate*gradient_tensor
        if self.regularizer!= None:
            weight_tensor = self.velocity + weight_tensor - (self.learning_rate * self.regularizer.calculate_gradient(weight_tensor))
        else:
            weight_tensor = self.velocity + weight_tensor

        return weight_tensor

class Adam(Optimizer):
    def __init__(self, learning_rate, mu, rho):
        super().__init__()
        self.learning_rate = learning_rate
        self.beta_1 = mu
        self.beta_2 = rho
        self.velocity = 0
        self.r = 0
        self.call = 1

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.epsilon = 2**-52
        self.velocity = self.beta_1*self.velocity+(1. - self.beta_1)*gradient_tensor
        self.r = self.beta_2*self.r+(1. - self.beta_2)*gradient_tensor**2
        velocity_grad = self.velocity/(1. - self.beta_1**(self.call))
        r_gradient = self.r/(1. - self.beta_2 ** (self.call))
        if self.regularizer!= None:
            weight_tensor = weight_tensor - (self.learning_rate / (np.sqrt((r_gradient) + self.epsilon)) * velocity_grad) - (self.learning_rate * self.regularizer.calculate_gradient(weight_tensor))
        else:
            weight_tensor = weight_tensor - (self.learning_rate / (np.sqrt((r_gradient) + self.epsilon)) * velocity_grad)
        self.call += 1
        return weight_tensor



