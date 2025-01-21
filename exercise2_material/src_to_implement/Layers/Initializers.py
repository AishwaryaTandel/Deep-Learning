import numpy as np
import math

class Constant:
    def __init__(self, constant = 0.1):
        self.constant = constant

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        return np.ones(weights_shape) * self.constant

class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        return np.random.uniform(0.0, 1.0, size=weights_shape)

class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        variance = math.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0.0, variance, size=weights_shape)

class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        variance = math.sqrt(2 / fan_in)
        weights_tensor = np.random.normal(0.0, variance, size=weights_shape)
        return weights_tensor