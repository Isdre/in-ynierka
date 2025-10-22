import math

def linear_activation(x):
    return x

def sigmoid_activation(x):
    return 1 / (1 + math.exp(-x))

def tanh_activation(x):
    return math.tanh(x)

def relu_activation(x):
    return max(0, x)


class ActivationFunction:

    activation_functions = {
        'linear': linear_activation,
        'sigmoid': sigmoid_activation,
        'tanh': tanh_activation,
        'relu': relu_activation,
    }

    def __init__(self, name):
        if name not in self.activation_functions:
            raise ValueError(f"Activation function '{name}' is not defined.")
        self.name = name
        self.function = self.activation_functions[name]
