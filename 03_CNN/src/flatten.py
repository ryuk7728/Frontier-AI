import numpy as np
from .layer import Layer

class Flatten(Layer):

    def __init__(self,input_shape):
        self.input_shape = input_shape

    def forward(self,input):
        return input.reshape(-1,1)
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient.reshape(self.input_shape)
