import numpy as np
from scipy import signal
from .layer import Layer

class Convolutional(Layer):

    def __init__(self,input_shape,kernel_size,depth):
        self.depth = depth
        self.input_depth = input_shape[0]
        self.input_shape = input_shape
        self.kernel_shape = (self.depth,self.input_depth,kernel_size,kernel_size)

        self.output_shape = (self.depth,input_shape[1]-kernel_size+1,input_shape[1]-kernel_size+1)
        self.kernels = np.random.randn(*self.kernel_shape) * np.sqrt(2/kernel_size**2)
        self.biases = np.zeros(self.output_shape)

    def forward(self,X):
        self.input = X
        output = np.zeros(self.output_shape)
        for i in range(self.depth):
            for j in range(self.input_depth):
                output[i] += signal.correlate2d(X[j],self.kernels[i][j],mode = "valid")
        return output
    
    def backward(self, output_gradient, learning_rate):
        kernel_gradient = np.zeros(self.kernel_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernel_gradient[i][j] = signal.correlate2d(self.input[j],output_gradient[i],mode="valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i],self.kernels[i][j],mode="full")

        self.kernels -= learning_rate * kernel_gradient
        self.biases -= learning_rate * output_gradient

        return input_gradient





