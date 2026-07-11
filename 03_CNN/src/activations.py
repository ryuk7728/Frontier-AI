import numpy as np
from src.layer import Layer

class ReLU(Layer):

    def forward(self,X):
        self.input = X
        mod_X = np.maximum(0,X)
        return mod_X
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient * (self.input>0)
    
class Softmax(Layer):

    def forward(self,X):
        shifted = X - np.max(X,axis=0,keepdims=True)
        exp = np.exp(shifted)
        sftmx = exp/np.sum(exp,axis=0,keepdims=True)
        return sftmx
    
    def backward(self, output_gradient, learning_rate):
        return output_gradient
        

