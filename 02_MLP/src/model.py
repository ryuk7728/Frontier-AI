import numpy as np
from .utils import softmax,relu

class NN:

    def __init__(self):

        self.W1 = np.random.uniform(-0.2,0.2,(10,784))
        self.b1 = np.zeros((10,1))
        self.W2 = np.random.uniform(-0.2,0.2,(10,10))
        self.b2 = np.zeros((10,1))

    def predict(self,X):

        Z1 = self.W1 @ X + self.b1
        A1 = relu(Z1)
        Z2 = self.W2 @ A1 + self.b2
        A2 = softmax(Z2)

        return A1,A2,Z1
    
    def infer(self,X):
        
        _,Y,_ = self.predict(X)
        return np.argmax(Y)
    
    