import numpy as np
import matplotlib.pyplot as plt
from .inference import load_weights
from .model import NN
from .utils import relu_dash



def dream(steps=100,learning_rate=0.01,no=0):

    X = np.random.uniform(0,1,(28,28))
    X[:,0:8] = 0
    X[:,22:] = 0
    X[0:4,:] = 0
    X[24:,:] = 0

    X = X.reshape(784,1)
    
    model = NN()
    model.W1,model.b1,model.W2,model.b2 = load_weights()
    actives = []
    for _ in range(steps):
        _,A2,Z1 = model.predict(X)
        dX = (model.W1.T)@(relu_dash(Z1)*(model.W2[no].reshape(10,1)))
        X += learning_rate*dX
        X = np.clip(X,0,1)

        X = X.reshape((28,28))

        X[X < 0.2] = 0

        X[:,0:8] = 0
        X[:,22:] = 0
        X[0:4,:] = 0
        X[24:,:] = 0

        X = X.reshape((784,1))

        actives.append(A2[no])
        print(f"The activation is {A2[no]}")
    
    return actives,X


if __name__ == "__main__":
    actives, X = dream(1000,0.01,3)
    plt.imshow(X.reshape(28,28))
    plt.show()
