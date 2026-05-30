import numpy as np
import matplotlib.pyplot as plt
from .inference import load_weights
from .model import NN
from .utils import relu_dash


def apply_digit_priors(X):
    X[X < 0.2] = 0

    X[:,0:8] = 0
    X[:,22:] = 0
    X[0:4,:] = 0
    X[24:,:] = 0

    return X


def dream(steps=100,learning_rate=0.01,no=0,model=None,verbose=True):

    X = np.random.uniform(0,1,(28,28))
    X = apply_digit_priors(X)

    X = X.reshape(784,1)
    
    if model is None:
        model = NN()
        model.W1,model.b1,model.W2,model.b2 = load_weights()

    actives = []
    for _ in range(steps):
        _,A2,Z1 = model.predict(X)
        dX = (model.W1.T)@(relu_dash(Z1)*(model.W2[no].reshape(10,1)))
        X += learning_rate*dX
        X = np.clip(X,0,1)

        X = X.reshape((28,28))

        X = apply_digit_priors(X)

        X = X.reshape((784,1))

        actives.append(float(A2[no,0]))
        if verbose:
            print(f"The activation is {A2[no]}")
    
    return actives,X


def dream_until(model,no,target_prob=0.99,max_steps=400,learning_rate=0.02,attempts=20):
    best_actives,best_X = [],None

    for _ in range(attempts):
        actives,X = dream(max_steps,learning_rate,no,model=model,verbose=False)
        if max(actives) >= target_prob:
            return actives,X
        if not best_actives or max(actives) > max(best_actives):
            best_actives,best_X = actives,X

    return best_actives,best_X


if __name__ == "__main__":
    actives, X = dream(1000,0.01,5)
    plt.imshow(X.reshape(28,28))
    plt.show()
