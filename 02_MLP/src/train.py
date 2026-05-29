from .model import NN
import numpy as np
from .utils import relu_dash,get_accuracy


def train(model :NN,X_train,y_train,learning_rate=0.7,epochs=500):
    losses = []
    m = X_train.shape[1]

    for i in range(epochs):

        A1, y_pred,Z1 = model.predict(X_train)

        dZ2 = y_pred - y_train
        accuracy = get_accuracy(y_pred,y_train)
        losses.append(accuracy)

        dW2 = (1/m)*(dZ2 @ A1.T)
        dB2 = (1/m)*(np.sum(dZ2,axis=1,keepdims=True))
        dZ1 = ((model.W2.T)@dZ2)*relu_dash(Z1)
        dW1 = (1/m)*(dZ1 @ X_train.T)
        dB1 = (1/m)*(np.sum(dZ1,axis=1,keepdims=True))

        model.W2 -= learning_rate*dW2
        model.b2 -= learning_rate*dB2
        model.W1 -= learning_rate*dW1
        model.b1 -= learning_rate*dB1
        

        if i%10 == 0:
            print(f"Epoch {i}, accuracy is {accuracy}")

    return losses

        
        






