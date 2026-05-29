import numpy as np

def softmax(X):
    X -= np.max(X,axis=0,keepdims=True)
    X = np.exp(X)
    X /= np.sum(X,axis=0,keepdims=True)
    return X

def relu(X):
    X[X<=0] = 0
    return X

def relu_dash(X):
    X[X<0] = 0
    X[X>0] = 1
    return X

def get_accuracy(y_pred,y_train):
    return np.sum(np.argmax(y_train,axis=0)==np.argmax(y_pred,axis=0))/y_pred.shape[1]