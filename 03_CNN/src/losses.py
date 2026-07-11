import numpy as np


def categorical_cross_entropy(y_true, y_pred):
    y_pred = np.clip(y_pred.copy(),1e-12,1)
    return - np.sum(y_true*np.log(y_pred))


def categorical_cross_entropy_prime(y_true, y_pred):
    return y_pred - y_true
