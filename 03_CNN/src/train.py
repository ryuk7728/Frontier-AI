import numpy as np


def predict(network,input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(
    network,
    loss,
    loss_prime,
    x_train,
    y_train,
    epochs=1000,
    learning_rate=0.01,
    verbose=True,
):
    history = []
    for i in range(epochs):
        error = 0
        for x,y in zip(x_train,y_train):
            y_pred = predict(network,x)
            error += loss(y,y_pred)
            gradient = loss_prime(y,y_pred)
            for layer in reversed(network):
                gradient = layer.backward(gradient,learning_rate)
        error /= len(x_train)
        history.append(error)
        if verbose:
            print(f"Epoch {i} error is {error}")
    return history


def accuracy(network, x_data, y_data):
    correct = 0
    for x,y in zip(x_data,y_data):
        correct += 1 if np.argmax(predict(network,x)) == np.argmax(y) else 0
    return correct/len(x_data)