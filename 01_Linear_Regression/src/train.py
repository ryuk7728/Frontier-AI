import numpy as np
from .utils import mse_loss
from .model import LinearRegression
# import matplotlib.pyplot as plt

def train(model:LinearRegression,X_train,y_train,X_test,y_test,learning_rate=0.003,epochs=1000):
    losses = []
    n = len(X_train)

    # plt.ion()
    # _,ax = plt.subplots()
    # ax.set_ylim(14, 17)
    # ax.scatter(X_train,y_train)
    # line, = ax.plot(X_train,model.predict(X_train))


    for i in range(epochs):
        

        y_pred = model.predict(X_test)

        loss = mse_loss(y_test,y_pred)
        losses.append(loss)

        y_pred = model.predict(X_train)
        
        dW = (-2/n)*np.sum(X_train*(y_train-y_pred))
        dB = (-2/n)*np.sum(y_train-y_pred)

        model.w = model.w - learning_rate*dW
        model.b = model.b - learning_rate*dB

        # line.set_ydata(model.predict(X_train))
        # plt.pause(0.01)

        if i%10==0:
            print(f"Epoch:{i}, loss:{loss:.4f}")
    # plt.ioff()
    return losses