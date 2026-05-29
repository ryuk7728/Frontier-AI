from src.model import NN
from src.dataset import load_data
from src.train import train
from src.utils import get_accuracy
import numpy as np
import matplotlib.pyplot as plt

model = NN()

X_train,X_test,y_train,y_test = load_data("data",noise_samples=10000)

losses = train(model,X_train,y_train)

_,y_pred,_ = model.predict(X_test)
acc = get_accuracy(y_pred,y_test)

np.savez(r"outputs\models\mnist_weights.npz", W1 = model.W1, W2 = model.W2, b1 = model.b1, b2 = model.b2)
print("Model Weights Saved")
print(f"Test accuracy is {acc}")

plt.plot(losses)
plt.show()