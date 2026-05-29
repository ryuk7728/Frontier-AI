from src.train import train
from src.dataset import load_data
from src.model import LinearRegression
from src.utils import mse_loss
import matplotlib.pyplot as plt
import numpy as np

path = r"data\Housing.csv"
X_train,X_test,y_train,y_test = load_data(path)

model = LinearRegression()
losses = train(model,X_train,y_train,X_test,y_test,epochs=1000)

y_pred = model.predict(X_test)
test_loss = mse_loss(y_test,y_pred)
print(f"The final test loss is: {test_loss}")

plt.figure()
plt.plot(losses)
plt.savefig(r"outputs\plots\loss_curve.png")
plt.show()

print(f"Your final answer is {model.w}x+{model.b}")

plt.figure()
plt.scatter(X_train,y_train)
plt.plot(X_train,model.predict(X_train),)
plt.savefig(r"outputs\plots\regression_line.png")
plt.show()

