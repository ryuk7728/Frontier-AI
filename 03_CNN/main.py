from src.dataset import load_dataset
from src.model import build_mnist_cnn
from src.train import train,accuracy
from src.losses import categorical_cross_entropy,categorical_cross_entropy_prime
import matplotlib.pyplot as plt


def main():
    X_train,X_test,y_train,y_test = load_dataset(path="data",train_limit=None,test_limit=None)
    model = build_mnist_cnn()
    history = train(model,categorical_cross_entropy,categorical_cross_entropy_prime,X_train,y_train,epochs=1,learning_rate=0.01,verbose=True)
    acc = accuracy(model,X_test,y_test)
    print(f"The test accuracy is: {acc}")
    plt.plot(history)

if __name__ == "__main__":
    main()