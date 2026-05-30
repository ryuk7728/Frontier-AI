from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.dataset import load_data
from src.deep_dream import dream_until
from src.model import NN
from src.train import train
from src.utils import get_accuracy


MODEL_PATH = Path("outputs/models/mnist_weights.npz")
PLOTS_DIR = Path("outputs/plots")


def save_training_accuracy_plot(accuracies):
    plt.figure(figsize=(7,4))
    plt.plot(accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Training Accuracy")
    plt.title("MNIST Training Accuracy")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "training_accuracy.png", dpi=160)
    plt.close()


def save_deep_dream_grid(model):
    fig, axes = plt.subplots(1,10,figsize=(12,2))
    final_probs = []

    for digit, ax in enumerate(axes):
        actives, X = dream_until(model,digit,target_prob=0.99,max_steps=5000,learning_rate=0.02)
        final_probs.append(actives[-1])

        ax.imshow(X.reshape(28,28),cmap = "grey")
        ax.set_title(f"{digit}\n{actives[-1]:.3f}",fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "deep_dream_digits.png", dpi=180)
    plt.close()
    return final_probs


def main():

    model = NN()
    X_train,X_test,y_train,y_test = load_data("data",noise_samples=10000)

    accuracies = train(model,X_train,y_train)

    _,y_pred,_ = model.predict(X_test)
    test_acc = get_accuracy(y_pred,y_test)

    np.savez(MODEL_PATH, W1=model.W1, W2=model.W2, b1=model.b1, b2=model.b2)
    save_training_accuracy_plot(accuracies)
    final_probs = save_deep_dream_grid(model)

    print(f"Model weights saved to {MODEL_PATH}")
    print(f"Training accuracy plot saved to {PLOTS_DIR / 'training_accuracy.png'}")
    print(f"Deep dream digit grid saved to {PLOTS_DIR / 'deep_dream_digits.png'}")
    print(f"Test accuracy is {test_acc}")
    print(f"Dream probabilities: {[round(p,4) for p in final_probs]}")


if __name__ == "__main__":
    main()
