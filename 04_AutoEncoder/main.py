import json

import torch

from src.dataset import load_data
from src.inference import plot_results
from src.model import AutoEncoder
from src.train import evaluate, train


LATENT_SIZES = [64, 128, 256, 512, 1024, 2048, 3072]
EPOCHS = 200
LEARNING_RATE = 3e-3
BATCH_SIZE = 128


def main():
    train_loader, test_loader = load_data("data", BATCH_SIZE)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []

    for latent_size in LATENT_SIZES:
        print(f"\nTraining latent size {latent_size}")
        model = AutoEncoder(latent_size).to(device)
        train(model, train_loader, EPOCHS, LEARNING_RATE)
        mse, test_psnr = evaluate(model, test_loader)

        result = {
            "latent_size": latent_size,
            "compression": 3072 / latent_size,
            "mse": mse,
            "psnr": test_psnr,
        }
        results.append(result)
        print(f"Test MSE: {mse:.6f}, Test PSNR: {test_psnr:.3f} dB")

    with open("results.json", "w") as file:
        json.dump(results, file, indent=2)

    plot_results(results)


if __name__ == "__main__":
    main()
