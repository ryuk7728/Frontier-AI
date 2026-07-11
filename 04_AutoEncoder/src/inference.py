"""Plots and image artifacts for autoencoder experiments."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


@torch.inference_mode()
def save_reconstructions(model, loader, device, path, count: int = 12):
    model.eval()
    images = next(iter(loader))[0][:count].to(device)
    reconstructions = model(images)
    comparison = torch.cat([images.cpu(), reconstructions.cpu()])
    grid = make_grid(comparison, nrow=count, padding=2)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(18, 4))
    plt.imshow(grid.permute(1, 2, 0).clamp(0, 1))
    plt.axis("off")
    plt.title("Originals (top) and reconstructions (bottom)")
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def save_tradeoff_plot(results, path):
    ordered = sorted(results, key=lambda item: item["latent_size"])
    ratios = [item["compression_ratio"] for item in ordered]
    scores = [item["test_psnr"] for item in ordered]
    labels = [str(item["latent_size"]) for item in ordered]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(ratios, scores, marker="o", linewidth=2)
    for ratio, score, label in zip(ratios, scores, labels):
        plt.annotate(f"z={label}", (ratio, score), xytext=(5, 5), textcoords="offset points")
    plt.axhline(27, color="tab:red", linestyle="--", alpha=0.7, label="512 target: 27 dB")
    plt.xscale("log", base=2)
    plt.xlabel("Compression ratio (3072 / latent size; higher means more compression)")
    plt.ylabel("Test PSNR (dB; higher is better)")
    plt.title("CIFAR-10 autoencoder rate--distortion tradeoff")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
