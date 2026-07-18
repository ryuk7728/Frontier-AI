import matplotlib.pyplot as plt
import os
import torch


@torch.no_grad()
def save_reconstructions(model, test_loader, latent_size, device, count=8):
    model.eval()
    images = next(iter(test_loader))[0][:count].to(device)
    reconstructed = model(images)
    os.makedirs("outputs/predictions/reconstructions")

    fig, axes = plt.subplots(2, count, figsize=(2 * count, 4))
    for index in range(count):
        axes[0, index].imshow(images[index].cpu().permute(1, 2, 0))
        axes[1, index].imshow(reconstructed[index].cpu().permute(1, 2, 0))
        axes[0, index].axis("off")
        axes[1, index].axis("off")

    axes[0, 0].set_title("Originals")
    axes[1, 0].set_title("Reconstructions")
    plt.tight_layout()
    plt.savefig(f"outputs/predictions/reconstructions/latent_{latent_size}.png")
    plt.close()


def plot_results(results):
    compression = [result["compression"] for result in results]
    psnr = [result["psnr"] for result in results]

    plt.plot(compression, psnr, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Compression ratio (3072 / latent size)")
    plt.ylabel("Test PSNR (dB)")
    plt.title("Compression vs reconstruction quality")
    plt.grid()
    plt.savefig("outputs/plots/compression_vs_psnr.png")
    plt.close()
