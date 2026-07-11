"""CLI for a reproducible CIFAR-10 compression/quality experiment."""

import argparse
import json
from pathlib import Path

import torch

from src.dataset import load_data
from src.inference import save_reconstructions, save_tradeoff_plot
from src.model import AutoEncoder
from src.train import evaluate, train
from src.utils import save_json, seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "sweep", "evaluate"), default="train")
    parser.add_argument("--latent-size", type=int, default=512)
    parser.add_argument("--latent-sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 3072])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def get_loaders(args):
    return load_data(
        args.data_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        download=args.download,
    )


def run_one(args, latent_size, loaders):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, validation_loader, test_loader = loaders
    model = AutoEncoder(latent_size)
    run_dir = Path(args.output_dir) / f"latent_{latent_size}"
    run_dir.mkdir(parents=True, exist_ok=True)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"\nLatent {latent_size} | compression {model.compression_ratio:.2f}x | "
        f"parameters {parameters:,} | device {device}"
    )
    if model.exact_identity_path:
        model.to(device)
        validation = evaluate(model, validation_loader, device)
        torch.save(
            {"model_state": model.state_dict(), "latent_size": latent_size,
             "epoch": 0, "validation": validation},
            run_dir / "best.pt",
        )
        save_json(
            [{"epoch": 0, "validation_mse": validation["mse"],
              "validation_psnr": validation["psnr"]}],
            run_dir / "history.json",
        )
        print("Exact identity initialization: no optimization required.")
    else:
        train(
            model,
            train_loader,
            validation_loader,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            output_dir=run_dir,
            device=device,
        )
    test = evaluate(model, test_loader, device)
    result = {
        "latent_size": latent_size,
        "compression_ratio": model.compression_ratio,
        "test_mse": test["mse"],
        "test_psnr": test["psnr"],
        "parameters": parameters,
        "epochs": 0 if model.exact_identity_path else args.epochs,
        "learning_rate": args.lr,
        "seed": args.seed,
    }
    save_json(result, run_dir / "result.json")
    save_reconstructions(model, test_loader, device, run_dir / "reconstructions.png")
    print(f"Test MSE {test['mse']:.6f} | Test PSNR {test['psnr']:.2f} dB")
    return result


def evaluate_checkpoint(args, loaders):
    if not args.checkpoint:
        raise ValueError("--checkpoint is required in evaluate mode")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = AutoEncoder(checkpoint["latent_size"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    test = evaluate(model, loaders[2], device)
    print(json.dumps({"latent_size": model.latent_size, **test}, indent=2))


def main():
    args = parse_args()
    loaders = get_loaders(args)
    if args.mode == "evaluate":
        evaluate_checkpoint(args, loaders)
        return
    if args.mode == "train":
        run_one(args, args.latent_size, loaders)
        return

    results = []
    for latent_size in args.latent_sizes:
        results.append(run_one(args, latent_size, loaders))
        save_json(results, Path(args.output_dir) / "sweep_results.json")
        save_tradeoff_plot(results, Path(args.output_dir) / "compression_vs_psnr.png")


if __name__ == "__main__":
    main()
