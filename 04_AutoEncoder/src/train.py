"""Training and evaluation for reconstruction models."""

import time
from pathlib import Path

import torch
import torch.nn as nn

from src.utils import psnr, save_json


@torch.inference_mode()
def evaluate(model, loader, device: torch.device):
    """Return true per-pixel MSE and PSNR over an entire dataset."""
    model.eval()
    squared_error = 0.0
    value_count = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        predictions = model(images)
        squared_error += nn.functional.mse_loss(
            predictions, images, reduction="sum"
        ).item()
        value_count += images.numel()
    mse = squared_error / value_count
    return {"mse": mse, "psnr": psnr(mse)}


def train(
    model,
    train_loader,
    validation_loader,
    epochs: int,
    lr: float,
    output_dir,
    weight_decay: float = 1e-5,
    device=None,
):
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.99)
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )
    criterion = nn.MSELoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    best_psnr = -float("inf")
    started = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        squared_error = 0.0
        value_count = 0
        for images, _ in train_loader:
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                predictions = model(images)
                loss = criterion(predictions, images)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            squared_error += loss.item() * images.numel()
            value_count += images.numel()

        train_mse = squared_error / value_count
        validation = evaluate(model, validation_loader, device)
        record = {
            "epoch": epoch,
            "train_mse": train_mse,
            "train_psnr": psnr(train_mse),
            "validation_mse": validation["mse"],
            "validation_psnr": validation["psnr"],
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": time.time() - started,
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d}/{epochs} | train {record['train_psnr']:.2f} dB | "
            f"val {record['validation_psnr']:.2f} dB | lr {record['lr']:.2e}"
        )
        save_json(history, output_dir / "history.json")

        if validation["psnr"] > best_psnr:
            best_psnr = validation["psnr"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "latent_size": model.latent_size,
                    "epoch": epoch,
                    "validation": validation,
                },
                output_dir / "best.pt",
            )

    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    return history
