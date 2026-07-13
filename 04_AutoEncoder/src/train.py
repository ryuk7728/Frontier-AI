import torch
import torch.nn as nn
import torch.optim as optim

from src.utils import psnr


def train(model, train_loader, epochs, lr):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for x_batch, _ in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, x_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)
        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"MSE: {average_loss:.6f}, PSNR: {psnr(average_loss):.3f} dB"
        )


@torch.no_grad()
def evaluate(model, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0
    total_images = 0

    for x_batch, _ in test_loader:
        x_batch = x_batch.to(device)
        batch_loss = criterion(model(x_batch), x_batch)
        total_loss += batch_loss.item() * len(x_batch)
        total_images += len(x_batch)

    mse = total_loss / total_images
    return mse, psnr(mse)
