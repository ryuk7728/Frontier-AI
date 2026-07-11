import torch
import torch.nn as nn
from src.model import AutoEncoder
from src.dataset import load_data
from src.train import train


train_loader,test_loader = load_data("data")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoEncoder().to(device)
epochs = 100
lr = 3e-3
history = train(model,train_loader,epochs,lr)
criterion = nn.MSELoss()

with torch.no_grad():
    model.eval()
    avg_loss = 0
    for x_batch,_ in test_loader:
        x_batch = x_batch.to(device)
        y_pred = model(x_batch)
        loss = criterion(y_pred,x_batch)
        avg_loss += loss.item()/len(test_loader)
    print(f"Test MSE: {avg_loss:.6f}")

