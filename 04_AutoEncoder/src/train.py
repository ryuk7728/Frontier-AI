import torch
import torch.nn as nn
import torch.optim as optim
from src.utils import psnr


def train(model,train_loader,epochs,lr):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    optimizer = optim.AdamW(model.parameters(),lr=lr)
    criterion = nn.MSELoss()
    history = []

    for epoch in range(epochs):
        running_loss = 0
        model.train()
        for x_batch,_ in train_loader:
            x_batch = x_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred,x_batch)
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        avg_loss = running_loss/len(train_loader)
        history.append(avg_loss)
        print(f"The loss for epoch {epoch+1} is {avg_loss}, the psnr is {psnr(avg_loss):.3f}")
    
    return history 
