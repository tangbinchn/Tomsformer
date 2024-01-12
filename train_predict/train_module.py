import torch
import torch.optim as optim
import torch.nn as nn


def train_model(model, dataloader, epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)  # 移动数据到 GPU
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model
