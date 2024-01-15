import time

import torch
import torch.optim as optim
import torch.nn as nn


def train_model(model, dataloader, args, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 如果 args.gpu_nums > 1，使用 DataParallel 包装模型，即可在多个 GPU 上训练
    if args.gpu_nums > 1:
        model = nn.DataParallel(model)

    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for data in dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = correct / total
        # 记录花费的时间，并格式化为时分秒
        cost_time = time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time))
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss}, Accuracy: {epoch_acc}, Cost time: {cost_time}')
    return model
