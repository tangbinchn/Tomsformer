import torch


def load_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def predict(model, dataloader, device):
    predictions = []
    with torch.no_grad():
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(device)  # 移动数据到 GPU
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().tolist())  # 将预测移回 CPU
    return predictions
