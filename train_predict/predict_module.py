import torch
from sklearn.metrics import accuracy_score, f1_score


def load_model(model, model_path, gpu_nums):
    if gpu_nums < 2:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    else:
        # 加载模型状态字典
        state_dict = torch.load(model_path)
        # 移除 'module.' 前缀
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        # 加载更新后的状态字典
        model.load_state_dict(new_state_dict)

        return model


def predict(model, dataloader, device):
    model.eval()
    predictions = []
    labels_list = []

    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().tolist())
            labels_list.extend(labels.tolist())

    accuracy = accuracy_score(labels_list, predictions)
    f1 = f1_score(labels_list, predictions, average='macro')

    print(f'Accuracy: {accuracy}, F1: {f1}')
    return predictions
