from data.data_module import get_dataloader
from models.model_module import Net
from train_predict.predict_module import load_model, predict
from train_predict.train_module import train_model
import torch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    exit()

    # 训练部分
    dataloader_train = get_dataloader(batch_size=32, shuffle=True)
    model = Net().to(device)
    trained_model = train_model(model, dataloader_train, epochs=10, device=device)
    torch.save(trained_model.state_dict(), 'model.pth')

    # 预测部分
    dataloader_test = get_dataloader(batch_size=32, shuffle=False)
    model = Net().to(device)
    model = load_model(model, 'model.pth')
    predictions = predict(model, dataloader_test, device=device)
    print(predictions)  # 或者对预测结果进行其他处理


if __name__ == "__main__":
    main()
