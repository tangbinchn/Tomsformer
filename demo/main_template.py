import warnings, sys, os, yaml, argparse

warnings.filterwarnings("ignore", category=Warning)
# 将本路径加入到系统路径中
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# 将上级路径加入到系统路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_module import get_dataloader
from models.model_module import Net
from train_predict.predict_module import load_model, predict
from train_predict.train_module import train_model
import torch
from utils.check_data import check_model_device, check_data_device


def main(args, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')
    # exit()

    # 训练部分
    dataloader_train = get_dataloader(config['datasets_path'], args.batch_size, shuffle=True)
    data, label = next(iter(dataloader_train))
    data, label = data.to(device), label.to(device)
    check_data_device(data, label, device)

    model = Net().to(device)
    check_model_device(model, device)

    trained_model = train_model(model, dataloader_train, args, device=device)
    torch.save(trained_model.state_dict(), 'model.pth')

    # 预测部分
    dataloader_test = get_dataloader(config['datasets_path'], args.batch_size, shuffle=False)
    model = Net().to(device)
    model = load_model(model, 'model.pth', args.gpu_nums)
    predictions = predict(model, dataloader_test, device=device)
    print(predictions)  # 或者对预测结果进行其他处理


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='args')
    parser.add_argument('--batch_size', type=int, default=512, help='batch_size')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--gpu_nums', type=int, default=2, help='gpu_nums')
    args = parser.parse_args()
    # 读取配置文件
    with open('config/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    main(args, config)
