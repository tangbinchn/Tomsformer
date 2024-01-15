import json

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, AutoTokenizer


class CustomDataset(Dataset):
    def __init__(self, json_data, label_file, tokenizer):
        self.data = json_data
        self.label_file = label_file
        self.tokenizer = tokenizer

        # 加载标签
        with open(self.label_file, 'r') as f:
            self.labels = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = list(self.data.values())[idx]
        filename = list(self.data.keys())[idx]
        label = self.labels[filename]

        # BERT Tokenizer
        encoded_input = self.tokenizer(item['responses'],
                                       padding='max_length',
                                       truncation=True,
                                       max_length=128,
                                       return_tensors='pt')

        # 将 LIWC 特征转换为张量
        liwc_features = torch.tensor(item['feat_values'],
                                     dtype=torch.float)

        return encoded_input, liwc_features, label


def get_dataset(datasets_path, transform):
    with open(datasets_path, 'r') as f:
        json_data = json.load(f)

    dataset = CustomDataset(json_data,
                            'your_label_file.json',
                            tokenizer=BertTokenizer.from_pretrained("bert-base-chinese"))
    # 查看dataset的长度
    print(len(dataset))
    exit()
    return dataset


def get_dataloader(datasets_path, batch_size, shuffle=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = get_dataset(datasets_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def get_dataloader_tmp(datasets_path, batch_size, shuffle=True):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root=datasets_path, train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == '__main__':
    datasets_path = '../datasets/bigfive/preprocessed_data/text/liwc51*42_feats_and_responses.json'
    dataloader = get_dataloader(datasets_path, batch_size=32, shuffle=True)
    for i, (images, labels) in enumerate(dataloader):
        print(images.shape)
        print(labels.shape)
        break
