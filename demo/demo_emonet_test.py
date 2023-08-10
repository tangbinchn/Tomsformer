import os
import platform

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataloader.dataloader_bigfive import BigfiveDataset
from networks.emonet import EmoNet
from pathlib import Path

from utils.process_data_bigfive import process_asr_file


def main(label_path, asr_folder, video_folder, tokenizer):
    # 加载标签文件
    label_df = pd.read_excel(label_path)
    # print(label_df.head())
    # exit()

    # 对于ASR-TXT文件夹中的每个文件，执行筛选和合并操作，并且记录单个问题的最大时长
    all_asr_data = []
    # 初始化最大时长
    max_duration = 0
    # max_txt_length = max([len(self.tokenizer.encode(text['asr_txt'])) for text in self.asr_list])
    max_txt_length = 0
    for asr_file in os.listdir(asr_folder):
        file_path = os.path.join(asr_folder, asr_file)
        processed_data, max_duration_in_file, max_txt_length_in_file = process_asr_file(file_path, tokenizer)
        all_asr_data.extend(processed_data)
        if max_duration_in_file > max_duration:
            max_duration = int(max_duration_in_file)
        if max_txt_length_in_file > max_txt_length:
            max_txt_length = int(max_txt_length_in_file)
    # print(all_asr_data[:3])
    # exit()

    # 使用示例：
    dataset = BigfiveDataset(video_folder, label_df, all_asr_data, tokenizer, max_duration, max_txt_length,
                             num_frames=10, num_mfcc=13, frames_size=256)
    dataloader = DataLoader(dataset, batch_size=2)

    for video_frames, audio_features, asr_txt, txt_encoding, label in dataloader:
        print(f'video_frames.shape: ', video_frames.shape)
        print(f'audio_features.shape: ', audio_features.shape)
        print(f'asr_txt: ', asr_txt)
        print(f'txt_encoding: ', txt_encoding)
        print(f'label: ', label)
        print(f'label.shape: ', label.shape)


if __name__ == '__main__':
    if platform.system() in ["Linux", "Windows"]:
        device = 'cuda:0'
    else:
        device = 'mps'
    print(f'Using device: {device}')

    n_expression = 5

    # Loading the model
    state_dict_path = "/Users/tangbin/Library/CloudStorage/OneDrive-shu.edu.cn/Projects/PycharmProjects/Tomsformer/models/pretrained/emonet_5.pth"
    print(f'Loading the model from {state_dict_path}.')
    # exit()
    state_dict = torch.load(str(state_dict_path), map_location='cpu')
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    net = EmoNet(n_expression=5).to(device)
    net.load_state_dict(state_dict, strict=False)
    net.eval()

    label_path = "/Users/tangbin/Documents/q_id_label20230804.xlsx"
    asr_folder = "/Users/tangbin/Documents/txts-asr-v2"
    video_folder = "/Volumes/Toms Shield/datasets/VRLab/BigFive/videos"
    # Tokenization
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    # 加载标签文件
    label_df = pd.read_excel(label_path)
    # print(label_df.head())
    # exit()

    # 对于ASR-TXT文件夹中的每个文件，执行筛选和合并操作，并且记录单个问题的最大时长
    all_asr_data = []
    # 初始化最大时长
    max_duration = 0
    # max_txt_length = max([len(self.tokenizer.encode(text['asr_txt'])) for text in self.asr_list])
    max_txt_length = 0
    for asr_file in os.listdir(asr_folder):
        file_path = os.path.join(asr_folder, asr_file)
        processed_data, max_duration_in_file, max_txt_length_in_file = process_asr_file(file_path, tokenizer)
        all_asr_data.extend(processed_data)
        if max_duration_in_file > max_duration:
            max_duration = int(max_duration_in_file)
        if max_txt_length_in_file > max_txt_length:
            max_txt_length = int(max_txt_length_in_file)
    # print(all_asr_data[:3])
    # exit()

    # 使用示例：
    dataset = BigfiveDataset(video_folder, label_df, all_asr_data, tokenizer, max_duration, max_txt_length,
                             num_frames=10, num_mfcc=13, frames_size=256)
    dataloader = DataLoader(dataset, batch_size=2)

    for video_tensor, audio_features, asr_txt, txt_encoding, label in dataloader:
        # print(f'video_frames.shape: ', video_frames.shape)
        # print(f'audio_features.shape: ', audio_features.shape)
        # print(f'asr_txt: ', asr_txt)
        # print(f'txt_encoding: ', txt_encoding)
        # print(f'label: ', label)
        # print(f'label.shape: ', label.shape)
        # Adjust the channel dimension for all frames
        video_tensor = video_tensor.permute(0, 1, 4, 2, 3)  # New shape: [N, num_frames, C, H, W]

        # Use a loop to input each frame to the model and collect predictions
        predictions = []

        for sample_tensor in video_tensor:
            sample_preds = []
            for frame in sample_tensor:
                frame = frame.unsqueeze(0)  # Add a batch dimension
                print(f'frame.shape: ', frame.shape)
                # exit()
                output = net(frame.to(device))
                valence = output['valence']
                valence = np.squeeze(valence.detach().cpu().numpy())
                # valence = np.argmax(np.squeeze(valence.detach().cpu().numpy()))
                arousal = output['arousal']
                arousal = np.squeeze(arousal.detach().cpu().numpy())
                # arousal = np.argmax(np.squeeze(arousal.detach().cpu().numpy()))
                print(f'valence: {valence}, arousal: {arousal}')
                # exit()
                sample_preds.append(output)
            # Take the average of predictions for the 10 frames of each sample
            # sample_avg_pred = torch.stack(sample_preds).mean(dim=0)
            # predictions.append(sample_avg_pred)

        # Convert the list of predictions to a tensor
        # predictions_tensor = torch.stack(predictions)
        exit()