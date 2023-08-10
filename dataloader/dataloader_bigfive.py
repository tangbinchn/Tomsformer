import json
import os
import cv2
import librosa
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from torchvision.transforms import Compose, Normalize
from transformers import BertTokenizer

from networks.tomsformer import VideoFeatureProjector, AudioFeatureProjector
from utils.process_data_bigfive import process_asr_file
# 忽视警告
import warnings

warnings.filterwarnings('ignore')


class BigfiveDataset(Dataset):
    def __init__(self, video_folder, label_df, asr_list, tokenizer, max_duration, max_txt_length, num_frames=10,
                 num_mfcc=13, frames_size=256, transform_video=None, transform_text=None):
        self.video_folder = video_folder
        self.label_df = label_df
        self.transform_video = transform_video
        self.transform_text = transform_text
        self.frames_size = frames_size
        # Video feature extraction
        self.video_transform = Compose([
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.video_projector = VideoFeatureProjector(input_dim=1000)
        self.audio_projector = AudioFeatureProjector(input_dim=26*1121)
        self.video_model = resnet50(pretrained=True)
        self.video_model.eval()
        # 过滤 asr_list，确保只选择有标签的数据
        self.asr_list = [entry for entry in asr_list
                         if not label_df[(label_df['filename'] == entry['filename']) &
                                         (label_df['q_id'] == int(entry['q_id']))].empty]
        # print(f'self.asr_list: {self.asr_list}')

        self.tokenizer = tokenizer
        self.num_frames = num_frames
        self.num_mfcc = num_mfcc
        self.max_duration = max_duration
        self.max_txt_length = max_txt_length
        # print(f'self.max_duration: {self.max_duration}')

        # # 获取数据集中文本的最大长度
        # self.max_txt_length = max([len(self.tokenizer.encode(text['asr_txt'])) for text in self.asr_list])
        # print('self.max_txt_length:', self.max_txt_length)
        # # 查看max_txt_length的数据类型
        # print('type(self.max_txt_length):', type(self.max_txt_length))
        # exit()

    def __len__(self):
        # print('len(self.asr_list):', len(self.asr_list))
        return len(self.asr_list)

    def __getitem__(self, idx):
        asr_entry = self.asr_list[idx]
        filename = asr_entry['filename']
        q_id = asr_entry['q_id']
        start_sec = asr_entry['start_sec']
        end_sec = asr_entry['end_sec']
        asr_txt = asr_entry['asr_txt']

        # Load video framframeses
        video_frames = self._load_video_frames(filename, start_sec, end_sec, self.frames_size)

        # Convert frames to tensor
        video_tensor = torch.stack([torch.tensor(frame, dtype=torch.float32) for frame in video_frames])
        # print(f'video_tensor.shape: {video_tensor.shape}') # video_tensor.shape: torch.Size([10, 256, 256, 3])

        video_tensor = video_tensor.permute(0, 3, 1, 2)  # (10, 3, 256, 256)

        with torch.no_grad():
            video_features = self.video_model(video_tensor)
            # print(f'video_features.shape: {video_features.shape}')  # video_features.shape: torch.Size([10, 1000])
            # video_features = self.video_projector(video_features)  # shape: torch.Size([512])
            print(f'one sample video_features.shape: {video_features.shape}')  # video_features.shape: torch.Size([512])

        # Load audio segment and extract features
        audio_features = self._load_audio_features(filename, start_sec, end_sec, self.num_mfcc)
        # print(f'audio_features.shape: {audio_features.shape}')  # audio_features.shape: torch.Size([26, 1121])
        # audio_features = self.audio_projector(audio_features)  # shape: torch.Size([512])
        print(f'one sample audio_features.shape: {audio_features.shape}')  # audio_features.shape: torch.Size([512])
        # exit()

        # Get label
        label = \
            self.label_df[(self.label_df['filename'] == filename) & (self.label_df['q_id'] == int(q_id))][
                'label'].values[0]

        # 使用tokenizer处理asr_txt
        txt_encoding = self.tokenizer.encode_plus(
            asr_txt,
            add_special_tokens=True,
            max_length=self.max_txt_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        # print(
        #     f'filename: {filename}, q_id: {q_id}, start_sec: {start_sec}, end_sec: {end_sec}, asr_txt: {asr_txt}, label: {label}')
        # exit()

        # # 下面两步放在多模态模型中再提取
        # input_ids = txt_encoding['input_ids'].squeeze()  # (max_txt_length,)
        # attention_mask = txt_encoding['attention_mask'].squeeze()  # (max_txt_length,)

        return video_features, audio_features, asr_txt, txt_encoding, label

    def _load_video_frames(self, filename, start_sec, end_sec, frames_size):
        video_path = os.path.join(self.video_folder, filename + '.mp4')
        cap = cv2.VideoCapture(video_path)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frames = []

        # Calculate the indices of frames to capture
        start_idx = int(start_sec * fps)
        end_idx = int(end_sec * fps)
        step = max((end_idx - start_idx) // self.num_frames, 1)
        # print(f'start_idx: {start_idx}, end_idx: {end_idx}, step: {step}')
        # exit()

        for idx in range(start_idx, end_idx, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert the frame from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Crop the frame to 1080x1080 from the center
                center_x = frame.shape[1] // 2
                center_y = frame.shape[0] // 2
                half_width = 540  # Half of 1080
                cropped_frame = frame[center_y - half_width:center_y + half_width,
                                center_x - half_width:center_x + half_width]

                # Resize the frame to image_size (224x224)
                resized_frame = cv2.resize(cropped_frame, (frames_size, frames_size))

                frames.append(resized_frame)

                if len(frames) >= self.num_frames:
                    break

        cap.release()

        # # Convert frames to tensor
        # frames_tensor = torch.stack([torch.tensor(frame, dtype=torch.float32) for frame in frames])
        return frames

    def _load_audio_features(self, filename, start_sec, end_sec, num_mfcc):
        video_path = os.path.join(self.video_folder, filename + '.mp4')
        y, sr = librosa.load(video_path, sr=16000, offset=start_sec, duration=end_sec - start_sec)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
        delta_mfccs = librosa.feature.delta(mfccs)
        features_mfcc = np.concatenate([mfccs, delta_mfccs], axis=0)

        # Convert to torch tensor
        features_tensor = torch.tensor(features_mfcc, dtype=torch.float32)

        # Pad or truncate audio features to ensure consistent length
        current_length = features_tensor.shape[1]

        # If current length is less than target length, pad
        if current_length < self.max_duration:
            # print('current_length:', current_length)
            # # 查看current_length的数据类型
            # print('type(current_length):', type(current_length))
            # exit()
            padding_length = self.max_duration - current_length
            padding = torch.zeros((features_tensor.shape[0], padding_length))
            features_tensor = torch.cat([features_tensor, padding], dim=1)

        # If current length is more than target length, truncate
        elif current_length > self.max_duration:
            features_tensor = features_tensor[:, :self.max_duration]

        return features_tensor
