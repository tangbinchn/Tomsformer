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
from utils.process_data_bigfive import process_asr_file, process_data_bigfive
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
        self.audio_projector = AudioFeatureProjector(input_dim=26 * 1121)
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
        # print(f'video_tensor.shape: {video_tensor.shape}') # video_tensor.shape: torch.Size([10, 3, 256, 256])

        # with torch.no_grad():
        #     video_features = self.video_model(video_tensor)
        #     # print(f'video_features.shape: {video_features.shape}')  # video_features.shape: torch.Size([10, 1000])
        #     # video_features = self.video_projector(video_features)  # shape: torch.Size([512])
        #     print(f'one sample video_features.shape: {video_features.shape}')  # video_features.shape: torch.Size([512])

        # Load audio segment tensor
        audio_tensor = self._load_audio_tensor(filename, start_sec, end_sec)  # (17936000,)
        # print(f'one sample audio_tensor.shape: {audio_tensor.shape}')  # audio_tensor.shape: torch.Size([17936000])
        # exit()

        # Get label
        label = \
            self.label_df[(self.label_df['filename'] == filename) & (self.label_df['q_id'] == int(q_id))][
                'label'].values[0]
        label_tensor = torch.tensor(label, dtype=torch.long)
        # print(f'label: {label}')  # label
        # print(f'shape of label: {label.shape}')  # shape of label: torch.Size([])
        # exit()

        # 使用tokenizer处理asr_txt
        txt_encoding = self.tokenizer.encode_plus(
            asr_txt,
            add_special_tokens=True,
            # max_length=self.max_txt_length, # Bert支持最长的文本长度为512，数据集中的文本长度最大为4648
            max_length=512,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        '''
        shape of txt_encoding['input_ids']: torch.Size([1, 4648])
        shape of txt_encoding['attention_mask']: torch.Size([1, 4648])
        '''

        # print(f'one sample txt_encoding: {txt_encoding}')
        # print(f'shape of input_ids: {txt_encoding["input_ids"].shape}')
        # print(f'shape of attention_mask: {txt_encoding["attention_mask"].shape}')
        # exit()

        # print(
        #     f'filename: {filename}, q_id: {q_id}, start_sec: {start_sec}, end_sec: {end_sec}, asr_txt: {asr_txt}, label: {label}')
        # exit()

        # 下面两步也可以放在多模态模型中再提取
        txt_encoding['input_ids'] = txt_encoding['input_ids'].squeeze()  # (max_txt_length,)
        txt_encoding['attention_mask'] = txt_encoding['attention_mask'].squeeze()  # (max_txt_length,)
        # print(f'one sample txt_encoding: {txt_encoding}')
        # print(f'shape of input_ids: {txt_encoding["input_ids"].shape}')
        # print(f'shape of attention_mask: {txt_encoding["attention_mask"].shape}')
        # exit()
        '''
        shape of txt_encoding['input_ids']: torch.Size([4648])
        shape of txt_encoding['attention_mask']: torch.Size([4648])
        '''

        return video_tensor, audio_tensor, asr_txt, txt_encoding, label_tensor

    def _load_video_frames(self, filename, start_sec, end_sec, frames_size):
        video_path = os.path.join(self.video_folder, filename + '.mp4')
        cap = cv2.VideoCapture(video_path)

        # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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

    def _load_audio_tensor(self, filename, start_sec, end_sec):
        video_path = os.path.join(self.video_folder, filename + '.mp4')
        y, sr = librosa.load(video_path, sr=16000, offset=start_sec, duration=end_sec - start_sec)
        audio_tensor = torch.tensor(y, dtype=torch.float32)
        max_samples = self.max_duration * sr
        padding_length = max_samples - audio_tensor.shape[0]
        if padding_length > 0:
            # Pad audio tensor
            padding = torch.zeros(padding_length, dtype=torch.float32)
            audio_tensor = torch.cat((audio_tensor, padding))

        # # Extract MFCC features
        # mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_mfcc)
        # delta_mfccs = librosa.feature.delta(mfccs)
        # features_mfcc = np.concatenate([mfccs, delta_mfccs], axis=0)
        #
        # # Convert to torch tensor
        # features_tensor = torch.tensor(features_mfcc, dtype=torch.float32)

        # # Pad or truncate audio features to ensure consistent length
        # current_length = audio_tensor.shape[0]

        # # If current length is less than target length, pad
        # if current_length < self.max_duration:
        #     # print('current_length:', current_length)
        #     # # 查看current_length的数据类型
        #     # print('type(current_length):', type(current_length))
        #     # exit()
        #     padding_length = self.max_duration - current_length
        #     padding = torch.zeros((audio_tensor.shape[0], padding_length))
        #     audio_tensor = torch.cat([audio_tensor, padding], dim=1)
        #
        # # If current length is more than target length, truncate
        # elif current_length > self.max_duration:
        #     audio_tensor = audio_tensor[:, :self.max_duration]

        return audio_tensor


if __name__ == '__main__':
    label_path = "../datasets/bigfive/q_id_label20230804.xlsx"
    asr_folder = "../datasets/bigfive/asr_data"
    video_folder = "../datasets/bigfive/videos"
    # Tokenization
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    label_df, all_asr_data, max_duration, max_txt_length = process_data_bigfive(label_path, asr_folder, tokenizer)

    # 使用示例：
    dataset = BigfiveDataset(video_folder, label_df, all_asr_data, tokenizer, max_duration, max_txt_length,
                             num_frames=10, num_mfcc=13, frames_size=256)
    dataloader = DataLoader(dataset, batch_size=5, num_workers=5)

    for video_tensor, audio_tensor, asr_txt, txt_encoding, label_tensor in dataloader:
        print(f'video_tensor.shape: ', video_tensor.shape)  # video_tensor.shape:  torch.Size([batch_size, num_frames, 3, 256, 256])
        print(f'audio_tensor.shape: ', audio_tensor.shape)  # audio_tensor.shape:  torch.Size([batch_size, 17936000])
        print(f'input_id.shape: ', txt_encoding['input_ids'].shape)  # input_id.shape:  torch.Size([batch_size, 4648])
        print(f'attention_mask.shape: ', txt_encoding['attention_mask'].shape)  # attention_mask.shape:  torch.Size([batch_size, 4648])
        print(f'input_id: ', txt_encoding['input_ids'])  # input_id:  tensor([[ 101,  720,  720,  ...,    0,    0,    0],
        # break
