from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataloader.dataloader_bigfive import BigfiveDataset
from networks.tomsformer import VideoFeatureProjector, AudioFeatureProjector
from utils.process_data_bigfive import process_data_bigfive


def main(label_path, asr_folder, video_folder, tokenizer):
    label_df, all_asr_data, max_duration, max_txt_length = process_data_bigfive(label_path, asr_folder, tokenizer)

    # 使用示例：
    dataset = BigfiveDataset(video_folder, label_df, all_asr_data, tokenizer, max_duration, max_txt_length,
                             num_frames=10, num_mfcc=13, frames_size=256)
    dataloader = DataLoader(dataset, batch_size=2)

    for video_features, audio_features, asr_txt, txt_encoding, label in dataloader:
        # print(f'video_features.shape: ', video_features.shape)  # video_frames.shape:  torch.Size([2, 10, 1000])
        # print(f'audio_features.shape: ', audio_features.shape)  # audio_features.shape:  torch.Size([2, 26, 1121])
        # print(f'asr_txt: ', asr_txt)
        # print(f'txt_encoding: ', txt_encoding)
        # print(f'label: ', label)
        # print(f'label.shape: ', label.shape)  # label.shape:  torch.Size([2])
        # video_projector = VideoFeatureProjector(input_dim=1000, output_dim=512)
        # video_features = video_projector(video_features)
        print(f'video_features.shape: ', video_features.shape)  # video_features.shape:  torch.Size([2, 512]
        # audio_projector = AudioFeatureProjector(input_dim=1121, output_dim=512)
        # audio_features = audio_projector(audio_features)
        print(f'audio_features.shape: ', audio_features.shape)  # audio_features.shape:  torch.Size([2, 512])
        # break


if __name__ == '__main__':
    label_path = "/Users/tangbin/Documents/q_id_label20230804.xlsx"
    asr_folder = "/Users/tangbin/Documents/txts-asr-v2"
    video_folder = "/Volumes/Toms Shield/datasets/VRLab/BigFive/videos"
    # Tokenization
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    main(label_path, asr_folder, video_folder, tokenizer)
