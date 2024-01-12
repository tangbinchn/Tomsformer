from torch.utils.data import DataLoader
from transformers import BertTokenizer

from dataloader.dataloader_bigfive import BigfiveDataset
from utils.process_data_bigfive import process_data_bigfive

if __name__ == '__main__':
    label_path = "/Users/tangbin/Documents/q_id_label20230804.xlsx"
    asr_folder = "/Users/tangbin/Documents/txts-asr-v2"
    video_folder = "/Volumes/Toms-Shield/datasets/VRLab/BigFive/videos"
    # Tokenization
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    label_df, all_asr_data, max_duration, max_txt_length = process_data_bigfive(label_path, asr_folder, tokenizer)

    # 使用示例：
    dataset = BigfiveDataset(video_folder, label_df, all_asr_data, tokenizer, max_duration, max_txt_length,
                             num_frames=10, num_mfcc=13, frames_size=256)
    dataloader = DataLoader(dataset, batch_size=5, num_workers=5)

    for video_tensor, audio_tensor, asr_txt, txt_encoding, label_tensor in dataloader:
        print(f'video_tensor.shape: {video_tensor.shape}')  # video_tensor.shape:  torch.Size([batch_size, num_frames, 3, 256, 256])
        print(f'audio_tensor.shape: ', audio_tensor.shape)  # audio_tensor.shape:  torch.Size([batch_size, 17936000])
        print(f'input_id.shape: ', txt_encoding['input_ids'].shape)  # input_id.shape:  torch.Size([batch_size, 4648])
        print(f'attention_mask.shape: ',
              txt_encoding['attention_mask'].shape)  # attention_mask.shape:  torch.Size([batch_size, 4648])
        # break
