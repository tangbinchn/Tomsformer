from torch import nn
from transformers import BertModel

class AudioFeatureProjector(nn.Module):
    def __init__(self, input_dim, output_dim=512, nhead=8, num_layers=2):
        super(AudioFeatureProjector, self).__init__()

        # Linear layer to match the transformer's expected input size
        self.embedding = nn.Linear(input_dim, output_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Linear layer to project the output to desired size (if needed)
        self.projector = nn.Linear(output_dim, 512)

    def forward(self, x):
        embedded = self.embedding(x)
        output = self.transformer_encoder(embedded)

        output = output.mean(dim=1)

        return self.projector(output)

class VideoFeatureProjector(nn.Module):
    def __init__(self, input_dim=1000, output_dim=512, nhead=8, num_layers=2):
        super(VideoFeatureProjector, self).__init__()

        # Linear layer to match the transformer's expected input size
        self.embedding = nn.Linear(input_dim, output_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Linear layer to project the output to desired size (if needed)
        self.projector = nn.Linear(output_dim, 512)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim), seq_len is 10 in your case
        embedded = self.embedding(x)
        output = self.transformer_encoder(embedded)

        # Use the last time step or average across time steps, depending on your requirement
        output = output.mean(dim=1)
        # print(f'output shape: ', output.shape)

        return self.projector(output)


class FeatureProjector(nn.Module):
    def __init__(self, input_dim, output_dim=512):
        super(FeatureProjector, self).__init__()
        self.projector = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projector(x)


class Tomsformer(nn.Module):
    def __init__(self, video_input_dim, audio_input_dim, num_labels):
        super(Tomsformer, self).__init__()
        self.video_input_dim = video_input_dim
        self.audio_input_dim = audio_input_dim
        self.num_labels = num_labels

        # 文本模块
        self.text_model = BertModel.from_pretrained("hfl/chinese-roberta-wwm-ext")

        # 视频模块
        self.video_model = nn.Sequential(
            nn.Linear(self.video_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 音频模块
        self.audio_model = nn.Sequential(
            nn.Linear(self.audio_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # 分类模块
        self.classifier = nn.Sequential(
            nn.Linear(768 + 512 + 512, 512),  # 768 from BERT, 512 from video and audio each
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, self.num_labels)
        )

    def forward(self, txt_encoding, video_data, audio_data):
        # 文本
        input_ids = txt_encoding['input_ids'].squeeze()  # (max_txt_length,)
        attention_mask = txt_encoding['attention_mask'].squeeze()  # (max_txt_length,)
        text_outputs = self.text_model(input_ids, attention_mask=attention_mask)
        text_representation = text_outputs[0][:, 0, :]  # 获取[CLS]的输出

        # 视频
        video_representation = self.video_model(video_data)

        # 音频
        audio_representation = self.audio_model(audio_data)

        # 合并所有表示
        combined_representation = torch.cat([text_representation, video_representation, audio_representation], dim=1)

        # 分类
        output = self.classifier(combined_representation)

        return output
