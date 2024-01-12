import os
import platform
import random
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup

# 忽视警告
import warnings

from dataloader.dataloader_bigfive import BigfiveDataset
from utils.process_data_bigfive import process_data_bigfive

warnings.filterwarnings('ignore')


# 1. Data loading and processing functions
def load_data(data_folder, label_file):
    label_data = pd.read_excel(label_file)
    all_data = []
    for file in os.listdir(data_folder):
        # print(f'file: {file}')
        if file.endswith(".csv"):
            file_path = os.path.join(data_folder, file)
            data = pd.read_csv(file_path)
            filtered_data = data[data['spk_id'] == 2][['q_id', 'asr_txt']]
            # print(f'filtered_data: {filtered_data}')
            filename = os.path.splitext(file)[0].split('_')[0]
            # print(f'filename: {filename}')
            filtered_labels = label_data[label_data['filename'] == filename]
            merged_data = pd.merge(
                filtered_data, filtered_labels, on='q_id', how='inner')
            all_data.append(merged_data)
    dataset = pd.concat(all_data, axis=0, ignore_index=True)
    dataset['label'] = dataset['label'] - 1
    # # 标记含有NaN值的行
    # nan_rows = dataset.isna().any(axis=1)
    # if not nan_rows.empty:
    #     print(f"In file {file}, rows with NaN values are:")
    #     print(dataset[nan_rows])
    #     exit()
    dataset = dataset.dropna()
    print(f"dataset shape: {dataset.shape}")
    return dataset


# 3. Training and evaluation functions
def train_model(model, data_loader, optimizer, scheduler, device):
    model = model.train()
    for video_tensor, audio_tensor, asr_txt, txt_encoding, label_tensor in data_loader:
        # print(type(batch))
        # print(batch)
        # print(f'txt_encoding["input_ids"]: {txt_encoding["input_ids"]}')
        # exit()

        input_ids = txt_encoding["input_ids"].to(device)
        attention_mask = txt_encoding["attention_mask"].to(device)
        # print(f'input_ids.shape: {input_ids.shape}')
        # print(f'attention_mask.shape: {attention_mask.shape}')
        labels = label_tensor.to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        print(f'loss: {loss}')
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


def evaluate_model(model, data_loader, device):
    model = model.eval()
    total_correct = 0
    total_predictions = 0
    all_preds = []
    all_labels = []
    for video_tensor, audio_tensor, asr_txt, txt_encoding, label_tensor in data_loader:
        input_ids = txt_encoding["input_ids"].to(device)
        attention_mask = txt_encoding["attention_mask"].to(device)
        labels = label_tensor.to(device)
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        _, preds = torch.max(outputs.logits, dim=1)
        correct_predictions = preds.eq(labels).float()
        total_correct += correct_predictions.sum().item()
        total_predictions += labels.shape[0]

    # Calculate metrics
    r, _ = pearsonr(all_preds, all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    return accuracy, precision, recall, f1, r


def test_model(model, data_loader, device):
    model = model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for video_tensor, audio_tensor, asr_txt, txt_encoding, label_tensor in data_loader:
            input_ids = txt_encoding["input_ids"].to(device)
            attention_mask = txt_encoding["attention_mask"].to(device)
            labels = label_tensor.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    r, _ = pearsonr(all_preds, all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')

    # # 写入日志文件
    # with open('log/adjust_parameter_text_embedding_log.txt', 'a') as log:
    #     log.write(f"Final Test Result:\n")
    #     log.write(f"Pearson Correlation Coefficient: {r}\n")
    #     log.write(f"Accuracy: {accuracy}\n")
    #     log.write(f"Precision: {precision}\n")
    #     log.write(f"Recall: {recall}\n")
    #     log.write(f"F1 Score: {f1}\n")

    print(f"Final Test Result:")
    print(f"Pearson Correlation Coefficient: {r}")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")


def generate_random_combinations(num_iterations=1):
    learning_rates = [2e-5]
    batch_sizes = [8]
    warmup_ratios = [0]

    random_combinations = []
    for _ in range(num_iterations):
        lr = random.choice(learning_rates)
        batch_size = random.choice(batch_sizes)
        warmup_ratio = random.choice(warmup_ratios)
        random_combinations.append((lr, batch_size, warmup_ratio))

    return random_combinations


# 4. Main function
def main():
    BATCH_SIZE = 32
    EPOCHS = 10
    label_path = "/Users/tangbin/Documents/q_id_label20230804.xlsx"
    asr_folder = "/Users/tangbin/Documents/txts-asr-v2"
    video_folder = "/Volumes/Toms Shield/datasets/VRLab/BigFive/videos"
    # Tokenization
    tokenizer = BertTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    label_df, all_asr_data, max_duration, max_txt_length = process_data_bigfive(label_path, asr_folder, tokenizer)

    # 使用示例：
    dataset = BigfiveDataset(video_folder, label_df, all_asr_data, tokenizer, max_duration, max_txt_length,
                             num_frames=10, num_mfcc=13, frames_size=256)
    # dataloader = DataLoader(dataset, batch_size=5, num_workers=5)

    # Calculate the lengths of splits
    total_size = dataset.__len__()
    print(f'Total size: {total_size}')
    train_size = int(0.7 * total_size)
    valid_size = int(0.2 * total_size)
    test_size = total_size - train_size - valid_size

    # Split the dataset
    train_set, valid_set, test_set = random_split(dataset, [train_size, valid_size, test_size])
    print(f'Train size: {train_size}')
    print(f'Valid size: {valid_size}')
    print(f'Test size: {test_size}')

    # # Create DataLoaders for each subset
    # # data_loader = DataLoader(data_set, batch_size=BATCH_SIZE, shuffle=True)
    # train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    # valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=True)
    # # No need to shuffle the test set
    # test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

    # Device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        # If on Linux with multiple GPUs, use all of them
        model = torch.nn.DataParallel(
            BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", num_labels=5))
    else:
        # If on Mac, use MPS
        device = torch.device("mps" if platform.system() == "Darwin" else "cpu")
        model = BertForSequenceClassification.from_pretrained("hfl/chinese-roberta-wwm-ext", num_labels=5)
    model = model.to(device)

    # Hyperparameter tuning
    # learning_rates = [2e-5, 3e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4]
    # epochs = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 20, 30, 50, 100, 200, 300]

    best_accuracy = 0  # track the best accuracy
    best_model_path = "../models/bert_best_model.pth"

    # 获取不同超参数组合
    random_combinations = generate_random_combinations(num_iterations=1)
    # print(random_combinations)
    # exit()

    # Early stopping parameters
    no_improve_counter = 0
    patience = 10  # number of epochs without improvement before stopping, you can adjust this

    for lr, batch_size, warmup_ratio in random_combinations:
        # Adjust data loaders based on the batch size
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
        valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=8)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

        optimizer = AdamW(model.parameters(), lr=lr)
        print(f'len(train_loader): {len(train_loader)}')
        print(f'EPOCHS: {EPOCHS}')
        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_ratio, num_training_steps=total_steps)

        for ep in range(1, EPOCHS + 1):
            print(f"Training for LR={lr}, Epoch={ep}, Batch Size={batch_size}, Warmup Ratio={warmup_ratio}")
            # train_model(model, train_loader, optimizer, scheduler, device)
            model = model.train()
            for iter, (video_tensor, audio_tensor, asr_txt, txt_encoding, label_tensor) in enumerate(train_loader):
                input_ids = txt_encoding["input_ids"].to(device)
                attention_mask = txt_encoding["attention_mask"].to(device)
                # print(f'input_ids.shape: {input_ids.shape}')
                # print(f'attention_mask.shape: {attention_mask.shape}')
                labels = label_tensor.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                # print(f'loss: {loss}')
                print(f'Training iter: {iter}/{total_steps}, epoch: {ep}, loss: {loss}')
                loss.backward()
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            accuracy, precision, recall, f1, r = evaluate_model(model, valid_loader, device)

            print(
                f"Accuracy for LR={lr}, Epochs={ep}, Batch Size={batch_size}, Warmup Ratio={warmup_ratio}: {accuracy}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            print(f"F1 Score: {f1}")
            print(f"Pearson Correlation Coefficient: {r}")

            # Assuming we want to maximize accuracy for early stopping
            if best_accuracy == 0 or accuracy > best_accuracy:
                best_accuracy = accuracy
                no_improve_counter = 0
                # Save model if it performs better
                torch.save(model.state_dict(), best_model_path)
            else:
                no_improve_counter += 1

            # Stop training if counter reaches patience
            if no_improve_counter >= patience:
                print(
                    f"Early stopping triggered for LR={lr}, Epoch={ep}, Batch Size={batch_size}, Warmup Ratio={warmup_ratio}.")
                break

            # # 把上述的print改成log，然后把log写入文件，然后用grep找到最好的结果
            # # 写入日志文件
            # with open('log/adjust_parameter_text_embedding_log.txt', 'a') as log:
            #     log.write(f"Training for LR={lr} and Epoch={ep}\n")
            #     log.write(
            #         f"[EVAL] Accuracy for LR={lr}, Epochs={ep}, Batch Size={batch_size}, Warmup Ratio={warmup_ratio}: {accuracy}\n")
            #     log.write(f"[EVAL] Precision: {precision}\n")
            #     log.write(f"[EVAL] Recall: {recall}\n")
            #     log.write(f"[EVAL] F1 Score: {f1}\n")
            #     log.write(f"[EVAL] Pearson Correlation Coefficient: {r}\n")

    # Load the best model for final evaluation
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    test_model(model, test_loader, device)


if __name__ == "__main__":
    main()
