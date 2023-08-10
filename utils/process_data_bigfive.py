import json
import os

import numpy as np
import pandas as pd


def process_data_bigfive(label_path, asr_folder, tokenizer):
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
    return label_df, all_asr_data, max_duration, max_txt_length


# 为每个ASR文件进行筛选和合并
def process_asr_file(file_path, tokenizer):
    asr_data = []
    if file_path.endswith('.txt'):
        with open(file_path, "r", encoding="utf-8") as file:
            asr_data = [json.loads(line) for line in file.readlines()]

    # 筛选出 spk_id 为 2 的数据
    filtered_asr_data = [entry for entry in asr_data if entry['spk_id'] == '2']

    # 合并同一个 q_id 下的 asr_txt
    merged_asr_data = {}
    for entry in filtered_asr_data:
        q_id = entry['q_id']
        if q_id not in merged_asr_data:
            merged_asr_data[q_id] = entry
        else:
            merged_asr_data[q_id]['asr_txt'] += " " + entry['asr_txt']
            merged_asr_data[q_id]['end_sec'] = entry['end_sec']

    # 初始化此文件中的最大音频段长度和最大文本长度
    max_duration_in_file = 0
    max_txt_length_in_file = 0
    for key, entry in merged_asr_data.items():
        # 计算此文件中的最大音频段长度
        if entry['end_sec'] - entry['start_sec'] > max_duration_in_file:
            max_duration_in_file = entry['end_sec'] - entry['start_sec']
            # 对max_duration_in_file float类型向上取整
            max_duration_in_file = int(np.ceil(max_duration_in_file))

        # 计算此文件中的最大文本长度
        current_txt_length = len(tokenizer.encode(entry['asr_txt']))
        if current_txt_length > max_txt_length_in_file:
            max_txt_length_in_file = current_txt_length

        # 添加filename字段，并确保它在每个条目的最前面
        filename = os.path.splitext(os.path.basename(file_path))[0].split('_')[0]
        ordered_entry = {'filename': filename}
        ordered_entry.update(entry)
        merged_asr_data[key] = ordered_entry

    # # 计算此文件中的最大音频段长度
    # max_duration_in_file = max([entry['end_sec'] - entry['start_sec'] for entry in merged_asr_data.values()])

    return list(merged_asr_data.values()), max_duration_in_file, max_txt_length_in_file
