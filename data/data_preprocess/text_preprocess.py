import re
from glob import glob
import os
import json
import warnings
import numpy as np
import pandas as pd
import jieba

warnings.filterwarnings("ignore")


def parse_asr_version(filename):
    room_number, sequence_number = map(int, re.findall(r'\d+', filename)[:2])

    if 'reality' in filename:
        if room_number in {2022, 2023} and 1 <= sequence_number <= 4:
            return 41
        elif room_number == 2022 and 5 <= sequence_number <= 25:
            return 40
        elif room_number == 2023 and 7 <= sequence_number <= 28:
            return 40
        elif room_number == 2025 and 9 <= sequence_number <= 29:
            return 40
    elif 'virtual' in filename:
        if room_number == 2022 and 1 <= sequence_number <= 46:
            return 40
        elif room_number == 2025 and 1 <= sequence_number <= 48:
            return 40
        elif room_number == 2022 and 47 <= sequence_number <= 79:
            return 42
        elif room_number == 2025 and 49 <= sequence_number <= 78:
            return 42
    else:
        return 42


def parse_asr_txt(txt_path, mapping_path, count_questions=42):
    df = pd.read_excel(mapping_path)
    version1Tnow = {}
    version2Tnow = {}
    for idx, row in df.iterrows():
        version1 = row[41]
        version2 = row[40]
        now = row[42]
        version1Tnow[int(version1)] = int(now)
        version2Tnow[int(version2)] = int(now)
    version = parse_asr_version(os.path.basename(txt_path))
    cur_robot_spk_id = 1
    lines1 = open(txt_path, 'r', encoding='gbk').readlines()
    responses = ['' for _ in range(count_questions)]
    questions = ['' for _ in range(count_questions)]
    statics_feat_names = ["speech_loudness", "speech_speed", "speech_pitch", "speech_length", "asr_length", "word_num",
                          "sentence_num", "pos_word_num", "neg_word_num", "n", "v", "a", "r", "d", "e", "t", "w",
                          "good_word", "happy_word", "sad_word", "angry_word", "fear_word", "disgust_word",
                          "surprise_word"]
    statics_feats = [np.zeros(len(statics_feat_names)) for _ in range(count_questions)]
    for line1 in lines1:
        info = json.loads(line1)
        statics_feat = np.asarray([float(info.get(feat_name, 0.)) for feat_name in statics_feat_names])
        spk_id = int(info.get('spk_id', -1))
        qid = int(info.get('q_id', -1))
        # 问题映射
        if version == 40:
            try:
                qid = version2Tnow[qid]
            except KeyError:
                # 如果KeyError显示42，说明没问题，因为已经把asr文件按照42题版本的格式进行了保存
                if qid != 42:
                    print("!!!Error txt_path!!!:", txt_path)
                    exit()
                continue
        elif version == 41:
            try:
                qid = version1Tnow[qid]
            except KeyError:
                # 如果KeyError显示42，说明没问题，因为已经把asr文件按照42题版本的格式进行了保存
                if qid != 42:
                    print("!!!Error txt_path!!!:", txt_path)
                    exit()
                continue
        if spk_id != cur_robot_spk_id:
            # RESPONSE
            if qid >= 1 and qid <= count_questions:
                # print(responses[qid-1], type(responses[qid-1]), line2[1], type(line2[1]))
                responses[qid - 1] = responses[qid - 1] + str(info.get('asr_txt', -1))
                statics_feats[qid - 1] += statics_feat
            else:
                print(f'skip response qid: {qid}')
        elif spk_id == cur_robot_spk_id:
            # RESPONSE
            if qid >= 1 and qid <= count_questions:
                # print(responses[qid-1], type(responses[qid-1]), line2[1], type(line2[1]))
                questions[qid - 1] = questions[qid - 1] + str(info.get('asr_txt', -1))
            else:
                print('skip question')
        else:
            print(f'skip {spk_id}')

    return questions, responses, np.reshape(np.asarray(statics_feats), [-1]).tolist(), np.reshape(
        np.asarray([['{}_{}'.format(ele, idx) for ele in statics_feat_names] for idx in range(1, count_questions + 1)]),
        [-1]).tolist()


def get_liwc_feat(responses, liwc_dict, stopwords, count_questions=42):
    questions = ['' for _ in range(count_questions)]

    statics_feat_names = [
        "funct", "negate", "quant", "number", "swear",
        "TenseM", "PastM", "PresentM", "FutureM", "ProgM",
        "social", "family", "friend", "humans", "affect",
        "posemo", "negemo", "anx", "anger", "sad",
        "cogmech", "insight", "cause", "discrep", "tentat",
        "certain", "inhib", "incl", "excl", "percept",
        "see", "hear", "feel", "bio", "body",
        "health", "sexual", "ingest", "relativ", "motion",
        "space", "time", "Personal_Concerns", "work", "achievef",
        "leisure", "home", "money", "relig", "death",
        "assent"
    ]

    # 添加LIWC特征值
    LIWC_values_list = []
    for _ in responses:
        feat_values = analyze_text(_, liwc_dict, stopwords)
        LIWC_values_list.append(feat_values)

    return questions, responses, np.reshape(np.asarray(LIWC_values_list), [-1]).tolist(), np.reshape(
        np.asarray([['{}_{}'.format(ele, idx) for ele in statics_feat_names] for idx in range(1, count_questions + 1)]),
        [-1]).tolist()


def analyze_text(text, liwc_dict, stopwords):
    words = jieba.cut(text)
    filtered_words = [word for word in words if word not in stopwords]
    # 初始化51维向量
    category_counts = [0] * 51

    for word in filtered_words:
        if word in liwc_dict:
            for category in liwc_dict[word]:
                category_index = int(category)  # 假设类别编号从0开始
                if 0 <= category_index < 51:
                    category_counts[category_index] += 1

    return category_counts


def load_stopwords(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        stopwords = set([line.strip() for line in file])
    return stopwords


def load_liwc_dictionary(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        liwc_dict = json.load(file)
    return liwc_dict


def get_liwc_text_feat(text_save_path, asr_dir, mapping_path, liwc_dict, stopwords):
    question_feat_data = {}
    for sample_path in glob(os.path.join(asr_dir, '*_asr.txt')):
        # print('sample_path: ', sample_path)
        # filename为文件名，不包含路径和后缀
        filename = sample_path.rsplit(".", 1)[0].rstrip("_asr").split('/')[-1]

        questions, responses, statics_feat_values, statics_feat_names = parse_asr_txt(sample_path,
                                                                                      mapping_path)
        # print('responses: ', responses)
        # exit()
        # 添加LIWC特征
        _, _, liwc_feat_values, liwc_feat_names = get_liwc_feat(responses, liwc_dict, stopwords)

        question_feat_data[filename] = {
            'feat_names': liwc_feat_names,
            'feat_values': liwc_feat_values,
            'responses': responses
        }

    with open(text_save_path, 'w') as file:
        print(f'Saving question level features to {text_save_path}')
        # ensure_ascii=False，输出中文
        json.dump(question_feat_data, file, ensure_ascii=False, indent=2)

    print('Done!')


if __name__ == '__main__':
    # time = get_current_time(level='day')
    # question_level_feat_path = f'../features/original/question_level/text/liwc_text_question_level_features_{time}.json'
    text_save_path = f'../../datasets/bigfive/preprocessed_data/text/liwc51*42_feats_and_responses.json'
    asr_dir = f'../../datasets/bigfive/asr_data/txts_q_id'
    mapping_path = f'./question/question_mapping_20231219.xlsx'

    stopwords = load_stopwords(f'./LIWC/cn_stopwords.txt')
    liwc_dict = load_liwc_dictionary(f'./LIWC/reencoded_liwc_dict.json')
    get_liwc_text_feat(text_save_path, asr_dir, mapping_path, liwc_dict, stopwords)
