from torch.utils.data import DataLoader
import torch
from data_analysis import data_analysis
from transformers import AutoTokenizer
import random
import os
import requests

# 检查pytorch_model.bin在不在src/model/bert_base_Chinese/ 中，如果不在则下载下来
if not os.path.exists('src/model/bert_base_Chinese/pytorch_model.bin'):
    print('pytorch_model 文件不存在，正在下载')
    url = 'https://huggingface.co/bert-base-chinese/resolve/main/pytorch_model.bin'
    r = requests.get(url)
    with open('src/model/bert_base_Chinese/pytorch_model.bin', 'rb') as f:
        f.write(r.content)
        f.close()
tokenizer = AutoTokenizer.from_pretrained("src/model/bert_base_Chinese")


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, json_path, is_train=False):
        self.is_train = is_train
        self.json_data = data_analysis.read_json_file(json_path)
        self.sentences = []
        self.dialogue_act = []
        self.type2id = {"Other": 0, "Request-Precautions": 1, "Request-Symptom": 2, "Inform-Symptom": 3,
                        "Request-Basic_Information": 4, "Inform-Drug_Recommendation": 5,
                        "Inform-Existing_Examination_and_Treatment": 6, "Inform-Precautions": 7,
                        "Inform-Basic_Information": 8,
                        "Request-Existing_Examination_and_Treatment": 9,
                        "Request-Drug_Recommendation": 10, "Inform-Medical_Advice": 11,
                        "Diagnose": 12, "Inform-Etiology": 13, "Request-Medical_Advice": 14,
                        "Request-Etiology": 15
                        }
        self.all_type_datas = {"Other": [], "Request-Precautions": [], "Request-Symptom": [], "Inform-Symptom": [],
                               "Request-Basic_Information": [], "Inform-Drug_Recommendation": [],
                               "Inform-Existing_Examination_and_Treatment": [], "Inform-Precautions": [],
                               "Inform-Basic_Information": [],
                               "Request-Existing_Examination_and_Treatment": [],
                               "Request-Drug_Recommendation": [], "Inform-Medical_Advice": [],
                               "Diagnose": [], "Inform-Etiology": [], "Request-Medical_Advice": [],
                               "Request-Etiology": []
                               }
        self.loadDatas()
        self.choose_data()
        self.shuffle_data()

    def choose_data(self):
        for k, v in self.all_type_datas.items():
            if len(v) > 2000:
                al_nums = [j for j in range(len(v) // 2000)] if self.is_train else [0]
                for s in v:
                    if random.randint(al_nums[0], al_nums[-1]) == al_nums[0]:
                        self.sentences.append(s)
                        self.dialogue_act.append(self.type2id[k])
            else:
                for s in v:
                    self.sentences.append(s)
                    self.dialogue_act.append(self.type2id[k])

    # 打乱所有的数据
    def shuffle_data(self):
        t = list(zip(self.sentences, self.dialogue_act))
        random.shuffle(t)
        self.sentences, self.dialogue_act = zip(*t)

    def loadDatas(self):
        for v in self.json_data.values():
            for j in v:
                self.all_type_datas[j["dialogue_act"]].append(j["sentence"])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        sentence = self.sentences[item]
        dialogue_act = self.dialogue_act[item]
        return sentence, dialogue_act


# 批处理函数
def collate_fn(data):
    sentences = [i[0] for i in data]
    labels = [i[1] for i in data]
    data = tokenizer.batch_encode_plus(batch_text_or_text_pairs=sentences,
                                       truncation=True,
                                       padding='max_length',
                                       max_length=200,
                                       return_tensors='pt',
                                       return_length=True)
    input_ids = data['input_ids']
    attention_mask = data['attention_mask']
    token_type_ids = data['token_type_ids']
    labels = torch.LongTensor(labels)
    return input_ids, attention_mask, token_type_ids, labels


def my_loader(json_path, batch_size=64, is_train=False):
    my_dataset = TrainDataset(json_path, is_train=is_train)
    my_data_loader = DataLoader(dataset=my_dataset,
                                batch_size=batch_size,
                                collate_fn=collate_fn,
                                shuffle=True,
                                drop_last=False)
    return my_data_loader


if __name__ == '__main__':
    m = my_loader(r"IMCS-DAC/IMCS-DAC_train.json")
    print(len(m))
