import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


# 返回字典
def read_json_file(data_path):
    return json.load(open(data_path, encoding='utf-8'))


# 统计dialogue_act种类分布
def cal_dialogue_act_distribution(data: dict):
    cnt = Counter()
    for k, v in data.items():
        for j in v:
            cnt[j["dialogue_act"]] += 1
    print(cnt)
    # 绘图
    dialogue_act_types = []
    dialogue_act_nums = []
    for k, v in cnt.items():
        dialogue_act_types.append(k)
        dialogue_act_nums.append(v)
    # 这两行代码解决 plt 中文显示的问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 给定x轴和y轴的数据
    plt.barh(dialogue_act_types, dialogue_act_nums)
    plt.title('对话意图类别分布情况')
    # 为了防止标签超出范围进行的整体缩放
    plt.tight_layout()
    plt.show()


# 统计句子的长度分布情况
def cal_text_length(data: dict):
    al_len = []
    cnt = Counter()
    for v in data.values():
        for j in v:
            al_len.append(len(j["sentence"]))
            cnt[len(j["sentence"])] += 1

    al_len = np.array(al_len)
    mode = np.argmax(np.bincount(al_len))  # 众数
    print('一共有{}条句子, 句子长度最短为{}, 最长为{}，长度的众数为{}, 一共有{}条'.format(
        len(al_len),
        min(al_len),
        max(al_len),
        mode,
        cnt[mode])
    )
    print('句子长度的平均值：{:.0f}'.format(al_len.mean()))
    proportions = ['25%', '50%', '75%', '99', '100%']
    for ind, length in enumerate(np.percentile(al_len, (25, 50, 75, 99, 100))):
        print('一共有 ' + proportions[ind] + ' 的句子长度 <= ' + str(length))

    # plt.hist(al_len, bins=400, facecolor="blue", edgecolor="black", alpha=0.7)
    # plt.show()


if __name__ == '__main__':
    json_path = r"IMCS-DAC\IMCS-DAC_train.json"
    json_data = read_json_file(json_path)
    cal_dialogue_act_distribution(json_data)
    # cal_text_length(json_data)
