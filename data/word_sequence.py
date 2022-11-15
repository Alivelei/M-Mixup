# _*_ coding: utf-8 _*_

"""
    @Time : 2021/9/23 9:31
    @Author : smile 笑
    @File : word_sequence.py
    @desc : 实现的是构建字典，实现方法把句子转换为数字序列并将其翻转
"""


import json
import pickle


class Sort2Id(object):
    def __init__(self, args):
        self.args = args
        self.sort_dict = {}
        self.inverse_sort_dict = dict()

    def building_classify(self):
        queries = json.load(open(self.args.train_dataset_json_path, encoding="utf-8"))
        locations = set([query["category"] for query in queries])

        for location in locations:
            self.sort_dict[location] = len(self.sort_dict)
        print(self.sort_dict)  # {'organ': 0, 'modality': 1, 'plane': 2, 'abnormality': 3}
        self.inverse_sort_dict = dict(zip(self.sort_dict.values(), self.sort_dict.keys()))

    def sort_to_id(self, location):
        return self.sort_dict.get(location)

    def id_to_sort(self, location_id):
        return self.inverse_sort_dict.get(location_id)


def sentence_to_word(sentence, qus=True):
    if qus:
        queries = str(sentence).lower().strip("?").strip(" ").split(" ")  # 将问题进行切分，且都转换为小写
    else:
        queries = str(sentence).lower().strip(" ").strip(".")  # 将答案都转换为小写
    return queries


class Word2Sequence(object):
    UNK_TAG = "UNK"
    PAD_TAG = "PAD"

    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }

        self.count = {}  # 统计词频

    def fit(self, sentence, qus=True):
        if qus:
            for word in sentence:
                self.count[word] = self.count.get(word, 0) + 1
        else:
            self.count[sentence] = self.count.get(sentence, 0) + 1

    def build_vocab(self, min_voc=None, max_voc=None, max_features=None):
        if min_voc is not None:
            self.count = {word: value for word, value in self.count.items() if value > min_voc}
        if max_voc is not None:
            self.count = {word: value for word, value in self.count.items() if value < max_voc}

        if max_features is not None:
            temp = sorted(self.count.items(), key=lambda x: x[-1], reverse=True)[:max_features]
            self.count = dict(temp)

        for word in self.count:
            self.dict[word] = len(self.dict)

        self.inverse_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len=None):
        if max_len is not None:
            if max_len > len(sentence):
                sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))
            if max_len < len(sentence):
                sentence = sentence[:max_len]

        return [self.dict.get(word, self.UNK) for word in sentence]

    def inverse_transform(self, indices):
        return [self.inverse_dict.get(idx) for idx in indices]

    def __len__(self):
        return len(self.dict)


class SaveWord2Vec(object):
    def __init__(self, lang="en"):
        self.lang = lang
        self.qus_ws = Word2Sequence()
        self.ans_ws = Word2Sequence()

    def forward(self):
        queries = json.load(open("./ref/Slake/train.json", encoding="utf-8"))

        questions = []
        answers = []

        for query in queries:
            # if query["q_lang"] == self.lang:  # 英文4919  中文4916
            questions.append(sentence_to_word(query["question"], True))
            answers.append(sentence_to_word(query["answer"], False))

        for question in questions:
            self.qus_ws.fit(question)
        for answer in answers:
            self.ans_ws.fit(answer, qus=False)

        self.qus_ws.build_vocab()
        self.ans_ws.build_vocab()  # en:305  en:223
        print(self.qus_ws.dict)  # 1371
        print(self.ans_ws.dict)
        print(len(self.qus_ws), len(self.ans_ws))
        # 保存词向量
        # pickle.dump(self.qus_ws, open("all_qus_ws.pkl", "wb"))
        # pickle.dump(self.ans_ws, open("all_ans_ws.pkl", "wb"))

        print("ws保存成功！")
        return self.qus_ws.dict, self.ans_ws.dict


if __name__ == '__main__':
    SaveWord2Vec().forward()

