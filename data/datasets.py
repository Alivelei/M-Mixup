# _*_ coding: utf-8 _*_

"""
    @Time : 2022/2/23 17:06
    @Author : smile 笑
    @File : standard_dataset.py
    @desc :
"""


from torch.utils.data import Dataset
import torch
from torch import nn
import json
import pickle
import os
import torchvision.transforms as tfs
from PIL import Image
from data.word_sequence import Word2Sequence, sentence_to_word
from data.word_embedding import WordEmbedding


class QusEmbeddingMap(nn.Module):
    def __init__(self, glove_path, word_size, embedding_dim):
        super(QusEmbeddingMap, self).__init__()

        self.embedding = WordEmbedding(word_size, embedding_dim, 0.0, False)
        self.embedding.init_embedding(glove_path)

    def forward(self, qus):
        text_embedding = self.embedding(qus)

        return text_embedding


def train_aug_img(img, args, img_mean, img_std):
    aug = tfs.Compose([
        tfs.RandomResizedCrop(args.img_height, scale=(args.resized_crop_left, args.resized_crop_right)),
        tfs.RandomApply([tfs.GaussianBlur(kernel_size=args.b_size, sigma=args.blur)], p=args.blur_p),
        tfs.RandomGrayscale(p=args.grayscale),
        tfs.RandomApply([
            tfs.ColorJitter(args.brightness, args.contrast, args.saturation, args.hue)],
            p=args.apply_p
        ),
        tfs.RandomRotation(args.img_rotation),
        tfs.RandomHorizontalFlip(args.img_flip),
        tfs.ToTensor(),
        tfs.Normalize(img_mean, img_std)
    ])

    return aug(img)


def test_aug_img(img, args, img_mean, img_std):
    aug = tfs.Compose([
        tfs.Resize([args.img_height, args.img_width]),
        tfs.ToTensor(),
        tfs.Normalize(img_mean, img_std)
    ])

    return aug(img)


class SlakeDatasetModule(Dataset):
    def __init__(self, args, dataset_path, mode):
        self.args = args

        self.mode = mode
        self.xm_path = args.slake_dataset_xm_path
        self.queries = json.load(open(dataset_path, encoding="utf-8"))
        self.qus_embed_flag = args.qus_embed_flag

        # 只取英文部分
        self.queries = [query for query in self.queries if query["q_lang"] == "en"]  # 4919、1061

        self.qus_ws = pickle.load(open(args.slake_qus_ws_path, "rb"))
        self.ans_ws = pickle.load(open(args.slake_ans_ws_path, "rb"))
        self.max_seq_len = args.qus_seq_len

        qus_word_size = args.slake_qus_word_size
        glove_path = args.slake_qus_glove_path
        qus_embedding_dim = 300

        self.text_embedding_map = QusEmbeddingMap(glove_path, word_size=qus_word_size, embedding_dim=qus_embedding_dim)

        self.slake_img_mean = args.slake_img_mean
        self.slake_img_std = args.slake_img_std

    def __getitem__(self, idx):
        query = self.queries[idx]

        img_path = os.path.join(self.xm_path + str(query["img_id"]), "source.jpg")

        question = sentence_to_word(query["question"], True)
        answer = sentence_to_word(query["answer"], False)

        if self.mode == "train":
            image = train_aug_img(Image.open(img_path).convert("RGB"), self.args, self.slake_img_mean, self.slake_img_std)
        else:
            image = test_aug_img(Image.open(img_path).convert("RGB"), self.args, self.slake_img_mean, self.slake_img_std)

        answer_type = query["answer_type"]
        if answer_type == "OPEN":
            answer_type_id = self.args.answer_open
        else:
            answer_type_id = self.args.answer_close

        qus_id = self.qus_ws.transform(question, max_len=self.max_seq_len)
        ans_id = self.ans_ws.transform([answer])
        if self.qus_embed_flag:
            qus_id = torch.tensor(qus_id, dtype=torch.int64)  # 将其转换为Tensor类型
            qus_id = self.text_embedding_map(qus_id)

        return image, qus_id, ans_id, answer_type_id

    def __len__(self):
        return len(self.queries)


class RadDatasetModule(Dataset):
    def __init__(self, args, rad_dataset_path, mode):
        self.args = args

        self.mode = mode

        self.images_path = args.rad_images_path
        self.queries = json.load(open(rad_dataset_path, encoding="utf-8"))
        self.qus_ws = pickle.load(open(args.rad_qus_ws_path, "rb"))
        self.ans_ws = pickle.load(open(args.rad_ans_ws_path, "rb"))
        self.max_seq_len = args.qus_seq_len
        self.qus_embed_flag = args.qus_embed_flag

        qus_word_size = args.rad_qus_word_size
        glove_path = args.rad_qus_glove_path
        qus_embedding_dim = 300

        self.text_embedding_map = QusEmbeddingMap(glove_path, word_size=qus_word_size, embedding_dim=qus_embedding_dim)

        self.rad_img_mean = args.rad_img_mean
        self.rad_img_std = args.rad_img_std

    def __getitem__(self, idx):
        query = self.queries[idx]

        img_path = os.path.join(self.images_path, str(query["image_name"]))

        question = sentence_to_word(query["question"], True)
        answer = sentence_to_word(str(query["answer"]), False)

        if self.mode == "train":
            image = train_aug_img(Image.open(img_path).convert("RGB"), self.args, self.rad_img_mean, self.rad_img_std)
        else:
            image = test_aug_img(Image.open(img_path).convert("RGB"), self.args, self.rad_img_mean, self.rad_img_std)

        answer_type = query["answer_type"]
        if answer_type == "OPEN":
            answer_type_id = self.args.answer_open
        else:
            answer_type_id = self.args.answer_close

        qus_id = self.qus_ws.transform(question, max_len=self.max_seq_len)
        ans_id = self.ans_ws.transform([answer])

        if self.qus_embed_flag:
            qus_id = torch.tensor(qus_id, dtype=torch.int64)  # 将其转换为Tensor类型
            qus_id = self.text_embedding_map(qus_id)

        return image, qus_id, ans_id, answer_type_id

    def __len__(self):
        return len(self.queries)


