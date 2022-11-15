# _*_ coding: utf-8 _*_

"""
    @Time : 2022/2/25 9:26 
    @Author : smile 笑
    @File : embedding.py
    @desc :
"""


import torch
import torch.nn as nn
import numpy as np


class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """
    def __init__(self, ntoken, emb_dim, dropout, cat=True):
        super(WordEmbedding, self).__init__()
        self.cat = cat
        self.emb = nn.Embedding(ntoken, emb_dim)
        if cat:
            self.emb_ = nn.Embedding(ntoken, emb_dim)
            self.emb_.weight.requires_grad = False  # fixed
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file, tfidf=None, tfidf_weights=None):
        weight_init = torch.from_numpy(np.load(np_file))

        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init

        self.emb.weight.data[1, :] = 0  # 第一个为padding 赋值为0

        if tfidf is not None:
            if 0 < tfidf_weights.size:
                weight_init = torch.cat([weight_init, torch.from_numpy(tfidf_weights)], 0)
            weight_init = tfidf.matmul(weight_init)  # (N x N') x (N', F)
            self.emb_.weight.requires_grad = True
        if self.cat:
            self.emb_.weight.data[:self.ntoken] = weight_init.clone()

    def forward(self, x):
        emb = self.emb(x)
        if self.cat:
            emb = torch.cat((emb, self.emb_(x)), 2)
        emb = self.dropout(emb)
        return emb







