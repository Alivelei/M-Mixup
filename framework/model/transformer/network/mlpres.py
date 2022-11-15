# _*_ coding: utf-8 _*_

"""
    @Time : 2022/3/15 20:25
    @Author : smile 笑
    @File : mlpres.py
    @desc :
"""


import torch
from torch import nn
from einops.layers.torch import Reduce, Rearrange


class MLPFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_size1, dropout=.0):
        super(MLPFeedForward, self).__init__()

        self.mlp_layer = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_size1),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size1, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp_layer(x) + x  # 残差连接相加


class MLPStructure(nn.Sequential):
    def __init__(self, depth: int, **kwargs):
        super(MLPStructure, self).__init__(*[MLPFeedForward(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Module):
    def __init__(self, emb_size: int = 768, n_classes: int = 1000):
        super(ClassificationHead, self).__init__()
        self.cls_linear = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.BatchNorm1d(emb_size),  # 使用BN
            nn.ReLU(inplace=True),
            nn.Linear(emb_size, n_classes)
        )

    def forward(self, x):
        x = Reduce('b n e -> b e', reduction='mean')(x)
        return self.cls_linear(x)  # 平均输出


if __name__ == '__main__':
    a = torch.randn([2, 196, 768]).cuda()
    b = torch.randn([2, 20, 768]).cuda()
    model = MLPStructure(18, input_dim=768, hidden_size1=1024, dropout=.0).cuda()
    print(model(torch.cat([a, b], dim=1)))






