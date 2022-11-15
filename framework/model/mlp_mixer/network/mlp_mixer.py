# _*_ coding: utf-8 _*_

"""
    @Time : 2022/7/19 16:07 
    @Author : smile 笑
    @File : mlp_mixer.py
    @desc :
"""


import torch
from torch import nn
from einops.layers.torch import Rearrange, Reduce


class MLPFeedForward(nn.Module):
    def __init__(self, input_dim, hidden_size, dropout=.0):
        super(MLPFeedForward, self).__init__()

        self.mlp_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_size, input_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.mlp_layer(x)


class MLPMixerBlock(nn.Module):
    def __init__(self, input_dim, num_patch, token_dim, channel_dim, dropout=.0):
        super(MLPMixerBlock, self).__init__()

        self.token_mixer = nn.Sequential(
            nn.LayerNorm(input_dim),
            Rearrange("b n d -> b d n"),
            MLPFeedForward(num_patch, token_dim, dropout),
            Rearrange("b d n -> b n d")
        )

        self.channel_mixer = nn.Sequential(
            nn.LayerNorm(input_dim),
            MLPFeedForward(input_dim, channel_dim, dropout)
        )

    def forward(self, x):
        x = x + self.channel_mixer(x)  # 先channel
        x = x + self.token_mixer(x)  # 后token

        return x


class MLPMixerStructure(nn.Sequential):
    def __init__(self, depth: int, **kwargs):
        super(MLPMixerStructure, self).__init__(*[MLPMixerBlock(**kwargs) for _ in range(depth)])


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
    a = torch.randn([2, 216, 768])
    model = MLPMixerBlock(768, 216, 1024, 1024)
    print(model(a).shape)


