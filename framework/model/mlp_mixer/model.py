# _*_ coding: utf-8 _*_

"""
    @Time : 2022/2/28 16:37
    @Author : smile 笑
    @File : standard_model.py
    @desc :
"""


import torch
from torch import nn
from einops.layers.torch import Rearrange
from framework.model.mlp_mixer.network.mlp_mixer import ClassificationHead, MLPMixerStructure
from framework.model.mlp_mixer.network.word_embedding import WordEmbedding


class QusEmbeddingMap(nn.Module):
    def __init__(self, glove_path, word_size, embedding_dim, hidden_size):
        super(QusEmbeddingMap, self).__init__()

        self.embedding = WordEmbedding(word_size, embedding_dim, 0.0, False)
        self.embedding.init_embedding(glove_path)

        self.linear = nn.Linear(embedding_dim, hidden_size)

    def forward(self, qus):
        text_embedding = self.embedding(qus)

        text_x = self.linear(text_embedding)

        return text_x


class MaxLinearPooling(nn.Module):
    def __init__(self, in_chans=3, out_channels=768, conv_f=32):
        super(MaxLinearPooling, self).__init__()

        self.max_pooling = nn.Sequential(
            nn.Conv2d(in_chans, conv_f, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(conv_f, conv_f, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(conv_f, conv_f, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(conv_f, out_channels, 3, 1, 1),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.max_pooling(x)


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int = 3, patch_size: int = 16, emb_size: int = 768):
        self.patch_size = patch_size
        super().__init__()
        self.projection = nn.Sequential(
            # using a conv layer instead of a linear one -> performance gains
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            # Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x):
        x = self.projection(x)

        return x


class ImageToTokens(nn.Module):
    def __init__(self, in_channels=3, out_channels=768, conv_kernel=7, conv_stride=4, conv_padding=4, pool_kernel=4, pool_stride=4):
        super(ImageToTokens, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, conv_kernel, conv_stride, conv_padding),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(pool_kernel, pool_stride)
        )

    def forward(self, x):
        img = self.conv(x)
        # b, c, _, _ = img.shape
        # img_to_token = img.view(b, c, -1).transpose(1, 2)

        return img


class MLPMixerVQASystem(nn.Module):
    def __init__(self, depth=12, emb_size=768, drop_out=.2, qus_embedding_dim=300, num_patch=216,
                 token_dim=1024, channel_dim=1024, glove_path="../../../save/embedding/slake_qus_glove_emb_300d.npy",
                 word_size=305, ans_size=223):
        super(MLPMixerVQASystem, self).__init__()

        self.text_embedding_linear = QusEmbeddingMap(glove_path, word_size, qus_embedding_dim, emb_size)

        # self.img_to_tokens = MaxLinearPooling(out_channels=emb_size)  # 自己实现的image_to_tokens
        # self.img_to_tokens = PatchEmbedding(emb_size=emb_size)  # 使用patch_embedding的方式，会比自己实现的方式模型大
        self.img_to_tokens = ImageToTokens(out_channels=emb_size)

        self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, emb_size), requires_grad=True)  # img modality cls
        self.qus_mod_embed = nn.Parameter(torch.zeros(1, 1, emb_size), requires_grad=True)  # qus modality cls

        self.mlp_mixer = MLPMixerStructure(depth, input_dim=emb_size, num_patch=num_patch, token_dim=token_dim,
                                           channel_dim=channel_dim, dropout=drop_out)

        self.cls_forward = ClassificationHead(emb_size, ans_size)

        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.img_mod_embed, std=.02)
        torch.nn.init.normal_(self.qus_mod_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, img, qus):
        img_tokens = self.img_to_tokens(img).flatten(2).transpose(1, 2).contiguous()

        qus_embedding = self.text_embedding_linear(qus)

        img_tokens = img_tokens + self.img_mod_embed
        qus_embedding = qus_embedding + self.qus_mod_embed

        all_features = self.mlp_mixer(torch.cat([img_tokens, qus_embedding], dim=1))

        res = self.cls_forward(all_features)  # 使用单层输出

        return res


def mlp_mixer_base(**kwargs):
    model = MLPMixerVQASystem(
        depth=12, emb_size=768, drop_out=.0, qus_embedding_dim=300, num_patch=216, token_dim=1024, channel_dim=1024,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
    )
    return model


def mlp_mixer_large(**kwargs):
    model = MLPMixerVQASystem(
        depth=24, emb_size=1024, drop_out=.0, qus_embedding_dim=300, num_patch=216, token_dim=1536, channel_dim=1536,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
    )
    return model


def mlp_mixer_huge(**kwargs):
    model = MLPMixerVQASystem(
        depth=32, emb_size=1280, drop_out=.0, qus_embedding_dim=300, num_patch=216, token_dim=2048, channel_dim=2048,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
    )
    return model


if __name__ == '__main__':
    a = torch.randn([2, 3, 224, 224]).cuda()
    b = torch.randint(0, 20, [2, 20]).cuda()
    model = MLPMixerVQASystem().cuda()
    # torch.save(model.state_dict(), "1.pth")
    print(model(a, b).shape)
    print(sum(x.numel() for x in model.parameters()))  # 25913835

