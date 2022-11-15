# _*_ coding: utf-8 _*_

"""
    @Time : 2022/2/28 16:37
    @Author : smile 笑
    @File : standard_model.py
    @desc :
"""


import torch
from torch import nn
from framework.model.transformer.network.mlpres import ClassificationHead
from framework.model.transformer.network.word_embedding import WordEmbedding
from framework.model.transformer.network.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed
from timm.models.vision_transformer import Block
from einops.layers.torch import Rearrange


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

        return img


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


class TransformerVQASystem(nn.Module):
    def __init__(self, depth=12, emb_size=768, qus_embedding_dim=300, num_heads=12, mlp_ratio=4,
                 glove_path="../../../save/embedding/slake_qus_glove_emb_300d.npy",
                 word_size=305, ans_size=223, img_num_patches=196, qus_seq_len=20):
        super(TransformerVQASystem, self).__init__()

        self.img_num_patches = img_num_patches
        self.qus_seq_len = qus_seq_len

        self.text_embedding_linear = QusEmbeddingMap(glove_path, word_size, qus_embedding_dim, emb_size)

        self.img_mod_embed = nn.Parameter(torch.zeros(1, 1, emb_size), requires_grad=True)  # img modality cls
        self.img_pos_embed = nn.Parameter(torch.zeros(1, img_num_patches, emb_size), requires_grad=False)

        self.qus_mod_embed = nn.Parameter(torch.zeros(1, 1, emb_size), requires_grad=True)  # qus modality cls
        self.qus_pos_embed = nn.Parameter(torch.zeros(1, qus_seq_len, emb_size), requires_grad=False)

        # self.img_to_tokens = MaxLinearPooling(out_channels=emb_size)  # 自己实现的image_to_tokens
        # self.img_to_tokens = PatchEmbedding(emb_size=emb_size)  # 使用patch_embedding的方式，会比自己实现的方式模型大
        self.img_to_tokens = ImageToTokens(out_channels=emb_size)

        self.blocks = nn.ModuleList([
            Block(emb_size, num_heads, mlp_ratio, qkv_bias=True)
            for i in range(depth)])

        self.cls_forward = ClassificationHead(emb_size, ans_size)

        # initialize x and y
        self.initialize_weights_x()
        self.initialize_weights_y()

    def initialize_weights_x(self):
        img_pos_embed = get_2d_sincos_pos_embed(self.img_pos_embed.shape[-1], int(self.img_num_patches ** .5), cls_token=False)

        self.img_pos_embed.data.copy_(torch.from_numpy(img_pos_embed).float().unsqueeze(0))
        torch.nn.init.normal_(self.img_mod_embed, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def initialize_weights_y(self):
        qus_pos_embed = get_1d_sincos_pos_embed(self.qus_pos_embed.shape[-1], int(self.qus_seq_len), cls_token=False)

        self.qus_pos_embed.data.copy_(torch.from_numpy(qus_pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.qus_mod_embed, std=.02)

        # initialize nn.Linear and nn.LayerNorm
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
        x = self.img_to_tokens(img).flatten(2).transpose(1, 2).contiguous()
        y = self.text_embedding_linear(qus)

        # add pos embed w/o cls token
        x = x + self.img_pos_embed

        # add pos embed for question
        y = y + self.qus_pos_embed

        x = x + self.img_mod_embed
        y = y + self.qus_mod_embed

        # concat img embedding and qus embedding in token dim
        z = torch.cat([x, y], dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            z = blk(z)

        res = self.cls_forward(z)  # 使用单层输出

        return res


def transformer_base(**kwargs):
    model = TransformerVQASystem(
        depth=12, emb_size=768, qus_embedding_dim=300, num_heads=12, mlp_ratio=4,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
        img_num_patches=196, qus_seq_len=20
    )
    return model


def transformer_large(**kwargs):
    model = TransformerVQASystem(
        depth=24, emb_size=1024, qus_embedding_dim=300, num_heads=16, mlp_ratio=4,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
        img_num_patches=196, qus_seq_len=20
    )
    return model


def transformer_huge(**kwargs):
    model = TransformerVQASystem(
        depth=32, emb_size=1280, qus_embedding_dim=300, num_heads=16, mlp_ratio=4,
        glove_path=kwargs["glove_path"], word_size=kwargs["word_size"], ans_size=kwargs["ans_size"],
        img_num_patches=196, qus_seq_len=20
    )
    return model


if __name__ == '__main__':
    a = torch.randn([2, 3, 224, 224]).cuda()
    b = torch.randint(0, 10, [2, 20]).cuda()
    model = TransformerVQASystem().cuda()
    # torch.save(model.state_dict(), "1.pth")
    print(model(a, b).shape)
    print(sum(x.numel() for x in model.parameters()))

