# _*_ coding: utf-8 _*_

"""
    @Time : 2022/2/28 16:37
    @Author : smile 笑
    @File : standard_model.py
    @desc :
"""


import torch
from torch import nn
from torchvision.models import resnet50


class TextModel(nn.Module):
    def __init__(self, qus_embedding_dim=300, lstm_n_hidden=1024, lstm_num_layers=1, dropout=.0):
        super(TextModel, self).__init__()

        self.qus_model = nn.LSTM(input_size=qus_embedding_dim, hidden_size=lstm_n_hidden, num_layers=lstm_num_layers, batch_first=True, dropout=dropout)

    def forward(self, qus_embedding):
        text_embedding_res, _ = self.qus_model(qus_embedding)  # 得到序列的输出 [b, 12, 1024]

        return text_embedding_res


class SanAttention(nn.Module):
    def __init__(self, d=1024, k=512, dropout=True):
        super(SanAttention, self).__init__()
        self.ff_image = nn.Linear(d, k)
        self.ff_ques = nn.Linear(d, k)
        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        self.ff_attention = nn.Linear(k, 1)

    def forward(self, vi, vq):
        # N * 196 * 1024 -> N * 196 * 512
        hi = self.ff_image(vi)
        # N * 1024 -> N * 512 -> N * 1 * 512
        hq = self.ff_ques(vq).unsqueeze(dim=1)
        # N * 196 * 512
        ha = torch.tanh(hi + hq)
        if getattr(self, 'dropout'):
            ha = self.dropout(ha)
        # N * 196 * 512 -> N * 196 * 1 -> N * 196
        ha = self.ff_attention(ha).squeeze(dim=2)
        pi = torch.softmax(ha, dim=-1)
        # (N * 196 * 1, N * 196 * 1024) -> N * 1024
        vi_attended = (pi.unsqueeze(dim=2) * vi).sum(dim=1)
        u = vi_attended + vq
        return u


class MultiModelFusion(nn.Module):
    def __init__(self, san_num_layers=3, n_hidden=1024, san_k=1024):
        super(MultiModelFusion, self).__init__()

        self.san_attention = nn.ModuleList([SanAttention(n_hidden, san_k) for _ in range(san_num_layers)])

    def forward(self, img_res, qus_res):
        vi = img_res
        u = qus_res[:, -1, :]  # 取最后一个节点的输出
        for att_layer in self.san_attention:
            u = att_layer(vi, u)
        return u


class ResNetSanModel(nn.Module):
    def __init__(self, ans_word_size, qus_embedding_dim=300, lstm_n_hidden=1024, lstm_num_layers=1,
                 dropout=.0, san_num_layers=3, san_k=512*4, linear_hid=256):
        super(ResNetSanModel, self).__init__()
        res_model = resnet50(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(res_model.children())[:7])  # 只取resnet的前七层，不需要后面的全连接层

        self.text_embedding_linear = nn.Sequential(
            nn.Linear(qus_embedding_dim, qus_embedding_dim),
            nn.LayerNorm(qus_embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.text_model = TextModel(qus_embedding_dim, lstm_n_hidden, lstm_num_layers, dropout)

        self.san_fusion = MultiModelFusion(san_num_layers, lstm_n_hidden, san_k)

        self.ans_model = nn.Sequential(
            nn.Linear(lstm_n_hidden, linear_hid),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(linear_hid, ans_word_size)
        )

    def forward(self, img, qus_embed):
        img_features = self.feature_extractor(img)
        img_features = img_features.flatten(2).transpose(1, 2).contiguous()

        qus_features = self.text_model(self.text_embedding_linear(qus_embed))
        fusion_features = self.san_fusion(img_features, qus_features)

        res = self.ans_model(fusion_features)

        return res


def resnet_san_base(**kwargs):
    model = ResNetSanModel(
        ans_word_size=kwargs["ans_size"], qus_embedding_dim=300, lstm_n_hidden=1024, lstm_num_layers=1,
        dropout=.0, san_num_layers=3, san_k=512, linear_hid=256
    )
    return model


if __name__ == '__main__':
    a = torch.randn([2, 3, 224, 224]).cuda()
    b = torch.randn([2, 20, 300]).cuda()
    model = ResNetSanModel(223).cuda()
    # torch.save(model.state_dict(), "1.pth")
    print(model(a, b).shape)
    print(sum(x.numel() for x in model.parameters()))  # 92345995


