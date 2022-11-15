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
from framework.model.resnet_ban.ban import BiAttention, BanBiResNet


class CBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=2, padding=1):
        super(CBR, self).__init__()
        self.cbl = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.cbl(x)


class TextModel(nn.Module):
    def __init__(self, qus_embedding_dim=300, lstm_n_hidden=1024, lstm_num_layers=1, dropout=.0):
        super(TextModel, self).__init__()

        self.qus_model = nn.LSTM(input_size=qus_embedding_dim, hidden_size=lstm_n_hidden, num_layers=lstm_num_layers, batch_first=True, dropout=dropout)

    def forward(self, qus_embedding):
        text_embedding_res, _ = self.qus_model(qus_embedding)  # 得到序列的输出 [b, 12, 1024]

        return text_embedding_res


class BanMultiModelFusion(nn.Module):
    def __init__(self, x_dim=256, y_dim=1024, z_dim=1024, glimpse=4, v_dim=256, hid_dim=1024):
        super(BanMultiModelFusion, self).__init__()

        self.bi_attn = BiAttention(x_dim, y_dim, z_dim, glimpse)
        self.ban_bi = BanBiResNet(v_dim, hid_dim, glimpse)

    def forward(self, img_res, qus_res):
        p, logits = self.bi_attn(img_res, qus_res)
        res = self.ban_bi(img_res, qus_res, p)

        return res


class ResNetBanModel(nn.Module):
    def __init__(self, ans_word_size, out_channel=256, qus_embedding_dim=300, lstm_n_hidden=1024, lstm_num_layers=1,
                 dropout=.0, x_dim=256, y_dim=1024, z_dim=1024, glimpse=4, v_dim=256, hid_dim=1024):
        super(ResNetBanModel, self).__init__()
        res_model = resnet50(pretrained=False)
        self.feature_extractor = nn.Sequential(*list(res_model.children())[:7])  # 只取resnet的前七层，不需要后面的全连接层
        self.down_pooling = nn.Sequential(
            CBR(1024, out_channel, stride=1),  # resnet的输出结果[b, 1024, 14, 14]
        )

        self.text_embedding_linear = nn.Sequential(
            nn.Linear(qus_embedding_dim, qus_embedding_dim),
            nn.LayerNorm(qus_embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.text_model = TextModel(qus_embedding_dim, lstm_n_hidden, lstm_num_layers, dropout)

        self.ban_fusion = BanMultiModelFusion(x_dim, y_dim, z_dim, glimpse, v_dim, hid_dim)

        self.ans_model = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hid_dim, ans_word_size)
        )

    def forward(self, img, qus_embed):
        img_features = self.down_pooling(self.feature_extractor(img))
        img_features = img_features.flatten(2).transpose(1, 2).contiguous()

        qus_features = self.text_model(self.text_embedding_linear(qus_embed))
        fusion_features = self.ban_fusion(img_features, qus_features)

        res = self.ans_model(fusion_features)

        return res


def resnet_ban_base(**kwargs):
    model = ResNetBanModel(
        ans_word_size=kwargs["ans_size"], out_channel=256, qus_embedding_dim=300, lstm_n_hidden=1024, lstm_num_layers=1,
        dropout=.0, x_dim=256, y_dim=1024, z_dim=1024, glimpse=4, v_dim=256, hid_dim=1024
    )
    return model


if __name__ == '__main__':
    a = torch.randn([2, 3, 224, 224]).cuda()
    b = torch.randn([2, 20, 300]).cuda()
    model = ResNetBanModel(223).cuda()
    # torch.save(model.state_dict(), "1.pth")
    print(model(a, b).shape)
    print(sum(x.numel() for x in model.parameters()))  # 92345995


