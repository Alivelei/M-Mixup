# _*_ coding: utf-8 _*_

"""
    @Time : 2021/11/4 18:48 
    @Author : smile 笑
    @File : mixup.py
    @desc :
"""


import numpy as np
import torch


def m_mix_up(model, image, qus_embedding, ans, criterion, mix_alpha1=5, mix_alpha2=1):
    # 1.mixup 使用mixup数据增强方式进行数据增强
    lam = np.random.beta(mix_alpha1, mix_alpha2)

    # randperm返回1~images.size(0)的一个随机排列
    index = torch.randperm(image.size(0)).cuda()
    inputs_img = lam * image + (1 - lam) * image[index, :]
    inputs_qus = lam * qus_embedding + (1 - lam) * qus_embedding[index, :]
    ans_a, ans_b = ans, ans[index]
    predict_ans = model(inputs_img, inputs_qus)

    ans_loss = lam * criterion(predict_ans, ans_a.view(-1)) + (1 - lam) * criterion(predict_ans, ans_b.view(-1))

    return ans_loss, predict_ans



