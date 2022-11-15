# _*_ coding: utf-8 _*_

"""
    @Time : 2021/11/6 18:31 
    @Author : smile 笑
    @File : method.py
    @desc :
"""


import torch
import math
import random
import os
import numpy as np


# 加入随机数种子
def set_seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# 加载对比学习学习到的模型
def load_pretrained_model(model, pre_trained_path):
    pretext_model = torch.load(pre_trained_path)  # 提取出预训练模型的参数
    model_dict = model.state_dict()  # 得到训练模型的参数

    state_dict = {k: v for k, v in pretext_model.items() if k in model_dict.keys()}  # 将预训练模型更新到自己定义的模型中
    model_dict.update(state_dict)

    # 对预训练模型参数设置微调
    for param in model.parameters():
        param.requires_grad = True

    return model


def compute_batch_score(ans_pred, ans, ans_type):
    """
    :param ans_pred: [batch_size]
    :param ans: [batch_size]
    :param ans_type: [batch_size]
    :return:
    """
    open_idx = torch.where(ans_type == 0)[0]  # 计算一个batch中的open的索引
    close_idx = torch.where(ans_type == 1)[0]  # 计算close的索引

    # 使用得到的open索引定位ans、ans_pred中对应open的预测值
    open_ans_pred = torch.index_select(ans_pred, 0, open_idx)
    open_ans = torch.index_select(ans, 0, open_idx)

    close_ans_pred = torch.index_select(ans_pred, 0, close_idx)
    close_ans = torch.index_select(ans, 0, close_idx)

    # 计算长度
    open_len = len(open_idx)
    close_len = len(close_idx)
    total_len = ans.size(0)

    # 计算open、close的相等的数量
    open_batch_acc = (open_ans_pred == open_ans).sum()
    close_batch_acc = (close_ans_pred == close_ans).sum()

    total_batch_acc = open_batch_acc + close_batch_acc

    return open_batch_acc, close_batch_acc, total_batch_acc, open_len, close_len, total_len


def clef_compute_batch_score(ans_pred, ans, ans_type):
    """
    :param ans_pred: [batch_size]
    :param ans: [batch_size]
    :param ans_type: [batch_size]
    :return:
    """
    organ_idx = torch.where(ans_type == 0)[0]  # 计算一个batch中的open的索引
    modality_idx = torch.where(ans_type == 1)[0]  # 计算close的索引
    plane_idx = torch.where(ans_type == 2)[0]  # 计算一个batch中的open的索引
    abnormality_idx = torch.where(ans_type == 3)[0]  # 计算close的索引

    # 使用得到的open索引定位ans、ans_pred中对应open的预测值
    organ_ans_pred = torch.index_select(ans_pred, 0, organ_idx)
    organ_ans = torch.index_select(ans, 0, organ_idx)

    modality_ans_pred = torch.index_select(ans_pred, 0, modality_idx)
    modality_ans = torch.index_select(ans, 0, modality_idx)

    plane_ans_pred = torch.index_select(ans_pred, 0, plane_idx)
    plane_ans = torch.index_select(ans, 0, plane_idx)

    abnormality_ans_pred = torch.index_select(ans_pred, 0, abnormality_idx)
    abnormality_ans = torch.index_select(ans, 0, abnormality_idx)

    # 计算长度
    organ_len = len(organ_idx)
    modality_len = len(modality_idx)
    plane_len = len(plane_idx)
    abnormality_len = len(abnormality_idx)
    total_len = ans.size(0)

    # 计算open、close的相等的数量
    organ_batch_acc = (organ_ans_pred == organ_ans).sum().cpu().item()
    modality_batch_acc = (modality_ans_pred == modality_ans).sum().cpu().item()
    plane_batch_acc = (plane_ans_pred == plane_ans).sum().cpu().item()
    abnormality_batch_acc = (abnormality_ans_pred == abnormality_ans).sum().cpu().item()

    total_batch_acc = organ_batch_acc + modality_batch_acc + plane_batch_acc + abnormality_batch_acc

    return organ_batch_acc, modality_batch_acc, plane_batch_acc, abnormality_batch_acc, total_batch_acc, organ_len, modality_len, plane_len, abnormality_len, total_len



