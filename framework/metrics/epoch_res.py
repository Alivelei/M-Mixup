# _*_ coding: utf-8 _*_

"""
    @Time : 2021/10/20 15:22 
    @Author : smile 笑
    @File : epoch_res.py
    @desc :
"""


import json
import os


# 获取到最小loss的模型
def get_epoch_min_mode(json_path, mode="loss"):
    any_epochs_res = json.load(open(json_path, encoding="utf-8"))

    if mode == "loss":
        epoch_loss = [epoch["m_loss"] for epoch in any_epochs_res]
        return min(epoch_loss)
    if mode == "acc":
        epoch_acc = [epoch["total_acc"] for epoch in any_epochs_res]
        return max(epoch_acc)


def save_epoch_res(json_path, epoch_res):
    if os.path.exists(json_path):
        any_epochs_res = json.load(open(json_path, encoding="utf-8"))
    else:
        any_epochs_res = []

    any_epochs_res.append(epoch_res)
    json.dump(any_epochs_res, open(json_path, "w", encoding="utf-8"), indent=1)

