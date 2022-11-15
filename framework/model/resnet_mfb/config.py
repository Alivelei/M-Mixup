# _*_ coding: utf-8 _*_

"""
    @Time : 2022/7/18 17:45 
    @Author : smile 笑
    @File : config.py
    @desc :
"""


class Cfgs(object):
    def __init__(self):
        self.HIGH_ORDER = False
        self.HIDDEN_SIZE = 512
        self.MFB_K = 5
        self.MFB_O = 1024
        self.LSTM_OUT_SIZE = 1024
        self.DROPOUT_R = 0.1
        self.I_GLIMPSES = 2
        self.Q_GLIMPSES = 2

        self.FRCN_FEAT_SIZE = 196
