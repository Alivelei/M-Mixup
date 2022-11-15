# _*_ coding: utf-8 _*_

"""
    @Time : 2022/6/4 20:50
    @Author : smile 笑
    @File : model_interface.py
    @desc :
"""


import pytorch_lightning as pl
import torch
from torch import nn
import numpy as np
from framework.metrics.method import compute_batch_score
from framework.metrics.mixup import m_mix_up
from framework.metrics.epoch_res import save_epoch_res
import torch.distributed as dist


class ModelInterfaceModule(pl.LightningModule):
    def __init__(self, model, args):
        super(ModelInterfaceModule, self).__init__()

        # self.hparams = args
        self.save_hyperparameters()

        if args.load_pre:
            if args.select_data == "slake":
                ans_word_size = args.slake_ans_word_size
            elif args.select_data == "rad":
                ans_word_size = args.rad_ans_word_size
            self.model = model(glove_path=args.qus_glove_path, word_size=args.qus_word_size, ans_size=ans_word_size, qus_embed_flag=args.qus_embed_flag)
        else:
            if args.select_data == "slake":
                qus_glove_path = args.slake_qus_glove_path
                qus_word_size = args.slake_qus_word_size
                ans_word_size = args.slake_ans_word_size
            elif args.select_data == "rad":
                qus_glove_path = args.rad_qus_glove_path
                qus_word_size = args.rad_qus_word_size
                ans_word_size = args.rad_ans_word_size
            self.model = model(glove_path=qus_glove_path, word_size=qus_word_size, ans_size=ans_word_size, qus_embed_flag=args.qus_embed_flag)

        if args.mix_choices == "m_mix_up":
            self.mix_up = m_mix_up

    def on_train_epoch_start(self):
        # 保存每个epoch的close、open准确值和数量
        self.train_e_close_acc = self.train_e_open_acc = self.train_e_total_acc = 0
        self.train_e_close_nums = self.train_e_open_nums = self.train_e_total_nums = 0

    def on_validation_epoch_start(self):
        self.test_e_close_acc = self.test_e_open_acc = self.test_e_total_acc = 0
        self.test_e_close_nums = self.test_e_open_nums = self.test_e_total_nums = 0

    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def forward(self, img, qus):
        return self.model(img, qus)

    def shares_step_compute_value(self, b_open_acc, b_close_acc, b_total_acc, b_open_nums, b_close_nums, b_total_nums):
        all_b_open_acc = torch.tensor(b_open_acc).cuda()
        all_b_close_acc = torch.tensor(b_close_acc).cuda()
        all_b_total_acc = torch.tensor(b_total_acc).cuda()
        all_b_open_nums = torch.tensor(b_open_nums).cuda()
        all_b_close_nums = torch.tensor(b_close_nums).cuda()
        all_b_total_nums = torch.tensor(b_total_nums).cuda()

        dist.all_reduce(all_b_open_acc)  # 对每个gpu上的value求和
        dist.all_reduce(all_b_close_acc)
        dist.all_reduce(all_b_total_acc)
        dist.all_reduce(all_b_open_nums)
        dist.all_reduce(all_b_close_nums)
        dist.all_reduce(all_b_total_nums)

        b_open_acc = all_b_open_acc / (all_b_open_nums + 1e-10)
        b_close_acc = all_b_close_acc / (all_b_close_nums + 1e-10)  # 加一个足够小的数防止出现0
        b_total_acc = all_b_total_acc / (all_b_total_nums + 1e-10)

        return b_open_acc, b_close_acc, b_total_acc

    def training_step(self, batch, batch_idx):
        image, qus, ans, ans_type = batch

        criterion = nn.CrossEntropyLoss()
        if np.random.random() <= self.hparams.args.mix_probability:
            ans_loss, predict_ans = self.mix_up(self, image, qus, ans, criterion, self.hparams.args.mix_alpha1, self.hparams.args.mix_alpha2)
        else:
            predict_ans = self(image, qus)
            ans_loss = criterion(predict_ans, ans.view(-1))

        _, ans_pred = predict_ans.max(-1)  # 取出预测值

        # 计算每个batch的准确率
        open_b_acc, close_b_acc, total_b_acc, open_len, close_len, total_b_len = compute_batch_score(ans_pred, ans.view(-1), ans_type)

        # 计算open、close、total的每个batch后平均精确率
        self.train_e_open_acc += open_b_acc
        self.train_e_open_nums += open_len
        self.train_e_close_acc += close_b_acc
        self.train_e_close_nums += close_len
        self.train_e_total_acc += total_b_acc
        self.train_e_total_nums += total_b_len

        train_open_acc, train_close_acc, train_total_acc = self.shares_step_compute_value(self.train_e_open_acc, self.train_e_close_acc, self.train_e_total_acc, self.train_e_open_nums, self.train_e_close_nums, self.train_e_total_nums)

        self.log("train_loss_step", ans_loss, prog_bar=True, on_epoch=True)
        self.log("train_open_acc_step", train_open_acc, prog_bar=True, on_epoch=False)
        self.log("train_close_acc_step", train_close_acc, prog_bar=True, on_epoch=False)
        self.log("train_total_acc_step", train_total_acc, prog_bar=True, on_epoch=False)

        return {"open_acc": train_open_acc, "close_acc": train_close_acc, "total_acc": train_total_acc, "loss": ans_loss}

    def training_epoch_end(self, outputs):
        # outputs 保存了每个step的值
        open_acc = outputs[-1]["open_acc"]  # 最后一个保存的就是平均好的准确率
        close_acc = outputs[-1]["close_acc"]
        total_acc = outputs[-1]["total_acc"]
        ans_loss = outputs[-1]["loss"]

        self.log("train_open_acc", open_acc, on_step=False, on_epoch=True)
        self.log("train_close_acc", close_acc, on_step=False, on_epoch=True)
        self.log("train_total_acc", total_acc, on_step=False, on_epoch=True)
        self.log("train_loss", ans_loss, on_step=False, on_epoch=True)

        if dist.get_rank() == 0:
            # 将每轮模型的结果保存在json中
            state_dict = {"epoch": self.current_epoch, "train_m_loss": float(ans_loss), "open_acc": float(open_acc), "close_acc": float(close_acc), "total_acc": float(total_acc)}
            save_epoch_res(self.hparams.args.train_epoch_effect_path, state_dict)

    def shares_validation_code(self, batch, batch_idx):
        image, qus, ans, ans_type = batch

        criterion = nn.CrossEntropyLoss()

        predict_ans = self(image, qus)
        ans_loss = criterion(predict_ans, ans.view(-1))

        _, ans_pred = predict_ans.max(-1)  # 取出预测值

        open_b_acc, close_b_acc, total_b_acc, open_len, close_len, total_b_len = compute_batch_score(ans_pred, ans.view(-1), ans_type)

        # 计算open、close、total的每个batch后平均精确率
        self.test_e_open_acc += open_b_acc
        self.test_e_open_nums += open_len
        self.test_e_close_acc += close_b_acc
        self.test_e_close_nums += close_len
        self.test_e_total_acc += total_b_acc
        self.test_e_total_nums += total_b_len

        test_open_acc, test_close_acc, test_total_acc = self.shares_step_compute_value(self.test_e_open_acc, self.test_e_close_acc, self.test_e_total_acc, self.test_e_open_nums, self.test_e_close_nums, self.test_e_total_nums)

        return test_open_acc, test_close_acc, test_total_acc, ans_loss

    def validation_step(self, batch, batch_idx):
        open_acc, close_acc, total_acc, ans_loss = self.shares_validation_code(batch, batch_idx)

        return {"open_acc": open_acc, "close_acc": close_acc, "total_acc": total_acc, "ans_loss": ans_loss}

    def validation_epoch_end(self, outputs):
        # outputs 保存了每个step的值
        open_acc = outputs[-1]["open_acc"]  # 最后一个保存的就是平均好的准确率
        close_acc = outputs[-1]["close_acc"]
        total_acc = outputs[-1]["total_acc"]
        ans_loss = outputs[-1]["ans_loss"]

        self.log("test_open_acc", open_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_close_acc", close_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_total_acc", total_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_loss", ans_loss, on_step=False, on_epoch=True, prog_bar=True)

        if dist.get_rank() == 0:
            # 将每轮模型的结果保存在json中  float 类型
            state_dict = {"epoch": self.current_epoch, "test_m_loss": float(ans_loss), "open_acc": float(open_acc), "close_acc": float(close_acc), "total_acc": float(total_acc)}
            save_epoch_res(self.hparams.args.test_epoch_effect_path, state_dict)

    def test_step(self, batch, batch_idx):
        open_acc, close_acc, total_acc, ans_loss = self.shares_validation_code(batch, batch_idx)
        return {"open_acc": open_acc, "close_acc": close_acc, "total_acc": total_acc, "ans_loss": ans_loss}

    def test_epoch_end(self, outputs):
        # outputs 保存了每个step的值
        open_acc = outputs[-1]["open_acc"]  # 最后一个保存的就是平均好的准确率
        close_acc = outputs[-1]["close_acc"]
        total_acc = outputs[-1]["total_acc"]
        ans_loss = outputs[-1]["ans_loss"]

        self.log("test_open_acc", open_acc, on_step=False, on_epoch=True)
        self.log("test_close_acc", close_acc, on_step=False, on_epoch=True)
        self.log("test_total_acc", total_acc, on_step=False, on_epoch=True)
        self.log("test_loss", ans_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.args.learning_rate, weight_decay=self.hparams.args.weights_decay)

        step_lr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, self.hparams.args.epochs)  # 余弦退火

        optim_dict = {"optimizer": optimizer, "lr_scheduler": step_lr}

        return optim_dict


