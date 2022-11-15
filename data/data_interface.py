# _*_ coding: utf-8 _*_

"""
    @Time : 2022/6/12 12:51 
    @Author : smile 笑
    @File : data_interface.py
    @desc :
"""


from pytorch_lightning import LightningDataModule
import torch
from torch import nn
from torch.utils.data import DataLoader
from data.word_sequence import Word2Sequence


class DataInterfaceModule(LightningDataModule):
    def __init__(self, dataset, args):
        super(DataInterfaceModule, self).__init__()

        self.args = args
        self.dataset = dataset
        self.qus_embed_flag = args.qus_embed_flag

    def setup(self, stage=None) -> None:
        if stage == "fit" or stage is None:
            if self.args.select_data == "slake":
                self.train_dataset = self.dataset(self.args, self.args.slake_train_dataset_path, "train")
                self.val_dataset = self.dataset(self.args, self.args.slake_test_dataset_path, "val")
            elif self.args.select_data == "rad":
                self.train_dataset = self.dataset(self.args, self.args.rad_train_dataset_path, "train")
                self.val_dataset = self.dataset(self.args, self.args.rad_test_dataset_path, "val")

        elif stage == "test" or stage is None:
            if self.args.select_data == "slake":
                self.test_dataset = self.dataset(self.args, self.args.slake_test_dataset_path, "test")
            elif self.args.select_data == "rad":
                self.test_dataset = self.dataset(self.args, self.args.rad_test_dataset_path, "test")

    def collate_fn(self, batch):
        image, qus_id, ans_id, ans_type_id = list(zip(*batch))
        image = torch.stack(image)
        if self.qus_embed_flag:
            qus_id = torch.stack(qus_id)
        else:
            qus_id = torch.tensor(qus_id, dtype=torch.int64)

        ans_id = torch.tensor(ans_id, dtype=torch.int64)
        ans_type_id = torch.tensor(ans_type_id, dtype=torch.int64)

        return image, qus_id, ans_id, ans_type_id

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=True, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=False, collate_fn=self.collate_fn)









