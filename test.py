# _*_ coding: utf-8 _*_

"""
    @Time : 2022/6/8 14:53
    @Author : smile 笑
    @File : train.py
    @desc :
"""


import torch
from torch import nn
import argparse
from data import Word2Sequence, Sort2Id
from data import DataInterfaceModule, SlakeDatasetModule, RadDatasetModule
from framework import ModelInterfaceModule, get_model_module
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import os


def create_model_module(args):
    model_name = args.model_select + "_" + args.model_size
    model_func = get_model_module(model_name)

    if args.select_data == "slake" or args.select_data == "rad":
        model = ModelInterfaceModule.load_from_checkpoint(args.test_best_model_path, model=model_func, args=args, strict=False)

    args.default_root_dir = os.path.join(args.default_root_dir, model_name + "/")

    return model, args


def dataset_select(args):
    if args.select_data == "slake":
        db = DataInterfaceModule(SlakeDatasetModule, args)
    if args.select_data == "rad":
        db = DataInterfaceModule(RadDatasetModule, args)

    # 用来获取是哪个版本的模型
    logger = TensorBoardLogger(
        save_dir=args.default_root_dir,
        version=args.select_data + "_" + str(args.version),
        name="test_logs"
    )

    return db, logger


def main(args):
    seed_everything(args.random_seed, True)  # 设置随机数种子

    model, args = create_model_module(args)

    db, logger = dataset_select(args)

    trainer = Trainer(
        gpus=args.device_ids,
        strategy="ddp",
        default_root_dir=args.default_root_dir,
        logger=logger,
        sync_batchnorm=True,
        gradient_clip_val=0.5,  # 加入梯度裁剪
    )

    trainer.test(model, db)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default="test")
    parser.add_argument("--model_select", default="mlp_struc", choices=["mlp_struc"])
    parser.add_argument("--model_size", default="base", choices=["base", "large", "huge"])
    parser.add_argument("--load_pre", default=False, choices=[True, False])
    parser.add_argument("--select_data", default="slake", choices=["slake", "rad"])
    parser.add_argument("--version", default=0)

    # configure
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--device_ids", default=[0, 1])
    parser.add_argument("--num_workers", default=4, type=int)

    # configure
    parser.add_argument("--epochs", default=10000, type=int)
    parser.add_argument("--qus_seq_len", default=20, type=int)
    parser.add_argument("--answer_open", default=0, type=int)
    parser.add_argument("--answer_close", default=1, type=int)

    parser.add_argument("--test_best_model_path", default="./save/model/test_best_model")
    parser.add_argument("--default_root_dir", default="./save/")

    # all dataset
    parser.add_argument("--qus_word_size", default=1371, type=int)
    parser.add_argument("--qus_glove_path", default="./save/embedding/all_qus_glove_emb_300d.npy")
    # slake dataset
    parser.add_argument("--slake_qus_ws_path", default="./save/ws/slake_qus_ws.pkl")
    parser.add_argument("--slake_ans_ws_path", default="./save/ws/slake_ans_ws.pkl")
    parser.add_argument("--slake_ans_glove_path", default="./save/embedding/slake_ans_glove_emb_300d.npy")
    parser.add_argument("--slake_qus_glove_path", default="./save/embedding/slake_qus_glove_emb_300d.npy")
    parser.add_argument("--slake_qus_word_size", default=305, type=int)
    parser.add_argument("--slake_ans_word_size", default=223, type=int)
    parser.add_argument("--slake_train_dataset_path", default="./data/ref/Slake1.0/train.json")
    parser.add_argument("--slake_test_dataset_path", default="./data/ref/Slake1.0/test.json")
    parser.add_argument("--slake_dataset_xm_path", default="./data/ref/Slake1.0/imgs/xmlab")

    # rad dataset
    parser.add_argument("--rad_qus_word_size", default=1229, type=int)
    parser.add_argument("--rad_ans_word_size", default=476, type=int)
    parser.add_argument("--rad_qus_ws_path", default="./save/ws/rad_qus_ws.pkl")
    parser.add_argument("--rad_ans_ws_path", default="./save/ws/rad_ans_ws.pkl")
    parser.add_argument("--rad_qus_glove_path", default="./save/embedding/rad_qus_glove_emb_300d.npy")
    parser.add_argument("--rad_ans_glove_path", default="./save/embedding/rad_ans_glove_emb_300d.npy")
    parser.add_argument("--rad_images_path", default="./data/ref/rad/images")
    parser.add_argument("--rad_train_dataset_path", default="./data/ref/rad/trainset.json")
    parser.add_argument("--rad_test_dataset_path", default="./data/ref/rad/testset.json")

    # model
    parser.add_argument("--learning_rate", default=0.005, type=float)
    parser.add_argument("--weights_decay", default=0.05, type=float)
    parser.add_argument("--random_seed", default=1024, type=int)

    # image
    parser.add_argument("--img_height", default=224, type=int)
    parser.add_argument("--img_width", default=224, type=int)
    parser.add_argument("--slake_img_mean", default=[0.38026, 0.38026, 0.38026])
    parser.add_argument("--slake_img_std", default=[0.2979, 0.2979, 0.2979])
    parser.add_argument("--rad_img_mean", default=[0.33640, 0.33630, 0.33610])
    parser.add_argument("--rad_img_std", default=[0.29664, 0.29659, 0.29642])

    args = parser.parse_args()

    main(args)

    # model = ConvMLPVQASystem(args).cuda()
    # a = torch.randn([2, 3, 224, 224]).cuda()
    # b = torch.ones([2, 20], dtype=torch.int64).cuda()
    #
    # print(model(a, b).shape)
    # print(sum(x.numel() for x in model.parameters()))  # small 8183576
    # torch.save(model.state_dict(), "1.pth")  # 12层 31M

    # dl = DataInterfaceModule(SlakeDatasetModule, args)
    # dl.setup(stage="test")
    #
    # for idx, (img, qus, ans, ans_flag) in enumerate(dl.test_dataloader()):
    #     print(img.shape)
    #     print(qus.shape)
    #     print(ans.shape)
    #     print(ans_flag.shape)
    # dl.teardown(stage="test")

