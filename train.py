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
from torch.utils.data import DataLoader
from data import DataInterfaceModule, SlakeDatasetModule, RadDatasetModule
from framework import ModelInterfaceModule, get_model_module
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
import os


def mkdir_println(dir_path, println):
    if os.path.exists(dir_path):
        print(println + "文件夹已创建.")
    else:
        os.mkdir(dir_path)
        print(println + "文件夹创建成功.")


def create_model_module(args):
    model_name = args.model_select + "_" + args.model_size
    model_func = get_model_module(model_name)

    if args.load_pre:
        if args.select_data == "slake" or args.select_data == "rad":
            model = ModelInterfaceModule.load_from_checkpoint(args.pre_best_model_path, model=model_func, args=args, strict=False)
    else:
        if args.select_data == "slake" or args.select_data == "rad":
            model = ModelInterfaceModule(model=model_func, args=args)

    args.default_root_dir = os.path.join(args.default_root_dir, model_name + "/")
    mkdir_println(args.default_root_dir, model_name + "根")  # 创建模型根文件夹

    args.train_epoch_effect_path = os.path.join(args.default_root_dir, args.train_epoch_effect_path)
    args.test_epoch_effect_path = os.path.join(args.default_root_dir, args.test_epoch_effect_path)
    mkdir_println(args.train_epoch_effect_path, model_name + "_param")  # 创建根文件夹下的param

    args.best_model_path = os.path.join(args.default_root_dir, args.best_model_path)
    mkdir_println(args.best_model_path, model_name + "_train_best_model")  # 创建根文件夹下的训练集最佳模型文件夹

    args.test_best_model_path = os.path.join(args.default_root_dir, args.test_best_model_path)
    mkdir_println(args.test_best_model_path, model_name + "_test_best_model")  # 创建根文件夹下测试集最佳模型文件夹

    return model, args


def dataset_select(args):
    if args.select_data == "slake":
        db = DataInterfaceModule(SlakeDatasetModule, args)
    if args.select_data == "rad":
        db = DataInterfaceModule(RadDatasetModule, args)

    # 用来获取是哪个版本的模型
    logger = TensorBoardLogger(
        save_dir=args.default_root_dir,
        version=args.model_select + "_" + args.select_data + "_" + str(args.version),
        name="train_logs"
    )

    return db, logger


def main(args):
    seed_everything(args.random_seed, True)  # 设置随机数种子

    model, args = create_model_module(args)
    db, logger = dataset_select(args)

    train_checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        dirpath=os.path.join(args.best_model_path, str(logger.version)),
        filename="{train_loss:.4f}",
        save_last=True
    )

    test_checkpoint_callback = ModelCheckpoint(
        monitor="test_total_acc",
        mode="max",
        save_top_k=3,
        dirpath=os.path.join(args.test_best_model_path, str(logger.version)),
        filename="{test_total_acc:.4f}",
        save_weights_only=True,
        # save_last=True
    )

    # 构建json保存路径
    epoch_effect_path = os.path.join(args.train_epoch_effect_path, str(logger.version))
    mkdir_println(epoch_effect_path, "model_param_version")  # 创建param下的version文件夹
    args.train_epoch_effect_path = os.path.join(epoch_effect_path, "train_epoch_effect.json")
    args.test_epoch_effect_path = os.path.join(epoch_effect_path, "test_epoch_effect.json")

    trainer = Trainer(
        gpus=args.device_ids,
        max_epochs=args.epochs,
        strategy="ddp",
        # checkpoint_callback=True,  # 将被移除用下面这个
        enable_checkpointing=True,
        check_val_every_n_epoch=5,
        logger=logger,
        # progress_bar_refresh_rate=5,
        sync_batchnorm=True,
        callbacks=[train_checkpoint_callback, test_checkpoint_callback],
        gradient_clip_val=0.5,  # 加入梯度裁剪
        resume_from_checkpoint=args.resume_from_checkpoint if os.path.exists(args.resume_from_checkpoint) else None,
    )

    trainer.fit(model, db)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", default="train")
    parser.add_argument("--model_select", default="mlp_struc", choices=["resnet_ban", "resnet_san", "resnet_mfb", "mlp_struc", "transformer", "mlp_mixer"])
    parser.add_argument("--model_size", default="base", choices=["small", "base", "large", "huge"])
    parser.add_argument("--select_data", default="slake", choices=["slake", "rad"])
    parser.add_argument("--qus_embed_flag", default=True, choices=[True, False])
    parser.add_argument("--load_pre", default=False, choices=[True, False])
    parser.add_argument("--mix_choices", default="m_mix_up", choices=["m_mix_up"])
    parser.add_argument("--mix_alpha1", default=5)
    parser.add_argument("--mix_alpha2", default=1)
    parser.add_argument("--version", default="con_base")

    # configure
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--device_ids", default=[0, 1])
    parser.add_argument("--num_workers", default=4, type=int)

    # constant_image
    parser.add_argument("--mix_probability", default=1, type=float)
    parser.add_argument("--img_rotation", default=15, type=int)
    parser.add_argument("--resized_crop_left", default=0.2, type=float)
    parser.add_argument("--resized_crop_right", default=1.0, type=float)
    parser.add_argument("--blur", default=[0.1, 2.0])
    parser.add_argument("--b_size", default=[5, 5])
    parser.add_argument("--blur_p", default=0.5, type=float)
    parser.add_argument("--apply_p", default=0.8, type=float)
    parser.add_argument("--img_flip", default=0.5, type=float)
    parser.add_argument("--brightness", default=0.4, type=float)
    parser.add_argument("--contrast", default=0.4, type=float)
    parser.add_argument("--saturation", default=0.4, type=float)
    parser.add_argument("--hue", default=0.4, type=float)
    parser.add_argument("--grayscale", default=0.2, type=float)

    # configure
    parser.add_argument("--epochs", default=10000, type=int)
    parser.add_argument("--qus_seq_len", default=20, type=int)
    parser.add_argument("--answer_open", default=0, type=int)
    parser.add_argument("--answer_close", default=1, type=int)
    parser.add_argument("--train_epoch_effect_path", default="param")
    parser.add_argument("--test_epoch_effect_path", default="param")

    parser.add_argument("--best_model_path", default="best_model")
    parser.add_argument("--test_best_model_path", default="test_best_model")
    parser.add_argument("--default_root_dir", default="./save/")
    parser.add_argument("--resume_from_checkpoint", default="./save/model/best_model/last.ckpt")
    parser.add_argument("--pre_best_model_path", default="./save/model/pre_best_model/0/train_loss=0.2651.ckpt")

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
    # dl.setup(stage="fit")
    #
    # for idx, (img, qus, ans, ans_flag) in enumerate(dl.train_dataloader()):
    #     print(img.shape)
    #     print(qus)
    #     print(ans.shape)
    #     print(ans_flag.shape)
    # dl.teardown(stage="fit")

