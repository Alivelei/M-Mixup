# _*_ coding: utf-8 _*_

"""
    @Time : 2022/6/12 12:38 
    @Author : smile 笑
    @File : __init__.py.py
    @desc :
"""


from .model_interface import ModelInterfaceModule
from .pre_model_interface import PreTrainModuleInterface
from .clef_model_interface import CLEFModelInterfaceModule
from .model.resnet_ban.model import resnet_ban_base
from .model.resnet_san.model import resnet_san_base
from .model.resnet_mfb.model import resnet_mfb_base
from .model.mlp_struc.model import mlp_struc_small, mlp_struc_base, mlp_struc_large, mlp_struc_huge
from .model.transformer.model import transformer_base, transformer_large, transformer_huge
from .model.mlp_mixer.model import mlp_mixer_base, mlp_mixer_large, mlp_mixer_huge
from .model.mlp_con.model import mlp_con_base, mlp_con_large, mlp_con_huge
from .model.mlp_con2.model import mlp_con2_base, mlp_con2_large, mlp_con2_huge


def get_model_module(model_name):
    if model_name == "resnet_ban_base":
        return resnet_ban_base

    if model_name == "resnet_san_base":
        return resnet_san_base

    if model_name == "resnet_mfb_base":
        return resnet_mfb_base

    if model_name == "mlp_struc_small":
        return mlp_struc_small

    if model_name == "mlp_struc_base":
        return mlp_struc_base

    if model_name == "mlp_struc_large":
        return mlp_struc_large

    if model_name == "mlp_struc_huge":
        return mlp_struc_huge

    if model_name == "transformer_base":
        return transformer_base

    if model_name == "transformer_large":
        return transformer_large

    if model_name == "transformer_huge":
        return transformer_huge

    if model_name == "mlp_mixer_base":
        return mlp_mixer_base

    if model_name == "mlp_mixer_large":
        return mlp_mixer_large

    if model_name == "mlp_mixer_huge":
        return mlp_mixer_huge

    if model_name == "mlp_con_base":
        return mlp_con_base

    if model_name == "mlp_con_large":
        return mlp_con_large

    if model_name == "mlp_con_huge":
        return mlp_con_huge

    if model_name == "mlp_con2_base":
        return mlp_con_base

    if model_name == "mlp_con2_large":
        return mlp_con_large

    if model_name == "mlp_con2_huge":
        return mlp_con_huge


def get_pre_model_module(pre_model_name):
    if pre_model_name == "m3ae_coformer_base":
        pass



