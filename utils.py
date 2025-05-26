import os
import shutil
import argparse

import torch
import torchvision.transforms as transforms
import random
import numpy as np

def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def restore_checkpoint(ckpt_dir, state, device):
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']

def diff2clf(x, is_imagenet=False): 
    # [-1, 1] to [0, 1]
    return torch.clamp((x / 2) + 0.5,0,1) 

def clf2diff(x):
    # [0, 1] to [-1, 1]
    return torch.clamp((x - 0.5) * 2,-1,1)

def normalize(x):
    # Normalization for ImageNet
    return transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])(x)
#---------------------------------------------------#
#   设置种子
#---------------------------------------------------#
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def update_state_dict(state_dict, idx_start=9):

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[idx_start:]  # remove 'module.0.' of dataparallel
        new_state_dict[name]=v

    return new_state_dict

def load_wideresnet_70_16():
    print('using cifar10 wideresnet-70-16 (dm_wrn-70-16)...')
    from robustbench.model_zoo.architectures.dm_wide_resnet import DMWideResNet, Swish

    model = DMWideResNet(num_classes=10, depth=70, width=16, activation_fn=Swish)  # pixel in [0, 1]

    model_path = './models/cifar10/Linf/weights-best.pt'
    print(f"=> loading wideresnet-70-16 checkpoint '{model_path}'")
    model.load_state_dict(update_state_dict(torch.load(model_path)['model_state_dict']))
    model.eval()
    print(f"=> loaded wideresnet-70-16 checkpoint")
    return model
