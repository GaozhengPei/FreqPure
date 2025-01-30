import os
import shutil
import argparse

import torch
import torchvision.transforms as transforms

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
