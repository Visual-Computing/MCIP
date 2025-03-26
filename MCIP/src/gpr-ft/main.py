import argparse
import torch
import numpy as np
import random
import warnings

from eval import evaluate
from train import train
from train_with_text import train_with_text
from create_pseudo_captions import create_pseudo_captions
from extract_image_features import extract_image_features
from gpr_utils.config import get_config


def parse_option():
    parser = argparse.ArgumentParser('GPIR training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'train_with_text', 'eval', 'create_pseudo_captions', 'extract_image_features'])
    parser.add_argument('--gpu_id', type=int, default=-1)

    # ds ablation params
    parser.add_argument('--images_per_cat', type=int, default=-1)
    parser.add_argument('--c_frac', type=int, default=-1)

    #loss ablation params
    parser.add_argument('--arcface_scale', type=int, default=30)
    parser.add_argument('--arcface_margin', type=float, default=0.1)

    parser.add_argument('--name_sufix', type=str, default="")

    args, _ = parser.parse_known_args()

    config = get_config(args.cfg)

    config.name_sufix = args.name_sufix
    config.arcface_scale = args.arcface_scale
    config.arcface_margin = args.arcface_margin

    if args.gpu_id > -1:
        config.gpu_id = args.gpu_id
    
    if args.images_per_cat > -1:
        config.images_per_cat = args.images_per_cat
    
    if args.c_frac > -1:
        config.c_frac = args.c_frac
    
    if args.name_sufix != "":
        config.model += args.name_sufix
    config.mode = args.mode

    return config


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 

    config = parse_option()

    seed = 37
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    
    torch.backends.cudnn.benchmark = True
    
    if config.mode == "eval":
        evaluate(config)
    elif config.mode == "train":
        train(config)
    elif config.mode == "train_with_text":
        train_with_text(config)
    elif config.mode == "extract_image_features":
        extract_image_features(config)
    elif config.mode == "create_pseudo_captions":
        create_pseudo_captions(config)
    else:
        train(config)
        evaluate(config)