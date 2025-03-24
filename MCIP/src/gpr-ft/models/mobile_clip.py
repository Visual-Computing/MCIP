import open_clip
import torch
from torch import nn



def get_mclip_model(model_key, output_dim=-1):

    clip_model = None
    
    clip_model, _, preprocess = open_clip.create_model_and_transforms("MobileCLIP-S2", pretrained="datacompdr", cache_dir="/mnt/bigdata/networks/TorchHub/")
    emb_dim = 512
    clip_model.visual.output_dim = emb_dim
   
    # from mobileclip.modules.common.mobileone import reparameterize_model
    # clip_model = reparameterize_model(clip_model)

    print("created mobile_clip")
    tokenizer = open_clip.get_tokenizer('MobileCLIP-S2')
    clip_model.text.eval()
    clip_model.set_grad_checkpointing(True)
    return clip_model, tokenizer