import open_clip
import torch
from torch import nn


class EmbeddingModel(torch.nn.Module):
    
    def __init__(self, backbone, use_proj=True, output_dim=-1, emb_dim=1280):
        super().__init__()
        self.backbone = backbone
        
        if not use_proj:
            self.backbone.proj = None
            self.emb_dim = emb_dim
        else:
            self.emb_dim = backbone.output_dim
        
        self.output_dim = output_dim
        if output_dim > 0:
            self.linear =  nn.Linear(self.emb_dim, output_dim)
            self.emb_dim = output_dim
    
    def forward(self, x):
        
        fv = self.backbone(x)
        if self.output_dim > 0:
            fv = self.linear(fv)

        return fv

def get_oclip_model(model_key, output_dim=-1, train_with_text=False):
    #print(model_key)
    clip_model = None
    if  "OClipVitH" in model_key:
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-H-14", pretrained="laion2b_s32b_b79k")
        emb_dim = 1280

    if  "OClipConvNextB" in model_key:
        clip_model, _, _ = open_clip.create_model_and_transforms("convnext_base", pretrained="laion400m_s13b_b51k")
        emb_dim = 512

    if  "OClipVitL" in model_key:
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-14", pretrained="datacomp_xl_s13b_b90k")
        emb_dim = 1024

    if  "OClipRoberta" in model_key:
        clip_model, _, _ = open_clip.create_model_and_transforms("xlm-roberta-large-ViT-H-14", pretrained="frozen_laion5b_s13b_b90k", cache_dir="/mnt/bigdata/networks/TorchHub/")
        emb_dim = 1024
    
    if  "OClipSigLIP" in model_key:
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-L-16-SigLIP-384", pretrained="webli", cache_dir="/mnt/bigdata/networks/TorchHub/")
        emb_dim = 1024
        clip_model.visual.output_dim = 1024
        #print("created SigLIP")

    if  "SigLIP400" in model_key:
        clip_model, _, _ = open_clip.create_model_and_transforms("ViT-SO400M-14-SigLIP-384", pretrained="webli", cache_dir="/mnt/bigdata/networks/TorchHub/", force_patch_dropout=0.33)# force_patch_dropout=0.
        #emb_dim = 1152
        #clip_model.visual.output_dim = emb_dim
        
        clip_model.set_grad_checkpointing(True)
        emb_dim = 1152
        clip_model.visual.output_dim = emb_dim

        if train_with_text:
           
            tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")
            clip_model.text.eval()
            return clip_model, tokenizer
        else:
            model =  clip_model.visual
            emb_dim = 1152

    if "Unicom" in model_key:
        import unicom
        clip_model,_ = unicom.load("ViT-L/14")
        #clip_model.set_grad_checkpointing(True)
        return clip_model
        #emb_dim = 1024
    
    model = clip_model.visual
    model.set_grad_checkpointing(True)

    return EmbeddingModel(model, use_proj=True, output_dim=-1, emb_dim=emb_dim)