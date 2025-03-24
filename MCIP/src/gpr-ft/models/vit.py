import torch
from torchvision import transforms as pth_transforms
import timm


class EmbeddingModel(torch.nn.Module):
    
    def __init__(self, backbone, emb_dim):
        super().__init__()
        self.backbone = backbone
        self.emb_dim = emb_dim
    
    def forward(self, x):
        fv = self.backbone.forward_features(x)[:, 0,:]
        return fv

def get_vit_model(model_key):

    model = None
    emb_dim = None
    if "VitB" in model_key:
        model = timm.create_model("vit_base_patch16_224_in21k", pretrained=True)
        emb_dim = 768
        
    if "VitL384" in model_key:
        model = timm.create_model("vit_large_patch16_384", pretrained=True)
        emb_dim = 1024

    return EmbeddingModel(model, emb_dim)
