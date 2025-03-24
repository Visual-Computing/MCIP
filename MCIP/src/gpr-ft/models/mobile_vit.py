import torch
from torchvision import transforms as pth_transforms
from torch import nn


class EmbeddingModel(torch.nn.Module):
    
    def __init__(self, backbone, emb_dim, output_dim):
        super().__init__()
        self.backbone = backbone
        self.emb_dim = emb_dim
        self.output_dim = output_dim
        if output_dim > 0:
            self.linear =  nn.Linear(self.emb_dim, output_dim)
            
            self.emb_dim = output_dim
    
    def forward(self, x):
        fv = self.backbone(x)
        if self.output_dim > 0:
            fv = self.linear(fv)

        return fv

def get_mobilevit_model(model_key, output_dim=-1):

    model = torch.jit.load("/mnt/data/networks/pytorch/MobileViTv2_jit.pt")
    emb_dim = 768

    # for param in model.parameters():
    #     param.requires_grad = False
        

    return EmbeddingModel(model, emb_dim, output_dim)
