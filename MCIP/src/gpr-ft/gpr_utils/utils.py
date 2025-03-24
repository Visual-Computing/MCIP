from pathlib import Path
import torch
import numpy as np
from tqdm import tqdm

class Dotdict(dict):
    """Dot notation access to dictionary attributes, recursively."""
    def __init__(self, d=None):
        super().__init__()
        if d is not None:
            for key, value in d.items():
                # Recursively wrap dictionaries.
                if isinstance(value, dict):
                    value = Dotdict(value)
                self[key] = value

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def list_all_images_at_dir(images_path):
    
    images = []
    for file in Path(images_path).rglob('*.*'):
        f = str(file).lower()
        if (("jpeg" in  f) | ("jpg" in f) | ("png" in f)) & ("._" not in f) & (".txt" not in f):
            
            images.append(str(file))
                 
    return images

def l2_norm(x):
    x = x / torch.norm(x, dim=-1, keepdim=True)
    return x


def get_embeddings(loader, model, device, use_l2=True):
    
    features = None
    data_lenght = len(loader.dataset)
    bs = loader.batch_size
    with torch.set_grad_enabled(False):
        

        pbar = tqdm(enumerate(loader), position=0, leave=True, total=len(loader))
        for i, batch in pbar:
            
            local_batch = batch.to(device)
            fv = model(local_batch)
            #print(local_batch.min(), local_batch.max(), fv.min(), fv.max())
            if use_l2:
                fv = l2_norm(fv)

            fv = fv.cpu().numpy()

            if i == 0:                
                features = np.zeros((data_lenght, fv.shape[-1]), dtype=fv.dtype)

            c_i = i*bs
            features[c_i:c_i+len(fv)] = fv
            pbar.update()
                  
    return np.array(features).astype(np.float32)