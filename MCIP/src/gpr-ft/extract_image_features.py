import torch
import numpy as np
from tqdm import tqdm
from gpr_utils.utils import l2_norm


from models.swin import get_swin_model
from models.open_clip import get_oclip_model
from models.vit import get_vit_model
from models.dinov2 import get_dino_model
from models.mobile_vit import get_mobilevit_model
from models.preprocessing import get_ppc_fn

from data.train_dataset import get_train_ds


def get_embeddings(loader, model, device, use_l2=True):
    
    features = None
    data_lenght = len(loader.dataset)
    bs = loader.batch_size
    with torch.set_grad_enabled(False):
        

        pbar = tqdm(enumerate(loader), position=0, leave=True, total=len(loader))
        for i, batch in pbar:
            
            x,y = batch
            local_batch = x.to(device)
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

def extract_image_features(exp_params):

    device = torch.device(f"cuda:{exp_params.gpu_id}")
    print("---- Using GPU:", device)
    
    model = None
    if "Swin" in exp_params.model and "Mixed" not in exp_params.model:
        output_dim = -1
        if "OD" in exp_params: output_dim = exp_params.OD
        model = get_swin_model(exp_params.model, output_dim)

    elif "Clip" in exp_params.model and "O" not in exp_params.model:
        from models.clip import get_clip_model
        output_dim = -1
        if "OD" in exp_params: output_dim = exp_params.OD
        model = get_clip_model(exp_params.model, output_dim)
        
    elif "Vit" == exp_params.model[:3]:
        model = get_vit_model(exp_params.model)
    
    elif "OClip" in exp_params.model:
        model = get_oclip_model(exp_params.model)

    elif "dino" in exp_params.model:
        model = get_dino_model(exp_params.model)
    
    elif "SigLIP" in exp_params.model:
        model = get_oclip_model(exp_params.model)

    elif "MobileViT" in exp_params.model:
        output_dim = -1
        if "OD" in exp_params: output_dim = exp_params.OD
        model = get_mobilevit_model(exp_params.model, output_dim)

    if "SigLIP" or "MovileCLIP" in exp_params.model:
        if type(model) == tuple:
            model = model[0].visual


    model.to(device)
    print("---- Created Model for Key:", exp_params.model)

    inference_ppc_fn = get_ppc_fn(exp_params.model, exp_params.image_size, mode="eval")

    if "OClip" in exp_params.model:
        inference_ppc_fn = get_ppc_fn(exp_params.model, exp_params.image_size, mode="eval", means=(0.48145466, 0.4578275, 0.40821073), stds=(0.26862954, 0.26130258, 0.27577711))
    if "MobileViT" in exp_params.model:
        inference_ppc_fn = get_ppc_fn(exp_params.model, exp_params.image_size, mode="eval", means=(0.0, 0.0, 0.0), stds=(1, 1, 1))
    if "SigLIP" in exp_params.model:
        inference_ppc_fn = get_ppc_fn(exp_params.model, exp_params.image_size, mode="eval", means=(0.5, 0.5, 0.5), stds=(0.5, 0.5, 0.5))

    train_ds = get_train_ds(inference_ppc_fn, exp_params)
    
        
    print(f"---- Created Train Dataset {exp_params.model} with {train_ds.n_classes} classes and {len(train_ds)} images")

    params = {
          'batch_size': exp_params.train_batch_size,
          'shuffle': False,
          'num_workers': 32}

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)   

    train_ds_loader = torch.utils.data.DataLoader(train_ds, collate_fn=collate_fn, **params)
    n_iter_per_epoch = len(train_ds_loader)
    
    model.eval()
    print(f"---- Ready to extract features from {n_iter_per_epoch} batches")
    
    with torch.cuda.amp.autocast(enabled=True):
        train_ds_embeddings = get_embeddings(train_ds_loader, model, device, use_l2=True)

    if exp_params.train_image_embeddings_file_path:
        print(f"---- Saving extracted features to {exp_params.train_image_embeddings_file_path}")
        np.save(exp_params.train_image_embeddings_file_path, train_ds_embeddings)
    else:
        print("---- No file path provided to save extracted features")
        os.makedirs("extracted_features", exist_ok=True)
        np.save(f"extracted_features/{exp_params.model}_train_embeddings.npy", train_ds_embeddings) 