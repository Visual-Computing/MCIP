import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.scheduler.cosine_lr import CosineLRScheduler
from tqdm import tqdm
from gpr_utils.logger import ExpLogger
from gpr_utils.utils import l2_norm, get_embeddings
from gpr_utils.plot_utils import plot_anchor_tsne

from models.swin import get_swin_model
from models.open_clip import get_oclip_model
from models.vit import get_vit_model
from models.dinov2 import get_dino_model
from models.mobile_vit import get_mobilevit_model


from data.train_dataset import get_train_ds
from loss.loss import get_loss
from models.preprocessing import get_ppc_fn
from data.gpr1200 import get_GPR1200_dataset, evaluate_GPR1200


from eval_model import eval_model

def train(exp_params):

    logger = ExpLogger(param_dict=exp_params)
    print("---- Created Logger with Name:", logger.get_name())

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

    model.to(device)
    print("---- Created Model for Key:", exp_params.model, "Embedding Dim:", model.emb_dim)

    train_ppc_fn = get_ppc_fn(exp_params.model, exp_params.image_size, mode="train")

    if "OClip" in exp_params.model:
        train_ppc_fn = get_ppc_fn(exp_params.model, exp_params.image_size, mode="train", means=(0.48145466, 0.4578275, 0.40821073), stds=(0.26862954, 0.26130258, 0.27577711))
    if "MobileViT" in exp_params.model:
        train_ppc_fn = get_ppc_fn(exp_params.model, exp_params.image_size, mode="train", means=(0.0, 0.0, 0.0), stds=(1, 1, 1))
    if "SigLIP" in exp_params.model:
        train_ppc_fn = get_ppc_fn(exp_params.model, exp_params.image_size, mode="train", means=(0.5, 0.5, 0.5), stds=(0.5, 0.5, 0.5))

    train_ds = get_train_ds(train_ppc_fn, exp_params)
    
        
    print(f"---- Created Train Dataset {exp_params.model} with {train_ds.n_classes} classes and {len(train_ds)} images")

    params = {
          'batch_size': exp_params.train_batch_size,
          'shuffle': True,
          'num_workers': 32}

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return torch.utils.data.dataloader.default_collate(batch)   

    train_loader = torch.utils.data.DataLoader(train_ds, collate_fn=collate_fn, **params)
    n_iter_per_epoch = len(train_loader)
    num_steps =  exp_params.n_epochs * n_iter_per_epoch
    warmup_steps = exp_params.warmup_iters * n_iter_per_epoch 
    print(f"---- Created Train Loader. Steps per Epoch: {n_iter_per_epoch}, Total Steps: {num_steps}, Warmup Steps: {warmup_steps}")

    loss_module = get_loss(exp_params, train_ds.n_classes, model.emb_dim, device)
    
    param_groups = [
        {'params': list(set(model.parameters())), 'lr': exp_params.base_lr},
        #{'params': list(set(model.linear.parameters())), 'lr': exp_params.base_lr*100},
        #{'params': list(set(model.block.parameters())), 'lr': exp_params.base_lr*100},
        {'params': list(set(loss_module.parameters())), 'lr': exp_params.base_lr*100}
    ]

    if exp_params.loss == "MultiLabelArcFace":
        from models.multi_label_wrapper import MultiLabelWrapper 
        model = MultiLabelWrapper(model)
    
    opt = None
    if exp_params.optimizer == "AdamW":
        opt = torch.optim.AdamW(param_groups, eps=1e-8, betas=(0.9, 0.999), weight_decay=exp_params.weight_decay)
    elif exp_params.optimizer == "SGD":
        opt = torch.optim.SGD(param_groups, weight_decay=exp_params.weight_decay, momentum = 0.9, nesterov=True)
    elif exp_params.optimizer == "Lion":
        from lion_pytorch import Lion
        opt = Lion(param_groups, weight_decay=exp_params.weight_decay*5)
    else:
        from dadaptation import DAdaptAdam
        opt = DAdaptAdam(param_groups, eps=1e-8, betas=(0.9, 0.999), weight_decay=exp_params.weight_decay, decouple=True)
    
    print(f"---- Created Optimizer {exp_params.optimizer} and Loss Module {exp_params.loss}")

    scheduler = CosineLRScheduler(
        opt,
        t_initial=num_steps,
        lr_min=exp_params.min_lr,
        warmup_lr_init=exp_params.warmup_lr,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    #TEST SETUP
    test_loader_params = {'batch_size': exp_params.test_batch_size,
          'shuffle': False,
          'num_workers': 16}

    eval_ppc_fn = get_ppc_fn(exp_params.model, exp_params.image_size, mode="eval")
    if "OClip" in exp_params.model:
        eval_ppc_fn = get_ppc_fn(exp_params.model, exp_params.image_size, mode="eval", means=(0.48145466, 0.4578275, 0.40821073), stds=(0.26862954, 0.26130258, 0.27577711))
    if "MobileViT" in exp_params.model:
        eval_ppc_fn = get_ppc_fn(exp_params.model, exp_params.image_size, mode="eval", means=(0.0, 0.0, 0.0), stds=(1, 1, 1))
    if "SigLIP" in exp_params.model:
        eval_ppc_fn = get_ppc_fn(exp_params.model, exp_params.image_size, mode="eval", means=(0.5, 0.5, 0.5), stds=(0.5, 0.5, 0.5))

    gpr1200_ds = get_GPR1200_dataset(exp_params.data.eval_base_path, eval_ppc_fn)
    gpr_loader = torch.utils.data.DataLoader(gpr1200_ds, **test_loader_params)

    
    test_step_i = int(n_iter_per_epoch / exp_params.n_tests_per_epoch)
    stop_training_after =  exp_params.abort_after_no_improvements

    best_gpr1000_map = -np.inf
    best_gpr1200_map = 0
    tests_since_last_increase = 0
    moving_loss = 10
    exit_training = False

    print(f"---- READY TO START TRAINING")
    print(f"---- Testing every {test_step_i} steps and aborting if mAP has not increased in {stop_training_after} test runs")
    for epoch in range(0, exp_params.n_epochs):

        if exit_training: break
    
        pbar = tqdm(enumerate(train_loader), position=0, leave=True)
    
        for batch_idx, (x, y) in pbar:
            batch_idx += 1
            if (batch_idx) % test_step_i == 0:
                model.eval()

                if "eval_mode" not in exp_params or exp_params.eval_mode == "GPR1200":
                    with torch.cuda.amp.autocast():
                        
                        current_step = epoch*n_iter_per_epoch + batch_idx
                        print(f"---- Benchmark at", current_step)

                        gpr1200_embeddings = get_embeddings(gpr_loader, model, device)
                        testfeatures = gpr1200_embeddings[::2000]
                        
                        print(np.round(testfeatures.dot(testfeatures.T), decimals=3))

                        gpr1200, gpr1000, lm, iNat, ims, instre, sop, faces = evaluate_GPR1200(gpr1200_ds.labels, gpr1200_embeddings, compute_partial=True)
                        print("GPR1200: {}, GPR1000: {}".format(gpr1200, gpr1000))
                        print("Landmarks: {}, IMSketch: {}, iNat: {}, Instre: {}, SOP: {}, faces: {}".format(lm, ims, iNat, instre, sop, faces))
                        
                        increased = False
                        if best_gpr1000_map <  gpr1000:
                            increased = True
                            tests_since_last_increase = 0
                            print("----------New Record,  GPR1000 ---------------")
                            best_gpr1000_map = gpr1000
                            torch.save(model.state_dict(), f"{logger.dir_name}/GPR1000.pth")

                        if best_gpr1200_map < gpr1200:
                            increased = True
                            tests_since_last_increase = 0
                            print("----------New Record,  GPR1200 ---------------")
                            best_gpr1200_map = gpr1200
                            torch.save(model.state_dict(), f"{logger.dir_name}/GPR1200.pth")
                                
                        logger.update({
                            "step": current_step,
                            "train_loss": moving_loss,
                            "gpr1200_mAP": gpr1200,
                            "gpr1000_mAP": gpr1000,
                            "lm_mAP": lm,
                            "iNat_mAP": iNat,
                            "ims_mAP": ims,
                            "sop_mAP": sop,
                            "instre_mAP": instre,
                            "faces_mAP": faces,
                            "lr1": opt.state_dict()["param_groups"][0]["lr"], 
                            "lr2": opt.state_dict()["param_groups"][1]["lr"]
                        }, test_step_i)

                        if not increased:
                            torch.save(model.state_dict(), f"{logger.dir_name}/last.pth")
                            tests_since_last_increase += 1
                            
                            if stop_training_after == tests_since_last_increase:
                                print(f"---- Exiting Training")
                                exit_training = True
                                break
                            else:
                                print(f"---- Not increased in {tests_since_last_increase} tests, aborting in {stop_training_after - tests_since_last_increase}")

                if exp_params.eval_mode == "all":
                    n_updates = batch_idx + epoch * n_iter_per_epoch
                    if n_updates >= 116700:
                        eval_model(model, exp_params=exp_params, logger=logger, n_updates=n_updates)
                        
                if (batch_idx) % test_step_i*4 == 0 and (exp_params.data == "GPR"):  
                    fig = plot_anchor_tsne(l2_norm(loss_module.proxies).detach().cpu().numpy(), gpr1200_embeddings, skip=20, title=f"Iter: {epoch * n_iter_per_epoch + batch_idx}, GPR1200: {gpr1200}, GPR1000: {gpr1000}")
                    fig.savefig(f"{logger.plot_dir_name}/{current_step}.jpg")

                model.train()
            
            with torch.cuda.amp.autocast():

                features = model(x.to(device))
                
                labels = y.to(device)
                losses = loss_module(features, labels)

                loss = losses.mean()
                loss.retain_grad()
                moving_loss = 0.99*moving_loss + 0.01*loss.item()
                
    
                scaler.scale(loss).backward()
        
                scheduler.step_update(epoch * n_iter_per_epoch + batch_idx)
                
                if exp_params.grad_norm > 0:
                    if exp_params.loss == "MultiLabelArcFace":
                        torch.nn.utils.clip_grad_norm_(model.model.backbone.parameters(), 1, norm_type=2.0, error_if_nonfinite=False)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.backbone.parameters(), 1, norm_type=2.0, error_if_nonfinite=False)
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1, norm_type=2.0, error_if_nonfinite=False)
                    
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            

            pbar.set_description(
                'Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.4f} LR: {:.12f}  LR2: {:.12f}'.format(
                    epoch, batch_idx + 1, len(train_loader),
                    100. * batch_idx / len(train_loader),
                    moving_loss, opt.state_dict()["param_groups"][0]["lr"], opt.state_dict()["param_groups"][1]["lr"]), False)
            
            pbar.update()