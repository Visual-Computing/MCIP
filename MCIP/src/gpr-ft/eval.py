import torch
import numpy as np
import os

from gpr_utils.logger import ExpLogger, EvalLogger
from gpr_utils.utils import get_embeddings

# from models.swin import get_swin_model
# from models.clip import get_clip_model
# from models.vit import get_vit_model
# from models.bamboo import get_bamboo_model
# from models.ms_vit import get_ms_vit_model
from models.preprocessing import get_ppc_fn
# from models.open_clip import get_oclip_model
# from models.dinov2 import get_dino_model
# from models.mobile_vit import get_mobilevit_model
# from models.dreamsim import get_dreamsim_model

from data.gpr1200 import get_GPR1200_dataset, evaluate_GPR1200
from data.flickr100k import get_Flickr100k_dataset
from data.cars import get_Cars_dataset, evaluate_Cars
from data.cub import get_CUB_dataset, evaluate_CUB
from data.sop import get_SOP_dataset, evaluate_SOP
from data.roxford import get_ROxford_dataset, evaluate_ROxford
from data.rparis import get_RParis_dataset, evaluate_RParis
from data.adience import get_AdienceFaces_dataset, evaluate_AdienceFaces
from data.instre import get_INSTRE_dataset, evaluate_INSTRE

from data.image_net.in1k import get_IN1k_dataset
from data.image_net.in_a import get_INA_dataset
from data.image_net.in_r import get_INR_dataset
from data.image_net.in_sketch import get_INSketch_dataset
from data.image_net.in_v2 import get_INv2_dataset

from evaluation.utils import compute_mean_average_precision, find_kNN
from gpr_utils.plot_utils import find_and_plot_nn, plot_nn

from sklearn.linear_model import LogisticRegression
import joblib
from evaluation.utils import compute_accuracy, nearest_neighbor_test, reduce_kNNs
from data.test_dataset import TestDataset
import torchvision

USE_L2 = True
WRITE_LOGS = True

def convert_hf_ds(hf_ds, ppc_fn):
    images, labels = list(zip(*hf_ds))
    return TestDataset(images, np.array(labels), ppc_fn=ppc_fn)

def evaluate(exp_params):
    
    logger = ExpLogger(param_dict=exp_params, save_path="/mnt/data/networks/GPR_FT/")
    print("---- Created Logger with Name:", logger.get_name())

    device = torch.device(f"cuda:{exp_params.gpu_id}")
    print("---- Using GPU:", device)

    
   
    if "SigLIP" in exp_params.model:
        eval_ppc_fn = get_ppc_fn(exp_params.model, exp_params.image_size, mode="eval", means=(0.5, 0.5, 0.5), stds=(0.5, 0.5, 0.5))
        eval_ppc_fn2 = get_ppc_fn(exp_params.model, exp_params.image_size, mode="eval", means=(0.5, 0.5, 0.5), stds=(0.5, 0.5, 0.5), include_open=False)
    elif "OClip" in exp_params.model:
        eval_ppc_fn = get_ppc_fn(exp_params.model, exp_params.image_size, mode="eval", means=(0.48145466, 0.4578275, 0.40821073), stds=(0.26862954, 0.26130258, 0.27577711))
        eval_ppc_fn2 = get_ppc_fn(exp_params.model, exp_params.image_size, mode="eval", means=(0.48145466, 0.4578275, 0.40821073), stds=(0.26862954, 0.26130258, 0.27577711), include_open=False)
    else:
        eval_ppc_fn = get_ppc_fn(exp_params.model, exp_params.image_size, mode="eval")
        eval_ppc_fn2 = get_ppc_fn(exp_params.model, exp_params.image_size, mode="eval", include_open=False)

    if "MobileCLIP" in exp_params.model:
        eval_ppc_fn = get_ppc_fn(exp_params.model, exp_params.image_size, mode="eval", means=(0.0, 0.0, 0.0), stds=(1, 1, 1))
        eval_ppc_fn2 = get_ppc_fn(exp_params.model, exp_params.image_size, mode="eval", means=(0.0, 0.0, 0.0), stds=(1, 1, 1), include_open=False)

    test_loader_params = {  'batch_size': exp_params.test_batch_size,
                            'shuffle': False,
                            'num_workers': 32}

    print(eval_ppc_fn)
    train_flowers = torchvision.datasets.Flowers102(root="/mnt/data/images", split="train", download=True)
    test_flowers = torchvision.datasets.Flowers102(root="/mnt/data/images", split="test", download=True)
    train_air = torchvision.datasets.FGVCAircraft(root="/mnt/data/images", split="train", download=True)
    test_air = torchvision.datasets.FGVCAircraft(root="/mnt/data/images", split="test", download=True)        
    
    # load all the eval datasets
    gpr1200_ds = get_GPR1200_dataset("/mnt/data/images/GPR10x1200/images/", eval_ppc_fn)
    gpr_loader = torch.utils.data.DataLoader(gpr1200_ds, **test_loader_params)

    flickr100k_ds = get_Flickr100k_dataset("/mnt/data/images/Flickr100k/", eval_ppc_fn)
    flickr100k_loader = torch.utils.data.DataLoader(flickr100k_ds, **test_loader_params)

    cars_ds = get_Cars_dataset("/mnt/data/images/cars196/", eval_ppc_fn)
    cars_loader = torch.utils.data.DataLoader(cars_ds, **test_loader_params)

    sop_ds = get_SOP_dataset("/mnt/data/images/Stanford_Online_Products/", eval_ppc_fn)
    sop_loader = torch.utils.data.DataLoader(sop_ds, **test_loader_params)

    cub_ds = get_CUB_dataset("/mnt/data/images/CUB_200_2011/images/", eval_ppc_fn)
    cub_loader = torch.utils.data.DataLoader(cub_ds, **test_loader_params)

    roxford_ds = get_ROxford_dataset("/mnt/data/images/Roxford_RParis/", eval_ppc_fn)
    roxford_loader = torch.utils.data.DataLoader(roxford_ds, **test_loader_params)

    rparis_ds = get_RParis_dataset("/mnt/data/images/Roxford_RParis/", eval_ppc_fn)
    rparis_loader = torch.utils.data.DataLoader(rparis_ds, **test_loader_params)

    adience_ds = get_AdienceFaces_dataset("/mnt/data/images/Adience_Faces/", eval_ppc_fn)
    adience_loader = torch.utils.data.DataLoader(adience_ds, **test_loader_params)

    instre_ds = get_INSTRE_dataset("/mnt/data/images/instre/", eval_ppc_fn)
    instre_loader = torch.utils.data.DataLoader(instre_ds, **test_loader_params)

    ### load all ImageNet Variants
    in1k_val_ds, IN_class_to_int_dict = get_IN1k_dataset("/mnt/data/images/ImageNet1k_2012", eval_ppc_fn, mode="val", include_class_dict=True)
    in1k_val_loader = torch.utils.data.DataLoader(in1k_val_ds, **test_loader_params)

    in_a_ds = get_INA_dataset("/mnt/data/images/ImageNetShifts/imagenet-a", eval_ppc_fn, IN_class_to_int_dict)
    in_a_loader = torch.utils.data.DataLoader(in_a_ds, **test_loader_params)

    in_r_ds = get_INR_dataset("/mnt/data/images/ImageNetShifts/imagenet-r", eval_ppc_fn, IN_class_to_int_dict)
    in_r_loader = torch.utils.data.DataLoader(in_r_ds, **test_loader_params)

    in_v2_ds = get_INv2_dataset("/mnt/data/images/ImageNetShifts/imagenet-v2", eval_ppc_fn, IN_class_to_int_dict)
    in_v2_loader = torch.utils.data.DataLoader(in_v2_ds, **test_loader_params)

    in_sketch_ds = get_INSketch_dataset("/mnt/data/images/ImageNetShifts/imagenet-sketch", eval_ppc_fn, IN_class_to_int_dict)
    in_sketch_loader = torch.utils.data.DataLoader(in_sketch_ds, **test_loader_params)

    in1k_train_ds = get_IN1k_dataset("/mnt/data/images/ImageNet1k_2012", eval_ppc_fn, IN_class_to_int_dict, mode="train")
    in1k_train_loader = torch.utils.data.DataLoader(in1k_train_ds, **test_loader_params)

    test_flower_ds = convert_hf_ds(test_flowers, eval_ppc_fn2)
    train_flower_ds = convert_hf_ds(train_flowers, eval_ppc_fn2)

    test_air_ds = convert_hf_ds(test_air, eval_ppc_fn2)
    train_air_ds = convert_hf_ds(train_air, eval_ppc_fn2)

    test_flower_loader = torch.utils.data.DataLoader(test_flower_ds, **test_loader_params)
    train_flower_loader = torch.utils.data.DataLoader(train_flower_ds, **test_loader_params)
    
    test_air_loader = torch.utils.data.DataLoader(test_air_ds, **test_loader_params)
    train_air_loader = torch.utils.data.DataLoader(train_air_ds, **test_loader_params)
    
    #start evaluation
    for checkpoint_type in ["baseline", "GPR1200", "GPR1000"]:#["baseline", "GPR1000", "INSTRE"]: 
        print()

        model = None
        if "Swin" in exp_params.model and "Mixed" not in exp_params.model:
            output_dim = -1
            if "OD" in exp_params: output_dim = exp_params.OD
            model = get_swin_model(exp_params.model, output_dim)

        elif "Clip" in exp_params.model and "O" not in exp_params.model:
            output_dim = -1
            if "OD" in exp_params: output_dim = exp_params.OD
            model = get_clip_model(exp_params.model, output_dim)

        elif "MixedSwin" in exp_params.model:
            # output_dim = -1
            # if "OD" in exp_params: output_dim = exp_params.OD
            model = get_mixed_swin_model(exp_params.model)
            
        elif "Vit" == exp_params.model[:3]:
            model = get_vit_model(exp_params.model)

        elif "Bamboo" in exp_params.model:
            model = get_bamboo_model()

        elif "MSVit" in exp_params.model:
            model = get_ms_vit_model(exp_params.model)

        elif "MEVit" in exp_params.model:
            model = get_me_vit_model(exp_params.model)

        elif "OClip" in exp_params.model:
            model = get_oclip_model(exp_params.model)

        elif "dino" in exp_params.model:
            model = get_dino_model(exp_params.model)

        elif "MobileViT" in exp_params.model:
            output_dim = -1
            if "OD" in exp_params: output_dim = exp_params.OD
            model = get_mobilevit_model(exp_params.model, output_dim)

        elif "SigLIP" in exp_params.model:
            model, _ = get_oclip_model(exp_params.model)

        elif "MobileCLIP" in exp_params.model:
            from models.mobile_clip import get_mclip_model
            model, _ = get_mclip_model(exp_params.model)
            
           


        model.to(device)
        model.eval()

        print("---- Created Model for Key:", exp_params.model, "Embedding Dim:")
        if checkpoint_type != "baseline":
            check_point_path = f"{logger.dir_name}/{checkpoint_type}.pth"
            model.load_state_dict(torch.load(check_point_path))
            print("---- Succesfully loaded checkoint:", check_point_path)
        if "SigLIP" in exp_params.model:
            model = model.visual

        if "MobileCLIP" in exp_params.model:
            model = model.visual
            model.eval()

        eval_logger = EvalLogger(check_point_type=checkpoint_type, exp_logger=logger, use_l2=USE_L2, write_logs=WRITE_LOGS)
        
        with torch.cuda.amp.autocast():

            ###INSTRE
            print("---- Computing INSTRE Embeddings:", checkpoint_type)
            instre_embeddings = get_embeddings(instre_loader, model, device, use_l2=USE_L2)
            instre_mAP = evaluate_INSTRE(cats=instre_ds.labels, features=instre_embeddings)
            print("INSTRE mAP:", instre_mAP, instre_embeddings.shape)
            find_and_plot_nn(instre_ds.file_paths, instre_embeddings, instre_ds.labels, device=device,
                            save_path=f"{eval_logger.plot_dir_name}/INSTRE_NN_{checkpoint_type}_{eval_logger.get_name()}.jpg",
                            title="INSTRE mAP: {}".format(instre_mAP))

            eval_logger.update({"INSTRE": instre_mAP})

            ### GPR und GPR+Flickr100k
            gpr1200_embeddings_path = f"{eval_logger.dir_name}/{checkpoint_type}_gpr1200_embeddings.npy"
            if os.path.isfile(gpr1200_embeddings_path):
                gpr1200_embeddings = np.load(gpr1200_embeddings_path)
            else:
                print("---- Computing GPR Embeddings:", checkpoint_type)
                gpr1200_embeddings = get_embeddings(gpr_loader, model, device, use_l2=USE_L2)
                #np.save(gpr1200_embeddings_path, gpr1200_embeddings)

            gpr1200, gpr1000, lm, iNat, ims, instre, sop, faces = evaluate_GPR1200(gpr1200_ds.labels, gpr1200_embeddings, compute_partial=True)
            print("GPR1200: {}, GPR1000: {}".format(gpr1200, gpr1000))
            print("Landmarks: {}, IMSketch: {}, iNat: {}, Instre: {}, SOP: {}, faces: {}".format(lm, ims, iNat, instre, sop, faces))

            find_and_plot_nn(gpr1200_ds.file_paths, gpr1200_embeddings, gpr1200_ds.labels, device=device,
                            save_path=f"{eval_logger.plot_dir_name}/GPR1200_NN_{checkpoint_type}_{eval_logger.get_name()}.jpg",
                            title="GPR1200: {}, GPR1000: {} Landmarks: {}, IMSketch: {}, iNat: {}, Instre: {}, SOP: {}, faces: {}".format(gpr1200, gpr1000, lm, ims, iNat, instre, sop, faces))

            eval_logger.update({"GPR": 
                                {"gpr1200_mAP": gpr1200,
                                "gpr1000_mAP": gpr1000,
                                "lm_mAP": lm,
                                "iNat_mAP": iNat,
                                "ims_mAP": ims,
                                "sop_mAP": sop,
                                "instre_mAP": instre,
                                "faces_mAP": faces}})

            
            
            print("---- Computing Flickr100k Embeddings:", checkpoint_type)
            flickr_embeddings = get_embeddings(flickr100k_loader, model, device, use_l2=USE_L2)
            gprflickr_embeddings = np.concatenate([gpr1200_embeddings, flickr_embeddings], axis=0)
            gprflickr_cats = np.concatenate([gpr1200_ds.labels, flickr100k_ds.labels], axis=0)
            gprflickr_images = np.concatenate([gpr1200_ds.file_paths, flickr100k_ds.file_paths], axis=0)
            aps = compute_mean_average_precision(categories_DB=gprflickr_cats,
                                                categories_Q=gpr1200_ds.labels,
                                                features_DB=gprflickr_embeddings,
                                                features_Q=gpr1200_embeddings)
            
            gpr12000_flickr_mAP = np.mean(aps).round(4)
            gpr10000_flickr_mAP = np.mean(aps[:10000]).round(4)
            print("GPR1200+Flickr100k: {}, GPR1000+Flickr100k: {}".format(gpr12000_flickr_mAP, gpr10000_flickr_mAP))

            skip_for_gprflickr = int(len(gpr1200_ds.labels) / 30)
            nn_images, nn_cats = find_kNN(gpr1200_embeddings[::skip_for_gprflickr], gprflickr_embeddings, device=device,
                                            k=20, val_list=[gprflickr_images, gprflickr_cats], skip_self=1)

            plot_nn(gpr1200_ds.file_paths[::skip_for_gprflickr], nn_images, gpr1200_ds.labels[::skip_for_gprflickr], 
                    nn_cats, save_path=f"{eval_logger.plot_dir_name}/GPR1200+Flickr100k_NN_{checkpoint_type}_{eval_logger.get_name()}.jpg",
                    title="GPR1200+Flickr100k: {}, GPR1000+Flickr100k: {}".format(gpr12000_flickr_mAP, gpr10000_flickr_mAP))

            eval_logger.update({"GPR+Flickr100k": 
                                {"GPR1200_mAP": gpr12000_flickr_mAP,
                                "GPR1000_mAP": gpr10000_flickr_mAP}})

            ### ROxford und Paris
            print("---- Computing ROxford Embeddings:", checkpoint_type)
            roxford_embeddings = get_embeddings(roxford_loader, model, device, use_l2=USE_L2)
            mapM_O, mapH_O = evaluate_ROxford("/mnt/data/images/Roxford_RParis/", roxford_embeddings)

            find_and_plot_nn(roxford_ds.file_paths, roxford_embeddings, roxford_ds.labels, device=device,
                            save_path=f"{eval_logger.plot_dir_name}/ROxford_NN_{checkpoint_type}_{eval_logger.get_name()}.jpg",
                            title=f"ROxford: {mapM_O}/{mapH_O}")

            print("---- Computing RParis Embeddings:", checkpoint_type)
            rparis_embeddings = get_embeddings(rparis_loader, model, device, use_l2=USE_L2)
            mapM_P, mapH_P = evaluate_RParis("/mnt/data/images/Roxford_RParis/", rparis_embeddings)

            find_and_plot_nn(rparis_ds.file_paths, rparis_embeddings, rparis_ds.labels, device=device,
                            save_path=f"{eval_logger.plot_dir_name}/RParis_NN_{checkpoint_type}_{eval_logger.get_name()}.jpg",
                            title=f"RParis: {mapM_P}/{mapH_P}")

            print(f"ROxford: {mapM_O}/{mapH_O} RParis: {mapM_P}/{mapH_P}")

            eval_logger.update({"Roxford_RParis": 
                                {"mapM_O": mapM_O,
                                "mapH_O": mapH_O,
                                "mapM_P": mapM_P,
                                "mapH_P": mapH_P}})
            
            ###DML
            print("---- Computing CUB200_2011 Embeddings:", checkpoint_type)
            cub_embeddings = get_embeddings(cub_loader, model, device, use_l2=USE_L2)
            cub_recalls = evaluate_CUB(cats=cub_ds.labels, features=cub_embeddings)
            print("CUB200_2011 R@k:", cub_recalls)
            find_and_plot_nn(cub_ds.file_paths, cub_embeddings, cub_ds.labels, device=device,
                            save_path=f"{eval_logger.plot_dir_name}/CUB_NN_{checkpoint_type}_{eval_logger.get_name()}.jpg",
                            title=f"CUB200_2011 R@k: {cub_recalls}")

            print("---- Computing Cars196 Embeddings:", checkpoint_type)
            cars_embeddings = get_embeddings(cars_loader, model, device, use_l2=USE_L2)
            cars_recalls = evaluate_Cars(cats=cars_ds.labels, features=cars_embeddings)
            print("Cars196 R@k:", cars_recalls)
            find_and_plot_nn(cars_ds.file_paths, cars_embeddings, cars_ds.labels, device=device,
                            save_path=f"{eval_logger.plot_dir_name}/Cars_NN_{checkpoint_type}_{eval_logger.get_name()}.jpg",
                            title=f"Cars196 R@k: {cars_recalls}")

            print("---- Computing SOP Embeddings:", checkpoint_type)
            sop_embeddings = get_embeddings(sop_loader, model, device, use_l2=USE_L2)
            sop_recalls = evaluate_SOP(cats=sop_ds.labels, features=sop_embeddings)
            print("SOP R@k:", sop_recalls)
            find_and_plot_nn(sop_ds.file_paths, sop_embeddings, sop_ds.labels, device=device,
                            save_path=f"{eval_logger.plot_dir_name}/SOP_NN_{checkpoint_type}_{eval_logger.get_name()}.jpg",
                            title=f"SOP R@k: {sop_recalls}")

            eval_logger.update({"DML": 
                                {"CUB200": list(cub_recalls),
                                "Cars196": list(cars_recalls),
                                "SOP": list(sop_recalls)}})

            ###Adience Faces
            print("---- Computing Adience Faces Embeddings:", checkpoint_type)
            adience_embeddings = get_embeddings(adience_loader, model, device, use_l2=USE_L2)
            adience_mAP = evaluate_AdienceFaces(cats=adience_ds.labels, features=adience_embeddings)
            print("Adience Faces mAP:", adience_mAP)
            find_and_plot_nn(adience_ds.file_paths, adience_embeddings, adience_ds.labels, device=device,
                            save_path=f"{eval_logger.plot_dir_name}/AdienceFaces_NN_{checkpoint_type}_{eval_logger.get_name()}.jpg",
                            title=f"Adience Faces mAP: {adience_mAP}")

            eval_logger.update({"Adience": adience_mAP})

            #### Flowers and Air
            test_flower_embeddings = get_embeddings(test_flower_loader, model, device, use_l2=USE_L2)
            train_flower_embeddings = get_embeddings(train_flower_loader, model, device, use_l2=USE_L2)
            _, _, flowers_acc1, flowers_acc5 = nearest_neighbor_test(train_flower_embeddings, test_flower_embeddings, train_flower_ds.labels, test_flower_ds.labels, device)

            print("flowers", flowers_acc1, flowers_acc5)
        
            test_air_embeddings = get_embeddings(test_air_loader, model, device, use_l2=USE_L2)
            train_air_embeddings = get_embeddings(train_air_loader, model, device, use_l2=USE_L2)
        
            _, _, airs_acc1, airs_acc5 = nearest_neighbor_test(train_air_embeddings, test_air_embeddings, train_air_ds.labels, test_air_ds.labels, device)
            print("aircraft", airs_acc1, airs_acc5)

            eval_logger.update({"F+A":
                                {"Flowers": [flowers_acc1, flowers_acc5],
                                "Aircraft": [airs_acc1, airs_acc5]}})

            # del test_air_loader
            # del train_air_loader
            # del test_air_ds
            # del train_air_ds

            # del test_flower_loader
            # del train_flower_loader
            # del test_flower_ds
            # del train_flower_ds

            # del train_flowers
            # del test_flowers
            # del train_air
            # del test_air


            #############################################################
            # ImageNet Distribution Shifts

            in1k_train_embeddings_path = f"{eval_logger.dir_name}/{checkpoint_type}_in1k_train_embeddings.npy"
            if os.path.isfile(in1k_train_embeddings_path):
                in1k_train_embeddings = np.load(in1k_train_embeddings_path)
            else:
                print("---- Computing IN1k Train Embeddings:", checkpoint_type)
                in1k_train_embeddings = get_embeddings(in1k_train_loader, model, device, use_l2=USE_L2)
                #np.save(in1k_train_embeddings_path, in1k_train_embeddings)


            in1k_val_embeddings_path = f"{eval_logger.dir_name}/{checkpoint_type}_in1k_val_embeddings.npy"
            if os.path.isfile(in1k_val_embeddings_path):
                in1k_val_embeddings = np.load(in1k_val_embeddings_path)
            else:
                print("---- Computing IN1k val Embeddings:", checkpoint_type)
                in1k_val_embeddings = get_embeddings(in1k_val_loader, model, device, use_l2=USE_L2)
                #np.save(in1k_val_embeddings_path, in1k_val_embeddings)

            in_a_embeddings_path = f"{eval_logger.dir_name}/{checkpoint_type}_in_a_embeddings.npy"
            if os.path.isfile(in_a_embeddings_path):
                in_a_embeddings = np.load(in_a_embeddings_path)
            else:
                print("---- Computing IN1k-A Embeddings:", checkpoint_type)
                in_a_embeddings = get_embeddings(in_a_loader, model, device, use_l2=USE_L2)
                #np.save(in_a_embeddings_path, in_a_embeddings)

            print("---- Computing IN1k-R Embeddings:", checkpoint_type)
            in_r_embeddings = get_embeddings(in_r_loader, model, device, use_l2=USE_L2)

            print("---- Computing IN1k-Sketch Embeddings:", checkpoint_type)
            in_sketch_embeddings = get_embeddings(in_sketch_loader, model, device, use_l2=USE_L2)

            print("---- Computing IN1k-V2 Embeddings:", checkpoint_type)
            in_v2_embeddings = get_embeddings(in_v2_loader, model, device, use_l2=USE_L2)
            
            # Evaluate using NN-Search
            in1k_indices, preds, in1k_acc1, in1k_acc5 = nearest_neighbor_test(in1k_train_embeddings, in1k_val_embeddings, in1k_train_ds.labels, in1k_val_ds.labels, device)
            skip, (nn_images, nn_cats) = reduce_kNNs([in1k_train_ds.file_paths, in1k_train_ds.labels], in1k_indices)
            plot_nn(in1k_val_ds.file_paths[::skip], nn_images, in1k_val_ds.labels[::skip], 
                    nn_cats, save_path=f"{eval_logger.plot_dir_name}/IN1k_NNs_{checkpoint_type}_{eval_logger.get_name()}.jpg", q_predictions=preds[::skip],
                    title="IN1k Top1 Acc: {}, IN1k Top5 Acc: {}".format(in1k_acc1, in1k_acc5))

            in_a_indices, preds, in_a_acc1, in_a_acc5 = nearest_neighbor_test(in1k_train_embeddings, in_a_embeddings, in1k_train_ds.labels, in_a_ds.labels, device)
            skip, (nn_images, nn_cats) = reduce_kNNs([in1k_train_ds.file_paths, in1k_train_ds.labels], in_a_indices)
            plot_nn(in_a_ds.file_paths[::skip], nn_images, in_a_ds.labels[::skip], 
                    nn_cats, save_path=f"{eval_logger.plot_dir_name}/in_a_NNs_{checkpoint_type}_{eval_logger.get_name()}.jpg", q_predictions=preds[::skip],
                    title="in_a Top1 Acc: {}, in_a Top5 Acc: {}".format(in_a_acc1, in_a_acc5))

            in_r_indices, preds, in_r_acc1, in_r_acc5 = nearest_neighbor_test(in1k_train_embeddings, in_r_embeddings, in1k_train_ds.labels, in_r_ds.labels, device)
            skip, (nn_images, nn_cats) = reduce_kNNs([in1k_train_ds.file_paths, in1k_train_ds.labels], in_r_indices)
            plot_nn(in_r_ds.file_paths[::skip], nn_images, in_r_ds.labels[::skip], 
                    nn_cats, save_path=f"{eval_logger.plot_dir_name}/in_r_NNs_{checkpoint_type}_{eval_logger.get_name()}.jpg", q_predictions=preds[::skip],
                    title="in_r Top1 Acc: {}, in_r Top5 Acc: {}".format(in_r_acc1, in_r_acc5))

            in_v2_indices, preds, in_v2_acc1, in_v2_acc5 = nearest_neighbor_test(in1k_train_embeddings, in_v2_embeddings, in1k_train_ds.labels, in_v2_ds.labels, device)
            skip, (nn_images, nn_cats) = reduce_kNNs([in1k_train_ds.file_paths, in1k_train_ds.labels], in_v2_indices)
            plot_nn(in_v2_ds.file_paths[::skip], nn_images, in_v2_ds.labels[::skip], 
                    nn_cats, save_path=f"{eval_logger.plot_dir_name}/in_v2_NNs_{checkpoint_type}_{eval_logger.get_name()}.jpg", q_predictions=preds[::skip],
                    title="in_v2 Top1 Acc: {}, in_v2 Top5 Acc: {}".format(in_v2_acc1, in_v2_acc5))

            in_sketch_indices, preds, in_sketch_acc1, in_sketch_acc5 = nearest_neighbor_test(in1k_train_embeddings, in_sketch_embeddings, in1k_train_ds.labels, in_sketch_ds.labels, device)
            skip, (nn_images, nn_cats) = reduce_kNNs([in1k_train_ds.file_paths, in1k_train_ds.labels], in_sketch_indices)
            plot_nn(in_sketch_ds.file_paths[::skip], nn_images, in_sketch_ds.labels[::skip], 
                    nn_cats, save_path=f"{eval_logger.plot_dir_name}/in_sketch_NNs_{checkpoint_type}_{eval_logger.get_name()}.jpg", q_predictions=preds[::skip],
                    title="in_sketch Top1 Acc: {}, in_sketch Top5 Acc: {}".format(in_sketch_acc1, in_sketch_acc5))

            print("---- Finished kNN ImageNet Evaluation:", checkpoint_type)
            print("TOP1---> IN1k: {}, IN-A: {}, IN-R: {}, IN-V2: {}, IN-Sketch: {}".format(in1k_acc1, in_a_acc1, in_r_acc1, in_v2_acc1, in_sketch_acc1))
            print("TOP5---> IN1k: {}, IN-A: {}, IN-R: {}, IN-V2: {}, IN-Sketch: {}".format(in1k_acc5, in_a_acc5, in_r_acc5, in_v2_acc5, in_sketch_acc5))
            
            eval_logger.update({"IN-kNN": 
                                {"IN1k": [in1k_acc1, in1k_acc5],
                                "IN-A": [in_a_acc1, in_a_acc5],
                                "IN-R": [in_r_acc1, in_r_acc5],
                                "IN-V2": [in_v2_acc1, in_v2_acc5],
                                "IN-Sketch": [in_sketch_acc1, in_sketch_acc5]}})
        

            if False: #do not perform linear probe evaluation
                classifier_path = f"{eval_logger.dir_name}/{checkpoint_type}_IN1k_linear_classifier.pkl"
                if os.path.isfile(classifier_path):
                        classifier = joblib.load(classifier_path)
                else:
                        print("---- Training IN1k Classifier:", checkpoint_type)
                        classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1, n_jobs=16)
                        classifier.fit(in1k_train_embeddings, in1k_train_ds.labels)
                        joblib.dump(classifier, classifier_path)

                # Evaluate using the logistic regression classifier
                in1k_preds = classifier.predict(in1k_val_embeddings)
                in1k_accuracy = compute_accuracy(in1k_val_ds.labels, in1k_preds)

                in_a_preds = classifier.predict(in_a_embeddings)
                in_a_accuracy = compute_accuracy(in_a_ds.labels, in_a_preds)

                in_r_preds = classifier.predict(in_r_embeddings)
                in_r_accuracy = compute_accuracy(in_r_ds.labels, in_r_preds)

                in_v2_preds = classifier.predict(in_v2_embeddings)
                in_v2_accuracy = compute_accuracy(in_v2_ds.labels, in_v2_preds)

                in_sketch_preds = classifier.predict(in_sketch_embeddings)
                in_sketch_accuracy = compute_accuracy(in_sketch_ds.labels, in_sketch_preds)

                print("---- Finished Linear ImageNet Evaluation:", checkpoint_type)
                print("IN1k: {}, IN-A: {}, IN-R: {}, IN-V2: {}, IN-Sketch: {}".format(in1k_accuracy, in_a_accuracy, in_r_accuracy, in_v2_accuracy, in_sketch_accuracy))
                
                eval_logger.update({"IN-Linear": 
                                        {"IN1k": in1k_accuracy,
                                        "IN-A": in_a_accuracy,
                                        "IN-R": in_r_accuracy,
                                        "IN-V2": in_v2_accuracy,
                                        "IN-Sketch": in_sketch_accuracy}})