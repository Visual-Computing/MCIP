import torch
import numpy as np
import torchvision

from gpr_utils.logger import ExpLogger, EvalLogger
from gpr_utils.utils import get_embeddings

from models.preprocessing import get_ppc_fn

from data.gpr1200 import get_GPR1200_dataset, evaluate_GPR1200
from data.flickr100k import get_Flickr100k_dataset
from data.cars import get_Cars_dataset, evaluate_Cars
from data.cub import get_CUB_dataset, evaluate_CUB
from data.sop import get_SOP_dataset, evaluate_SOP
from data.roxford import get_ROxford_dataset, evaluate_ROxford
from data.rparis import get_RParis_dataset, evaluate_RParis
from data.adience import get_AdienceFaces_dataset, evaluate_AdienceFaces
from data.instre import get_INSTRE_dataset, evaluate_INSTRE
from data.test_dataset import TestDataset
from data.image_net.in1k import get_IN1k_dataset
from data.image_net.in_a import get_INA_dataset
from data.image_net.in_r import get_INR_dataset
from data.image_net.in_sketch import get_INSketch_dataset
from data.image_net.in_v2 import get_INv2_dataset

from evaluation.utils import compute_mean_average_precision, find_kNN
from gpr_utils.plot_utils import find_and_plot_nn, plot_nn

from evaluation.utils import nearest_neighbor_test, reduce_kNNs




def convert_hf_ds(hf_ds, ppc_fn):
    images, labels = list(zip(*hf_ds))
    return TestDataset(images, np.array(labels), ppc_fn=ppc_fn)

def eval_model(model, exp_params, logger, n_updates, eval_ppc_fn=None, eval_ppc_fn2=None):

    train_flowers = torchvision.datasets.Flowers102(root="/mnt/data/images", split="train", download=True)
    test_flowers = torchvision.datasets.Flowers102(root="/mnt/data/images", split="test", download=True)
    train_air = torchvision.datasets.FGVCAircraft(root="/mnt/data/images", split="train", download=True)
    test_air = torchvision.datasets.FGVCAircraft(root="/mnt/data/images", split="test", download=True)

    USE_L2 = True

    if eval_ppc_fn is None:
        eval_ppc_fn = get_ppc_fn(exp_params.model, exp_params.image_size, mode="eval")

    if eval_ppc_fn2 is None:
        eval_ppc_fn2 = get_ppc_fn(exp_params.model, exp_params.image_size, mode="eval", include_open=False)

    eval_logger = EvalLogger(check_point_type=f"after_{n_updates}_updates", exp_logger=logger, use_l2=True, write_logs=True)


    test_loader_params = {  'batch_size': exp_params.test_batch_size,
                            'shuffle': False,
                            'num_workers': 32}
                        
    device = torch.device(f"cuda:{exp_params.gpu_id}")

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

    in_v2_ds = get_INSketch_dataset("/mnt/data/images/ImageNetShifts/imagenet-v2", eval_ppc_fn, IN_class_to_int_dict)
    in_v2_loader = torch.utils.data.DataLoader(in_v2_ds, **test_loader_params)

    in_sketch_ds = get_INv2_dataset("/mnt/data/images/ImageNetShifts/imagenet-sketch", eval_ppc_fn, IN_class_to_int_dict)
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
    
    with torch.cuda.amp.autocast():

            ###INSTRE
            print("---- Computing INSTRE Embeddings:")
            instre_embeddings = get_embeddings(instre_loader, model, device, use_l2=USE_L2)
            instre_mAP = evaluate_INSTRE(cats=instre_ds.labels, features=instre_embeddings)
            print("INSTRE mAP:", instre_mAP)

            eval_logger.update({"INSTRE": instre_mAP})

            
            print("---- Computing GPR Embeddings:")
            gpr1200_embeddings = get_embeddings(gpr_loader, model, device, use_l2=USE_L2)
            #np.save(gpr1200_embeddings_path, gpr1200_embeddings)

            gpr1200, gpr1000, lm, iNat, ims, instre, sop, faces = evaluate_GPR1200(gpr1200_ds.labels, gpr1200_embeddings, compute_partial=True)
            print("GPR1200: {}, GPR1000: {}".format(gpr1200, gpr1000))
            print("Landmarks: {}, IMSketch: {}, iNat: {}, Instre: {}, SOP: {}, faces: {}".format(lm, ims, iNat, instre, sop, faces))

            eval_logger.update({"GPR": 
                                {"gpr1200_mAP": gpr1200,
                                "gpr1000_mAP": gpr1000,
                                "lm_mAP": lm,
                                "iNat_mAP": iNat,
                                "ims_mAP": ims,
                                "sop_mAP": sop,
                                "instre_mAP": instre,
                                "faces_mAP": faces}})

            
            
            # print("---- Computing Flickr100k Embeddings:")
            # flickr_embeddings = get_embeddings(flickr100k_loader, model, device, use_l2=USE_L2)
            # gprflickr_embeddings = np.concatenate([gpr1200_embeddings, flickr_embeddings], axis=0)
            # gprflickr_cats = np.concatenate([gpr1200_ds.labels, flickr100k_ds.labels], axis=0)
            # aps = compute_mean_average_precision(categories_DB=gprflickr_cats,
            #                                     categories_Q=gpr1200_ds.labels,
            #                                     features_DB=gprflickr_embeddings,
            #                                     features_Q=gpr1200_embeddings)
            
            # gpr12000_flickr_mAP = np.mean(aps).round(4)
            # gpr10000_flickr_mAP = np.mean(aps[:10000]).round(4)
            # print("GPR1200+Flickr100k: {}, GPR1000+Flickr100k: {}".format(gpr12000_flickr_mAP, gpr10000_flickr_mAP))


            # eval_logger.update({"GPR+Flickr100k": 
            #                     {"GPR1200_mAP": gpr12000_flickr_mAP,
            #                     "GPR1000_mAP": gpr10000_flickr_mAP}})

            ### ROxford und Paris
            print("---- Computing ROxford Embeddings:")
            roxford_embeddings = get_embeddings(roxford_loader, model, device, use_l2=USE_L2)
            mapM_O, mapH_O = evaluate_ROxford("/mnt/data/images/Roxford_RParis/", roxford_embeddings)


            print("---- Computing RParis Embeddings:")
            rparis_embeddings = get_embeddings(rparis_loader, model, device, use_l2=USE_L2)
            mapM_P, mapH_P = evaluate_RParis("/mnt/data/images/Roxford_RParis/", rparis_embeddings)

            print(f"ROxford: {mapM_O}/{mapH_O} RParis: {mapM_P}/{mapH_P}")

            eval_logger.update({"Roxford_RParis": 
                                {"mapM_O": mapM_O,
                                "mapH_O": mapH_O,
                                "mapM_P": mapM_P,
                                "mapH_P": mapH_P}})
            
            ###DML
            print("---- Computing CUB200_2011 Embeddings:")
            cub_embeddings = get_embeddings(cub_loader, model, device, use_l2=USE_L2)
            cub_recalls = evaluate_CUB(cats=cub_ds.labels, features=cub_embeddings)
            print("CUB200_2011 R@k:", cub_recalls)
            
            print("---- Computing Cars196 Embeddings:")
            cars_embeddings = get_embeddings(cars_loader, model, device, use_l2=USE_L2)
            cars_recalls = evaluate_Cars(cats=cars_ds.labels, features=cars_embeddings)
            print("Cars196 R@k:", cars_recalls)

            print("---- Computing SOP Embeddings:")
            sop_embeddings = get_embeddings(sop_loader, model, device, use_l2=USE_L2)
            sop_recalls = evaluate_SOP(cats=sop_ds.labels, features=sop_embeddings)
            print("SOP R@k:", sop_recalls)

            eval_logger.update({"DML": 
                                {"CUB200": list(cub_recalls),
                                "Cars196": list(cars_recalls),
                                "SOP": list(sop_recalls)}})

            ###Adience Faces
            print("---- Computing Adience Faces Embeddings:")
            adience_embeddings = get_embeddings(adience_loader, model, device, use_l2=USE_L2)
            adience_mAP = evaluate_AdienceFaces(cats=adience_ds.labels, features=adience_embeddings)
            print("Adience Faces mAP:", adience_mAP)

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

            del test_air_loader
            del train_air_loader
            del test_air_ds
            del train_air_ds

            del test_flower_loader
            del train_flower_loader
            del test_flower_ds
            del train_flower_ds

            del train_flowers
            del test_flowers
            del train_air
            del test_air
            #############################################################
            # ImageNet Distribution Shifts

           
            print("---- Computing IN1k Train Embeddings:")
            in1k_train_embeddings = get_embeddings(in1k_train_loader, model, device, use_l2=USE_L2)
            #np.save(in1k_train_embeddings_path, in1k_train_embeddings)
          
            print("---- Computing IN1k val Embeddings:")
            in1k_val_embeddings = get_embeddings(in1k_val_loader, model, device, use_l2=USE_L2)
            #np.save(in1k_val_embeddings_path, in1k_val_embeddings)

           
            print("---- Computing IN1k-A Embeddings:")
            in_a_embeddings = get_embeddings(in_a_loader, model, device, use_l2=USE_L2)
            #np.save(in_a_embeddings_path, in_a_embeddings)

            print("---- Computing IN1k-R Embeddings:")
            in_r_embeddings = get_embeddings(in_r_loader, model, device, use_l2=USE_L2)

            print("---- Computing IN1k-Sketch Embeddings:")
            in_sketch_embeddings = get_embeddings(in_sketch_loader, model, device, use_l2=USE_L2)

            print("---- Computing IN1k-V2 Embeddings:")
            in_v2_embeddings = get_embeddings(in_v2_loader, model, device, use_l2=USE_L2)
            
            # Evaluate using NN-Search
            _, _, in1k_acc1, in1k_acc5 = nearest_neighbor_test(in1k_train_embeddings, in1k_val_embeddings, in1k_train_ds.labels, in1k_val_ds.labels, device)
            

            _, _, in_a_acc1, in_a_acc5 = nearest_neighbor_test(in1k_train_embeddings, in_a_embeddings, in1k_train_ds.labels, in_a_ds.labels, device)
            

            _, _, in_r_acc1, in_r_acc5 = nearest_neighbor_test(in1k_train_embeddings, in_r_embeddings, in1k_train_ds.labels, in_r_ds.labels, device)
            

            _, _, in_v2_acc1, in_v2_acc5 = nearest_neighbor_test(in1k_train_embeddings, in_v2_embeddings, in1k_train_ds.labels, in_v2_ds.labels, device)
            

            _, _, in_sketch_acc1, in_sketch_acc5 = nearest_neighbor_test(in1k_train_embeddings, in_sketch_embeddings, in1k_train_ds.labels, in_sketch_ds.labels, device)
            

            print("---- Finished kNN ImageNet Evaluation:")
            print("TOP1---> IN1k: {}, IN-A: {}, IN-R: {}, IN-V2: {}, IN-Sketch: {}".format(in1k_acc1, in_a_acc1, in_r_acc1, in_v2_acc1, in_sketch_acc1))
            print("TOP5---> IN1k: {}, IN-A: {}, IN-R: {}, IN-V2: {}, IN-Sketch: {}".format(in1k_acc5, in_a_acc5, in_r_acc5, in_v2_acc5, in_sketch_acc5))
            
            eval_logger.update({"IN-kNN": 
                                {"IN1k": [in1k_acc1, in1k_acc5],
                                "IN-A": [in_a_acc1, in_a_acc5],
                                "IN-R": [in_r_acc1, in_r_acc5],
                                "IN-V2": [in_v2_acc1, in_v2_acc5],
                                "IN-Sketch": [in_sketch_acc1, in_sketch_acc5]}})

            
            del in1k_train_embeddings
            del in1k_train_loader
            del in1k_train_ds

            del in1k_val_embeddings
            del in1k_val_loader
            del in1k_val_ds

            del in_sketch_embeddings
            del in_sketch_loader
            del in_sketch_ds

            del in_a_embeddings
            del in_a_loader
            del in_a_ds

            del in_r_embeddings
            del in_r_loader
            del in_r_ds