import numpy as np
import os
from data.test_dataset import TestDataset
from evaluation.utils import compute_mean_average_precision
from data.base import Base

class GPR1200(Base):

    """GPR1200 class
    
    The dataset contains 12k images from 1200 diverse categories. 
    """
    
    
    def __init__(self, base_dir):
        """
        Load the image information from the drive
        
        Parameters
        ----------
        base_dir : string 
            GPR1200 base directory path
        """
        self._base_dir = base_dir
        
        gpr10x1200_cats, gpr10x1200_files = [], []

        data = sorted(os.listdir(base_dir), key=lambda a: int(os.path.basename(a).split("_")[0]))
        for file in data:
            file_path = os.path.join(base_dir, file)
            cat = os.path.basename(file).split("_")[0]
            gpr10x1200_cats.append(cat)
            gpr10x1200_files.append(file_path)

        Base.__init__(self, base_dir, "GPR1200", gpr10x1200_cats, gpr10x1200_files)

def evaluate_GPR1200(cats, features, compute_partial=False, float_n=4):
    """
    Compute the mean average precision of each part of this combined data set. 
    Providing just the 'features' will assume the manhatten distance between all images will be computed 
    before calculating the mean average precision. This metric can 
    be changed with any scikit learn 'distance_metric'. 
    
        
    Parameters
    ----------
    features : ndarray 
        matrix representing the embeddings of all the images in the dataset
    indices: array-lile, shape = [n_samples_Q, n_samples_DB]
        Nearest neighbours indices 
    """

    aps = compute_mean_average_precision(cats, features_DB=features)
    all_map = np.round(np.mean(aps), decimals=float_n)

    if compute_partial: 

        cl_map = np.round(np.mean(aps[:2000]), decimals=float_n)
        iNat_map = np.round(np.mean(aps[2000:4000]), decimals=float_n)
        sketch_map = np.round(np.mean(aps[4000:6000]), decimals=float_n)
        instre_map = np.round(np.mean(aps[6000:8000]), decimals=float_n)
        sop_map = np.round(np.mean(aps[8000:10000]), decimals=float_n)
        faces_map = np.round(np.mean(aps[10000:]), decimals=float_n)
        gpr1000 = np.round(np.mean([cl_map, iNat_map, sketch_map, instre_map, sop_map]), float_n)

        return all_map, gpr1000, cl_map, iNat_map, sketch_map, instre_map, sop_map, faces_map

    return all_map

def get_GPR1200_dataset(base_path, ppc_fn):
    gpr_class = GPR1200(base_path)

    return TestDataset(gpr_class.image_files, gpr_class.image_categories, ppc_fn=ppc_fn)


