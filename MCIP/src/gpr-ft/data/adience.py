import numpy as np

from data.base import Base
from data.test_dataset import TestDataset
from gpr_utils.utils import list_all_images_at_dir
from evaluation.utils import compute_mean_average_precision

class AdienceFaces(Base):
    def __init__(self, base_dir):

        self._base_dir = base_dir
        
        cats = []

        files = list_all_images_at_dir(base_dir)

        for f in files: cats.append(f.split(".")[-3])

        Base.__init__(self, base_dir, "AdienceFaces", cats, files)

def get_AdienceFaces_dataset(base_path, ppc_fn):

    dataset = AdienceFaces(base_path)
    return TestDataset(dataset.image_files, dataset.image_categories, ppc_fn=ppc_fn)

def evaluate_AdienceFaces(cats, features, float_n=4):

    aps = compute_mean_average_precision(cats, features_DB=features)
    all_map = np.round(np.mean(aps), decimals=float_n)
    return all_map
