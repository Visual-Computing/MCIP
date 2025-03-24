import os
import numpy as np

from data.base import Base
from data.test_dataset import TestDataset
from evaluation.utils import compute_recalls
import torchvision

class CUB200_2011(Base):
    def __init__(self, base_dir, mode="eval"):

        
        self.mode = mode
        if self.mode == 'train':
            self.classes = range(0,100)
        elif self.mode == 'eval':
            self.classes = range(100,200)

        self._base_dir = base_dir
        
        cats, files = [], []

        for i in torchvision.datasets.ImageFolder(root=base_dir).imgs:
            # i[1]: label, i[0]: root
            y = i[1]
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.classes and fn[:2] != '._':
                cats.append(y)
                files.append(os.path.join(self.base_dir, i[0]))


        Base.__init__(self, base_dir, "CUB200_2011", cats, files)

def get_CUB_dataset(base_path, ppc_fn):

    dataset = CUB200_2011(base_path)
    return TestDataset(dataset.image_files, dataset.image_categories, ppc_fn=ppc_fn)

def evaluate_CUB(cats, features, float_n=4):
    return np.round(compute_recalls(features, cats, [1, 2, 8, 32]), float_n)
