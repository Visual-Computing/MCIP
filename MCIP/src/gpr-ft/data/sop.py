import os
import numpy as np

from data.base import Base
from data.test_dataset import TestDataset
from evaluation.utils import compute_recalls

class SOP(Base):
    def __init__(self, base_dir, mode="eval"):

        
        self.mode = mode
        if self.mode == 'train':
            self.classes = range(0,11318)
        elif self.mode == 'eval':
            self.classes = range(11318,22634)  

        self._base_dir = base_dir
       
        metadata = open(os.path.join(self.base_dir, 'Ebay_train.txt' if self.classes == range(0, 11318) else 'Ebay_test.txt'))
        
        cats, files = [], []

        for i, (_ , class_id, _, path) in enumerate(map(str.split, metadata)):
            if i > 0:
                if int(class_id)-1 in self.classes:
                    cats.append(int(class_id)-1)
                    files.append(os.path.join(self.base_dir, path))

        Base.__init__(self, base_dir, "SOP", cats, files)

def get_SOP_dataset(base_path, ppc_fn):

    dataset = SOP(base_path)
    return TestDataset(dataset.image_files, dataset.image_categories, ppc_fn=ppc_fn)

def evaluate_SOP(cats, features, float_n=4):
    return np.round(compute_recalls(features, cats, [1, 10, 100, 1000]), float_n)
