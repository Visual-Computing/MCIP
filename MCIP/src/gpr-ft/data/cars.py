import os
import numpy as np
import scipy.io

from data.base import Base
from data.test_dataset import TestDataset
from evaluation.utils import compute_recalls

class Cars196(Base):
    def __init__(self, base_dir, mode="eval"):

        
        self.mode = mode
        if self.mode == 'train':
            self.classes = range(0,98)
        elif self.mode == 'eval':
            self.classes = range(98,196)

        self._base_dir = base_dir
        
        cats, files = [], []
        annos_fn = 'cars_annos.mat'
        cars = scipy.io.loadmat(os.path.join(self.base_dir, annos_fn))
        ys = [int(a[5][0] - 1) for a in cars['annotations'][0]]
        im_paths = [a[0][0] for a in cars['annotations'][0]]

        for im_path, y in zip(im_paths, ys):
            if y in self.classes: # choose only specified classes
                files.append(os.path.join(self.base_dir, im_path))
                cats.append(y)

        Base.__init__(self, base_dir, "Cars196", cats, files)

def get_Cars_dataset(base_path, ppc_fn):

    dataset = Cars196(base_path)
    return TestDataset(dataset.image_files, dataset.image_categories, ppc_fn=ppc_fn)

def evaluate_Cars(cats, features, float_n=4):
    return np.round(compute_recalls(features, cats, [1, 2, 8, 32]), float_n)
