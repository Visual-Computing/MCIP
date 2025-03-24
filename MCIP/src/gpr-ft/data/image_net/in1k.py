import os
import numpy as np
from data.base import Base
from data.test_dataset import TestDataset
from gpr_utils.utils import list_all_images_at_dir


class IN1k(Base):
    def __init__(self, base_dir, mode="val", class_to_int_dict=None):

        self._base_dir = os.path.join(base_dir, mode)
        
        cats = []

        files = list_all_images_at_dir(self.base_dir)
        for f in files: cats.append(f.split(os.sep)[-2])

        if class_to_int_dict is None:
            self.class_to_int_dict = self.get_class_to_int_dict(cats)
        else:
            self.class_to_int_dict = class_to_int_dict
        
        cats = list(map(lambda c_s: self.class_to_int_dict[c_s], cats))
        Base.__init__(self, base_dir, "IN1k", cats, files)

    def get_class_to_int_dict(self, categories):

        unique_cats = sorted(np.unique(categories))
        d = {}
        for i, c in enumerate(unique_cats):
            d[c] = i
        return d

def get_IN1k_dataset(base_path, ppc_fn, class_to_int_dict=None, mode="val", include_class_dict=False):

    dataset = IN1k(base_path, mode, class_to_int_dict)
    if include_class_dict:
        return TestDataset(dataset.image_files, dataset.image_categories, ppc_fn=ppc_fn), dataset.class_to_int_dict
    else:
        return TestDataset(dataset.image_files, dataset.image_categories, ppc_fn=ppc_fn)


