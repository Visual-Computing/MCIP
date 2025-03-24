import os
import numpy as np

from data.base import Base
from data.test_dataset import TestDataset
from gpr_utils.utils import list_all_images_at_dir

class Flickr100k(Base):
    def __init__(self, base_dir):

        self._base_dir = base_dir
        
        cats = []

        files = list_all_images_at_dir(base_dir)

        for _ in files: cats.append("Flickr100k")

        Base.__init__(self, base_dir, "Flickr100k", cats, files)

def get_Flickr100k_dataset(base_path, ppc_fn):

    dataset = Flickr100k(base_path)
    return TestDataset(dataset.image_files, dataset.image_categories, ppc_fn=ppc_fn)

