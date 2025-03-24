import os
from data.base import Base
from data.test_dataset import TestDataset
from gpr_utils.utils import list_all_images_at_dir

class INv2(Base):
    def __init__(self, base_dir, class_to_int_dict):

        self._base_dir = base_dir
        
        cats = []

        files = list_all_images_at_dir(base_dir)
        #print(class_to_int_dict)

        for f in files: cats.append(f.split(os.sep)[-2])
        cats = list(map(lambda c_s: class_to_int_dict[c_s], cats))

        Base.__init__(self, base_dir, "INv2", cats, files)

def get_INv2_dataset(base_path, ppc_fn, class_to_int_dict):

    dataset = INv2(base_path, class_to_int_dict)
    return TestDataset(dataset.image_files, dataset.image_categories, ppc_fn=ppc_fn)


