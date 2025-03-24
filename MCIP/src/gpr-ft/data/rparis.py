import numpy as np

from data.base import Base
from data.test_dataset import TestDataset

from data.revisited_oxford_paris.dataset import configdataset
from data.revisited_oxford_paris.evaluate_r import evaluate

class RParis(Base):
    def __init__(self, base_dir):

        self._base_dir = base_dir
       
        cfg_paris = configdataset("rparis6k", base_dir)
        
        files, cats = [], []

        for i in range(len(cfg_paris['imlist'])):
            files.append(cfg_paris['im_fname'](cfg_paris, i))
            cats.append("")
        
        self.num_db = len(files)
        for i in range(len(cfg_paris['qimlist'])):
            files.append(cfg_paris['qim_fname'](cfg_paris, i))
            cats.append("")

        self.cfg = cfg_paris
        Base.__init__(self, base_dir, "RParis", cats, files)

def get_RParis_dataset(base_path, ppc_fn):

    dataset = RParis(base_path)
    return TestDataset(dataset.image_files, dataset.image_categories, ppc_fn=ppc_fn)

def evaluate_RParis(base_path, features):
    
    dataset = RParis(base_path)
    db_fvs = features[:dataset.num_db]
    q_fvs = features[dataset.num_db:]

    return evaluate(db_fvs, q_fvs, dataset.cfg)

