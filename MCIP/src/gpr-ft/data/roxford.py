import numpy as np

from data.base import Base
from data.test_dataset import TestDataset

from data.revisited_oxford_paris.dataset import configdataset
from data.revisited_oxford_paris.evaluate_r import evaluate

class ROxford(Base):
    def __init__(self, base_dir):

        self._base_dir = base_dir
       
        cfg_oxford = configdataset("roxford5k", base_dir)
        
        files, cats = [], []

        for i in range(len(cfg_oxford['imlist'])):
            files.append(cfg_oxford['im_fname'](cfg_oxford, i))
            cats.append("")
        
        self.num_db = len(files)
        for i in range(len(cfg_oxford['qimlist'])):
            files.append(cfg_oxford['qim_fname'](cfg_oxford, i))
            cats.append("")

        self.cfg = cfg_oxford
        Base.__init__(self, base_dir, "ROxford", cats, files)

def get_ROxford_dataset(base_path, ppc_fn):

    dataset = ROxford(base_path)
    return TestDataset(dataset.image_files, dataset.image_categories, ppc_fn=ppc_fn)

def evaluate_ROxford(base_path, features):
    
    dataset = ROxford(base_path)
    db_fvs = features[:dataset.num_db]
    q_fvs = features[dataset.num_db:]

    return evaluate(db_fvs, q_fvs, dataset.cfg)

