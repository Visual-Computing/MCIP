import numpy as np
from torch.utils.data import Dataset
import pickle
import torch
import os

class TrainDataset(Dataset):
  
  def __init__(self, file_paths, labels, ppc_fn, exp_params, tokenizer=None, sim_th=0):
        'Initialization'
        self.file_paths = file_paths
        self.labels = labels
        self.uniqe_labels = np.unique(labels)
        self.ppc_fn = ppc_fn
        self.n_classes = len(self.uniqe_labels)
        self.exp_params = exp_params
        self.tokenizer = tokenizer

        self.n_max_texts = 40 # maximum number of pseudo-captions per image

        
  def __len__(self):
        #print(self.file_paths)
        'Denotes the total number of samples'
        return len(self.file_paths)
  
  def _pad_text(self, text_file_name):
    """
    function that pads the pseudo-captions per image to a uniform distributed number
    needed for batching
    """
    vec = np.load(text_file_name)
    if ".npz" in text_file_name:
        vec = vec.f.arr_0
    l = len(vec)
    
    if l == self.n_max_texts: return self.tokenizer(vec)
    else:
        zeros = np.array(["" for i in range(self.n_max_texts-l)])
        padded_vec = np.concatenate([vec, zeros])
        
        return self.tokenizer(padded_vec)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        try:
            # this is not needed if the categories are already numerical
            cat_index = np.where(self.uniqe_labels == self.labels[index])[0]
            if self.tokenizer is None:
                return self.ppc_fn(self.file_paths[index]), cat_index
            else:
                image_file_name = self.file_paths[index]
                #text_file_name = os.path.join(self.exp_params.data.caption_output_dir, os.path.basename(image_file_name).split(".")[0] + f"_text_{self.exp_params.MCIP.sim_th:.3f}.npz")
                text_file_name = image_file_name.replace("/mnt/data/images/", "/mnt/bigdata/features/texts/") + f"_text_EN_{self.exp_params.MCIP.sim_th:.3f}.npz"
                return self.ppc_fn(image_file_name), cat_index, self._pad_text(text_file_name)
        except Exception as e:
            print(e)
            return None

def get_train_ds(train_preprocessing_fn, exp_params, include_text=False, tokenizer=None):

    
    train_cats = np.load(exp_params.data.image_categories)
    train_images = np.load(exp_params.data.image_paths)

    return TrainDataset(train_images, train_cats, train_preprocessing_fn, exp_params, tokenizer=tokenizer)