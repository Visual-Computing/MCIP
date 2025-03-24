from torch.utils.data import Dataset

class TestDataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, file_paths, labels, ppc_fn):
        'Initialization'
        self.file_paths = file_paths
        self.labels = labels
        self.ppc_fn = ppc_fn
        
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.file_paths)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.ppc_fn(self.file_paths[index])