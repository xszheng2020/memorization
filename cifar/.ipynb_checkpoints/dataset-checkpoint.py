import numpy as np
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data, root):
        self.data = data
        self.root = root
        
    def __getitem__(self, index):

        row = self.data.iloc[index]
        
        cifar_index = row[0]
        label = row[1]

            
        with open(self.root + '{}.npy'.format(str(cifar_index)), 'rb') as f:
            feature = np.load(f)
            feature = feature[0] # 2048 7 7
        
        return cifar_index, label, feature
        
    def __len__(self):
        return len(self.data)



class CustomDatasetWithMask(Dataset):
    def __init__(self, data, root, mask=None):
        self.data = data
        self.root = root

        self.mask = mask
        
    def __getitem__(self, index):

        row = self.data.iloc[index]
                
        cifar_index = row[0]
        label = row[1]
        sample_index = row[2]
        
        mask = self.mask[sample_index]
        
        with open(self.root + '{}.npy'.format(str(cifar_index)), 'rb') as f:
            feature = np.load(f)
            feature = feature[0] # 2048 7 7
            
        return cifar_index, label, feature, mask
        
    def __len__(self):
        return len(self.data)




