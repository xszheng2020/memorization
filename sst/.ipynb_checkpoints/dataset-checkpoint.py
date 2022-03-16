from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):

        row = self.data.iloc[index]
        
        label = row[0]
        sentence = row[1]

        return label, sentence 
        
    def __len__(self):
        return len(self.data)



class CustomDatasetWithMask(Dataset):
    def __init__(self, data, mask=None):
        self.data = data
        self.mask = mask
        
    def __getitem__(self, index):

        row = self.data.iloc[index]
        
        label = row[0]
        sentence = row[1]
        sample_index = row[2]
        
        mask = self.mask[sample_index]
        
        return label, sentence, mask
        
    def __len__(self):
        return len(self.data)
