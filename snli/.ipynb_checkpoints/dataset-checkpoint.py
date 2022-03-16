from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):

        row = self.data.iloc[index]
        
        label = row[0]
        text_a = row[1]
        text_b = row[2]

        return label, text_a, text_b
        
    def __len__(self):
        return len(self.data)


