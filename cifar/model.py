import torch
from torch import nn


class CustomModel(nn.Module):
    def __init__(self, opt):
        super(CustomModel, self).__init__()
        
        ####
        self.pre_classifier = nn.Linear(opt.HIDDEN_DIM, opt.HIDDEN_DIM)       
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(opt.HIDDEN_DIM, opt.NUM_LABELS)       
        ####
        
    def forward(self, features):
        ####
        features = torch.mean(features, -1) # (bs, 512, 7)
        features = torch.mean(features, -1) # (bs, 512)
        ####
        features = self.pre_classifier(features) # (bs, 512)
        features = self.relu(features)  # (bs, dim)
        logits = self.classifier(features)  # (bs, 10)
        ####
        
        return logits
