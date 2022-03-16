# -*- coding: utf-8 -*-
import os
import config

import random
import numpy as np
import pandas as pd
pd.set_option('max_colwidth', 256)

# +
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats

# %matplotlib inline
set_matplotlib_formats('svg')
# -



import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

# +
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torchvision import models
# -



from tqdm import tqdm
from sklearn.metrics import classification_report

import pickle
from contexttimer import Timer

from model import CustomModel
from dataset import CustomDataset, CustomDatasetWithMask




def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def compute_s(model,
              v,
              train_data_loader, 
              damp,
              scale,
              num_samples):
    
    last_estimate = list(v).copy()
    with tqdm(total=num_samples) as pbar:
        for i, batch in enumerate(train_data_loader):
            ####
            labels = batch[1].cuda()
            inputs = batch[2].cuda()
            ####
            this_estimate = compute_hessian_vector_products(model=model,
                                                            vectors=last_estimate,
                                                            labels=labels,
                                                            inputs=inputs,
                                                           )
            # Recursively caclulate h_estimate
            # https://github.com/dedeswim/pytorch_influence_functions/blob/master/pytorch_influence_functions/influence_functions/hvp_grad.py#L118
            with torch.no_grad():
                new_estimate = [
                    a + (1 - damp) * b - c / scale 
                    for a, b, c in zip(v, last_estimate, this_estimate)
                ]
            ####    
            pbar.update(1)
            
            new_estimate_norm = new_estimate[0].norm().item()
            last_estimate_norm = last_estimate[0].norm().item()
            estimate_norm_diff = new_estimate_norm - last_estimate_norm
            pbar.set_description(f"{new_estimate_norm:.2f} | {estimate_norm_diff:.2f}")
            ####
            last_estimate = new_estimate
            
            if i > num_samples: # should be i>=(num_samples-1) but does not matters
                break
                
    # References:
    # https://github.com/kohpangwei/influence-release/blob/master/influence/genericNeuralNet.py#L475
    # Do this for each iteration of estimation
    # Since we use one estimation, we put this at the end
    inverse_hvp = [X / scale for X in last_estimate]
    
    return inverse_hvp


def compute_hessian_vector_products(model,
                                    vectors,
                                    labels,
                                    inputs):
    ####
    outputs = model(inputs)
    ce_loss = F.cross_entropy(outputs, labels)
    ####
    hack_loss = torch.cat([
        (p**2).view(-1)
        for n, p in model.named_parameters()
        if ((not any(nd in n for nd in no_decay)) and (p.requires_grad==True))
    ]).sum() * (opt.L2_LAMBDA)     
    ####
    loss = ce_loss + hack_loss
    ####
    model.zero_grad()
    grad_tuple = torch.autograd.grad(
        outputs=loss,
        inputs=[param for name, param in model.named_parameters()
                if param.requires_grad], 
        create_graph=True)
    ####
#     model.zero_grad()
    grad_grad_tuple = torch.autograd.grad(
        outputs=grad_tuple,
        inputs=[param for name, param in model.named_parameters() 
                if param.requires_grad],
        grad_outputs=vectors,
        only_inputs=True
    )

    return grad_grad_tuple



# +
if __name__ == '__main__':
    opt = config.parse_opt_if()
    print(opt)
    ####
    set_seeds(opt.SEED)
    ####
    input_size = 224

    transform_test = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])    
    ####
    print(os.path.join(opt.DATA_PATH, opt.ORDER, '{}.csv'.format(opt.PERCENTAGE)))
    train = pd.read_csv(os.path.join(opt.DATA_PATH, opt.ORDER, '{}.csv'.format(opt.PERCENTAGE)))
    print(train.info())
    
    train_dev = pd.read_csv(os.path.join(opt.DATA_PATH, opt.TRAIN_DEV_DATA))
    dev = pd.read_csv(os.path.join(opt.DATA_PATH, opt.DEV_DATA))    
    ####
    train_dataset = CustomDataset(train, root='./data/cifar_10/train/')
    ####
    with open(os.path.join(opt.DATA_PATH, opt.ATTR_ORDER, '{}.pkl'.format(opt.ATTR_PERCENTAGE)), "rb") as handle:
        mask_list = pickle.load(handle)
    
    print(mask_list[0:5])    
    
    train_dev_dataset = CustomDatasetWithMask(data=train_dev, root='./data/cifar_10/train/', mask=mask_list)
    train_dev_dataloader = DataLoader(train_dev_dataset, batch_size=1,
                                               shuffle=False, 
#                                                pin_memory=True, 
                                               num_workers=0)
    ####
    dev_dataset = CustomDataset(dev, root='./data/cifar_10/train/')
    dev_dataloader = DataLoader(dev_dataset, batch_size=opt.TEST_BATCH_SIZE, 
                                         shuffle=False, 
#                                          pin_memory=True, 
                                         num_workers=0)
    ####
    model = CustomModel(opt)
    for name, param in model.named_parameters():
        print(name)
        print(param.requires_grad)
    model.cuda()

    filename = os.path.join(opt.OUTPUT, opt.ORDER, str(opt.PERCENTAGE), str(opt.CHECKPOINT), 'checkpoint/{}.pth'.format(opt.EPOCH-1))
    model.load_state_dict(
        torch.load(filename)
    )   
    
    no_decay = ['bias', 'LayerNorm.weight']

    model.cuda()
    model.eval()
    ####
#     predictions_list = []
#     for idx, batch in enumerate(tqdm(dev_dataloader)):
    
#         labels = batch[1].cuda()
#         inputs = batch[2].cuda()
    
#         with torch.no_grad():
#             outputs = model(inputs)    
        
#             predictions = outputs.detach().cpu().numpy()
#             predictions_list.append(predictions)
        
#     predictions = np.vstack(predictions_list)
#     predictions = np.argmax(predictions, axis=1)
#     dev['prediction'] = predictions

#     print(classification_report(dev['label'], dev['prediction'], digits=4))  
    ####
    start = opt.START
    length = opt.LENGTH
    print(start, length)

    output_collections = []
    #### 
    for idx, batch in enumerate(train_dev_dataloader):
         ####
        if idx < start:
            continue
        if idx >= start+length:
            break
    ####
        z_index = batch[0]
        z_labels = batch[1].cuda()
        z_inputs = batch[2].cuda()
        z_mask = batch[3].cuda()
    ####
        outputs = model(z_inputs)
    ####
        prob = F.softmax(outputs, dim=-1)
        prediction = torch.argmax(prob, dim=1)
        
#         if prediction==z_labels:
#             continue
            
        prob_gt = torch.gather(prob, 1, z_labels.unsqueeze(1))
    #     print(prob_gt)
    ####    
        model.zero_grad()

        v = torch.autograd.grad(outputs=prob_gt, 
                                inputs=[param for name, param in model.named_parameters() 
                                        if param.requires_grad],
                                create_graph=False)
    ####
        for repetition in range(4): # here we do not repeat      
            with Timer() as timer:
            ####
                train_dataloader = DataLoader(train_dataset, 
                                              batch_size=1, 
                                              shuffle=True, 
#                                               pin_memory=True,
                                              num_workers=0)
            ####
                s = compute_s(model=model,
                              v=v,
                              train_data_loader=train_dataloader,
                              damp=opt.DAMP, 
                              scale=opt.SCALE,
                              num_samples=opt.NUM_SAMPLES)
            ####
                time_elapsed = timer.elapsed
    #             print(f"{time_elapsed:.2f} seconds")
 
        ####
            ####
            z_inputs_masked = z_inputs.clone() # 1, 2048, 7, 7
            z_mask = z_mask.unsqueeze(1).expand(-1, 2048, -1, -1) # 1, 2048, 7, 7
            z_inputs_masked.masked_fill_(z_mask==1, 0.0)
            ####
            outputs = model(z_inputs_masked)
            ####
            ce_loss_gt = F.cross_entropy(outputs, z_labels)
            z_hack_loss = torch.cat([
            (p**2).view(-1)
            for n, p in model.named_parameters()
            if ((not any(nd in n for nd in no_decay)) and (p.requires_grad==True))
            ]).sum() * (opt.L2_LAMBDA)
            ####
            model.zero_grad()          

            grad_tuple_ = torch.autograd.grad(outputs=ce_loss_gt+z_hack_loss, 
                                                  inputs=[param for name, param in model.named_parameters() 
                                                          if param.requires_grad],
                                                  create_graph=True) 
            
        ####  
            influence = [-torch.sum(x * y) for x, y in zip(s, grad_tuple_)]
            influence = sum(influence).item()
        ####        
            outputs = {
            "index": z_index.detach().cpu().numpy()[0],
            "label": z_labels.detach().cpu().numpy()[0],
            "prob": prob.detach().cpu().numpy()[0],
            "prediction": prediction.detach().cpu().numpy()[0],
            "influence": influence,
            "repetition": repetition,
            "time_elapsed": time_elapsed,
            }        
            
            print(idx)
            print(outputs['index'])
            print(outputs['label'], outputs['prob'], outputs['prediction'])
            print('influence: ', outputs['influence'])
        
            output_collections.append(outputs)
            ####
            break
# +
filename = os.path.join(opt.OUTPUT, 'eval_attr/{}/{}/{}.pkl'.format(opt.ATTR_ORDER, opt.ATTR_PERCENTAGE, start))
os.makedirs(os.path.dirname(filename), exist_ok=True)

with open(filename, "wb") as handle:
    pickle.dump(output_collections, handle) 
# -



