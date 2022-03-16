# -*- coding: utf-8 -*-
import os
import config

import random
import numpy as np
import pandas as pd
pd.set_option('max_colwidth', 256)

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from transformers import DistilBertTokenizerFast, AdamW, get_linear_schedule_with_warmup

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
            label = batch[0].cuda()
            input_ids = batch[1].cuda()
            ####
            attention_mask = batch[2].cuda()
            ####
            this_estimate = compute_hessian_vector_products(model=model,
                                                            vectors=last_estimate,
                                                            label=label,
                                                            input_ids=input_ids,
                                                            attention_mask=attention_mask,
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
                                    label,
                                    input_ids,
                                    attention_mask):
    ####
    _, outputs = model(inputs=input_ids, is_ids=True,
                    attention_mask=attention_mask, 
                    )
    ce_loss = F.cross_entropy(outputs, label)
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
    tokenizer = DistilBertTokenizerFast.from_pretrained(opt.BERT_TYPE)
    
    def collate_fn(batch):
        label = [b[0] for b in batch]
        text_a = [b[1] for b in batch]
        my_dict = tokenizer(text_a, None,
                            add_special_tokens=True,
                            padding=True,
                            truncation=True,
                            max_length=opt.MAX_SEQ_LENGTH,
                            return_attention_mask=True,
                            return_tensors='pt'
                           )
        label = torch.tensor(label)   
        return label, my_dict['input_ids'], my_dict['attention_mask']

    from torch.nn.utils.rnn import pad_sequence
    def collate_fn_(batch):
        label = [b[0] for b in batch]
        text_a = [b[1] for b in batch]
        mask = [torch.tensor(b[2]) for b in batch]
        
        my_dict = tokenizer(text_a, None,
                        add_special_tokens=True,
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_attention_mask=True,
                        return_tensors='pt'
                        )
        input_ids = my_dict['input_ids']
    #     print(input_ids.size())
        mask = pad_sequence(mask, batch_first=True)
    #     print(mask)
    #     print(mask.size())
    
        input_ids_masked = input_ids.clone()
    #     print(input_ids_masked)
        input_ids_masked.masked_fill_(mask==1, 103)
    #     print(input_ids_masked)
    
        label = torch.tensor(label)   
    
        return label, input_ids, my_dict['attention_mask'], input_ids_masked
    ####
    train = pd.read_csv(os.path.join(opt.DATA_PATH, opt.ORDER, '{}.csv'.format(opt.PERCENTAGE)))
    
    train_dev = pd.read_csv(os.path.join(opt.DATA_PATH, opt.TRAIN_DEV_DATA))
    dev = pd.read_csv(os.path.join(opt.DATA_PATH, opt.DEV_DATA))
    ####
    train_dataset = CustomDataset(data=train)
#     train_dataloader = DataLoader(train_dataset, batch_size=opt.TRAIN_BATCH_SIZE,
#                                       num_workers=32, shuffle=True, pin_memory=True, collate_fn=collate_fn)
    ####
    with open(os.path.join(opt.DATA_PATH, opt.ATTR_ORDER, '{}.pkl'.format(opt.ATTR_PERCENTAGE)), "rb") as handle:
        mask_list = pickle.load(handle)
    
    print(mask_list[0:5])    
    
    train_dev_dataset = CustomDatasetWithMask(data=train_dev, mask=mask_list,)
    train_dev_dataloader = DataLoader(train_dev_dataset, batch_size=1,
                                      num_workers=0, shuffle=False, 
#                                       pin_memory=True, 
                                      collate_fn=collate_fn_)
    ####
    dev_dataset = CustomDataset(data=dev)
    dev_dataloader = DataLoader(dev_dataset, batch_size=opt.TEST_BATCH_SIZE, 
                                num_workers=0, shuffle=False, 
#                                 pin_memory=True, 
                                collate_fn=collate_fn)
    ####
    model = CustomModel(opt)
    
#     for name, param in model.named_parameters():
#         print(name)
#         print(param.requires_grad)

    filename = os.path.join(opt.OUTPUT, opt.ORDER, str(opt.PERCENTAGE), str(opt.CHECKPOINT), 'checkpoint/{}.pth'.format(opt.EPOCH-1))
    model.load_state_dict(
        torch.load(filename)
    )   
    
    no_decay = ['bias', 'output_layer_norm.weight']

    model.cuda()
    model.eval()
    ####
    predictions_list = []
    for idx, batch in enumerate(tqdm(dev_dataloader)):
    #     label = batch[0].cuda()
    
        input_ids = batch[1].cuda()
        attention_mask = batch[2].cuda()
    
        with torch.no_grad():
            _, outputs = model(input_ids, True, 
                           attention_mask=attention_mask)    
        
            predictions = outputs.detach().cpu().numpy()
            predictions_list.append(predictions)
        
    predictions = np.vstack(predictions_list)
    predictions = np.argmax(predictions, axis=1)
    dev['prediction'] = predictions

    print(classification_report(dev['label'], dev['prediction'], digits=4))
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
        z_label = batch[0].cuda()
        z_input_ids = batch[1].cuda()
        z_attention_mask = batch[2].cuda()
        
        z_input_ids_masked = batch[3].cuda()
    ####
        row = train_dev.iloc[idx]
    ####
        _, outputs = model(inputs=z_input_ids, is_ids=True,
                        attention_mask=z_attention_mask)
    ####
        prob = F.softmax(outputs, dim=-1)
        prediction = torch.argmax(prob, dim=1)
        
#         if prediction==z_label:
#             continue
            
        prob_gt = torch.gather(prob, 1, z_label.unsqueeze(1))
    #     print(prob_gt)
    ####    
        model.zero_grad()

        v = torch.autograd.grad(outputs=prob_gt, 
                                inputs=[param for name, param in model.named_parameters() 
                                        if param.requires_grad],
                                create_graph=False) 
    ####
        for repetition in range(4):        
            with Timer() as timer:
            ####
                train_dataloader = DataLoader(
                    train_dataset,
                    batch_size=1,
                    num_workers=0,
                    shuffle=True,
#                     pin_memory=True,
                    collate_fn=collate_fn
                )
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
            _, outputs = model(inputs=z_input_ids_masked, is_ids=True,
                                attention_mask=z_attention_mask)
            ####
            ce_loss_gt = F.cross_entropy(outputs, z_label)
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
                                                  create_graph=False)  
            
        ####  
            influence = [-torch.sum(x * y) for x, y in zip(s, grad_tuple_)]
            influence = sum(influence).item()
        ####        
            outputs = {
            "index": idx,
            "sentence": row['sentence'],
            "label": row['label'],
            "prob": prob.detach().cpu().numpy()[0],
            "prediction": prediction.detach().cpu().numpy()[0],
            "influence": influence,
            "repetition": repetition,
            "time_elapsed": time_elapsed,
            }        
            
            print(outputs['index'])
            print(outputs['sentence'])
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



