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

from model import CustomModel
from dataset import CustomDataset



def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



if __name__ == '__main__':
    opt = config.parse_opt()
    print(opt)
    ####
    set_seeds(opt.SEED)
    ####
    tokenizer = DistilBertTokenizerFast.from_pretrained(opt.BERT_TYPE)
    
    print(tokenizer.mask_token_id)
    
    def collate_fn(batch):
        label = [b[0] for b in batch]
        text_a = [b[1] for b in batch]
        text_b = [b[2] for b in batch]

        my_dict = tokenizer(text_a, text_b,
                            add_special_tokens=True,
                            padding=True,
                            truncation=True,
                            max_length=opt.MAX_SEQ_LENGTH,
                            return_attention_mask=True,
                            return_tensors='pt'
                           )
        label = torch.tensor(label)   
        return label, my_dict['input_ids'], my_dict['attention_mask']
    ####
    print(os.path.join(opt.DATA_PATH, opt.ORDER, '{}.csv'.format(opt.PERCENTAGE)))
    train = pd.read_csv(os.path.join(opt.DATA_PATH, opt.ORDER, '{}.csv'.format(opt.PERCENTAGE)))
    print(train.info())
#     print(train.head())
    
    train_dev = pd.read_csv(os.path.join(opt.DATA_PATH, opt.TRAIN_DEV_DATA))
    dev = pd.read_csv(os.path.join(opt.DATA_PATH, opt.DEV_DATA))
    ####
    train_dataset = CustomDataset(data=train)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.TRAIN_BATCH_SIZE,
                                      num_workers=0, shuffle=True, 
#                                   pin_memory=True, 
                                  collate_fn=collate_fn)
    ####
    train_dev_dataset = CustomDataset(data=train_dev)
    train_dev_dataloader = DataLoader(train_dev_dataset, batch_size=opt.TEST_BATCH_SIZE,
                                      num_workers=0, shuffle=False, 
#                                       pin_memory=True, 
                                      collate_fn=collate_fn)
    ####
    dev_dataset = CustomDataset(data=dev)
    dev_dataloader = DataLoader(dev_dataset, batch_size=opt.TEST_BATCH_SIZE, 
                                num_workers=0, shuffle=False, 
#                                 pin_memory=True, 
                                collate_fn=collate_fn)
    ####
    total_steps = len(train_dataloader) * opt.EPOCH
    print("total_steps: {}".format(total_steps))
    num_warmup_steps = int(total_steps * 0.0)
    print("num_warmup_steps: {}".format(num_warmup_steps))
    ####
    model = CustomModel(opt)
    for name, param in model.named_parameters():
        print(name)
        print(param.requires_grad)
    model.cuda()
    ####
    no_decay = ['bias', 'output_layer_norm.weight']
    
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if (p.requires_grad==True)]},
    ]
    optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=opt.LEARNING_RATE, momentum=opt.MOMENTUM)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, total_steps)
    ####
    for i in range(opt.EPOCH):
        print('epoch: {}'.format(i))
        model.train()

        for idx, batch in enumerate(tqdm(train_dataloader)):
            ####
            model.zero_grad()
            optimizer.zero_grad()
            ####
            label = batch[0].cuda()
            input_ids = batch[1].cuda()
            attention_mask = batch[2].cuda()        
        
            _, outputs = model(input_ids, True,
                            attention_mask=attention_mask)
        
            ce_loss = F.cross_entropy(outputs, label)

            hack_loss = torch.cat([
                (p**2).view(-1)
                for n, p in model.named_parameters()
                if ((not any(nd in n for nd in no_decay)) and (p.requires_grad==True))
            ]).sum() * (opt.L2_LAMBDA) 
        
            loss = ce_loss + hack_loss
        
            loss.backward()
            ####
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ####
            optimizer.step()
            scheduler.step()
            ####
#             break
        if opt.SAVE_CHECKPOINT:
            filename = os.path.join(opt.OUTPUT, opt.ORDER, str(opt.PERCENTAGE), str(opt.SEED), 'checkpoint/{}.pth'.format(i))
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            torch.save(model.state_dict(), filename)
        ####
        model.eval()
    
        prob_list = []
        predictions_list = []
        for idx, batch in enumerate(tqdm(train_dev_dataloader)):
            label = batch[0].cuda()
    
            input_ids = batch[1].cuda()
            attention_mask = batch[2].cuda()        
        
            with torch.no_grad():
                _, outputs = model(input_ids, True,
                               attention_mask=attention_mask)
            
                prob = F.softmax(outputs, dim=-1)
                prob = torch.gather(prob, 1, label.unsqueeze(1))
                prob_list.append(prob.detach().cpu().numpy())
            
                predictions = outputs.detach().cpu().numpy()
                predictions_list.append(predictions)
    
        probs = np.vstack(prob_list)
        print(probs[0: 5])
        predictions = np.vstack(predictions_list)
        predictions = np.argmax(predictions, axis=1)
        train_dev['prediction'] = predictions
        
        report = classification_report(train_dev['label'], train_dev['prediction'], digits=4, output_dict=True)
        print(report)
        report_df = pd.DataFrame(report).transpose()
    
        filename = os.path.join(opt.OUTPUT, opt.ORDER, str(opt.PERCENTAGE), str(opt.SEED), 'report/{}_{}.csv'.format(i, 'train_dev'))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        report_df.to_csv(filename)

        ####
        model.eval()

        predictions_list = []
        for idx, batch in enumerate(tqdm(dev_dataloader)):
    
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
    
        report = classification_report(dev['label'], dev['prediction'], digits=4, output_dict=True)
        print(report)
        report_df = pd.DataFrame(report).transpose()
    
        filename = os.path.join(opt.OUTPUT, opt.ORDER, str(opt.PERCENTAGE), str(opt.SEED), 'report/{}_{}.csv'.format(i, 'dev'))
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        report_df.to_csv(filename)
    ####
    test = pd.read_csv(os.path.join(opt.DATA_PATH, opt.TEST_DATA))
    test_dataset = CustomDataset(data=test)
    test_dataloader = DataLoader(test_dataset, batch_size=opt.TEST_BATCH_SIZE,
        num_workers=0,
        shuffle=False,
#         pin_memory=True,
        collate_fn=collate_fn)    
    ####
    model.eval()

    predictions_list = []
    for idx, batch in enumerate(tqdm(test_dataloader)):
    
        input_ids = batch[1].cuda()
        attention_mask = batch[2].cuda()                
        
        with torch.no_grad():
            _, outputs = model(input_ids, True, 
                            attention_mask=attention_mask)
            
            predictions = outputs.detach().cpu().numpy()
            predictions_list.append(predictions)
            
    predictions = np.vstack(predictions_list)
    predictions = np.argmax(predictions, axis=1)
    test['prediction'] = predictions
    
    report = classification_report(test['label'], test['prediction'], digits=4, output_dict=True)
    print(report)
    report_df = pd.DataFrame(report).transpose()
    
    filename = os.path.join(opt.OUTPUT, opt.ORDER, str(opt.PERCENTAGE), str(opt.SEED), 'report/{}_{}.csv'.format(i, 'test'))
    os.makedirs(os.path.dirname(filename), exist_ok=True)
        
    report_df.to_csv(filename)
    ####




