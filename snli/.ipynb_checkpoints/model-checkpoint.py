import torch
from torch import nn
from transformers import DistilBertModel, DistilBertConfig


class CustomModel(nn.Module):
    def __init__(self, opt):
        super(CustomModel, self).__init__()
        
        self.bert = DistilBertModel.from_pretrained(opt.BERT_TYPE, 
                                              output_hidden_states=False, 
                                              output_attentions=False)
#         configuration = DistilBertConfig()
#         self.bert = DistilBertModel(configuration)

        ####
        # Table 3 in https://arxiv.org/pdf/1911.03090.pdf
        params_to_freeze = [
            "bert.embeddings.",
            "bert.transformer.layer.0.",
            "bert.transformer.layer.1.",
            "bert.transformer.layer.2.",
            "bert.transformer.layer.3.",
        ]
        for name, param in self.named_parameters():
            # if "classifier" not in name:  # classifier layer
            #     param.requires_grad = False

            if any(pfreeze in name for pfreeze in params_to_freeze):
                param.requires_grad = False
        ####
        self.classifier = nn.Linear(opt.HIDDEN_DIM, opt.NUM_LABELS)       
        ####
        
    def forward(self, inputs, is_ids, attention_mask):
        if is_ids:
            last_hidden = self.bert(input_ids=inputs, attention_mask=attention_mask)[0]
        else:
            last_hidden = self.bert(inputs_embeds=inputs, attention_mask=attention_mask)[0]
        ####
        cls_embedding = last_hidden[:, 0, :] # (bs, dim) pooled_output = cls_embedding
        ####
        logits = self.classifier(cls_embedding)  # (bs, num_labels)
        ####
        
        return cls_embedding, logits


