import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import BertConfig, BertForMaskedLM
from models.FraBert import FraBert

class BertGAT(BertForMaskedLM):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self,config,max_tag_len):
        super().__init__(config)
        self.under_bert = FraBert(config)
        self.linear = nn.Linear(self.config.hidden_size,self.config.hidden_size)
        self.max_tag_len = max_tag_len
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_type_idx=None,
        token_type_ids=None,
        labels=None,
        above_mask_idx=None
    ):
        device = input_ids.device
        batch_size,tag_len,seq_len = input_ids.size()
        above_orig = torch.zeros((batch_size,tag_len,self.config.hidden_size)).to(device)
        input_ids = input_ids.view(batch_size*tag_len,-1)
        attention_mask = attention_mask.view(batch_size,self.max_tag_len+1,self.max_tag_len+1)
        attention_mask = attention_mask[:,:tag_len+1,:tag_len+1]
        inputs_type_idx = inputs_type_idx.view(batch_size*tag_len,-1)
        token_type_ids = token_type_ids.view(batch_size*tag_len,-1)
        labels=labels.view(batch_size*tag_len,-1)
        outputs = self.under_bert(input_ids=input_ids,
                                  attention_mask=None,
                                  inputs_type_idx=inputs_type_idx,
                                  token_type_ids=token_type_ids,
                                  position_ids=None,
                                  head_mask=None,
                                  inputs_embeds=None,
                                  labels=labels)
        under_loss = outputs['loss']
        above_inputs = outputs['output'][:,0,:]
        above_inputs=above_inputs.view(batch_size,tag_len,-1)
        above_inputs = torch.cat((self.bert.embeddings.word_embeddings(torch.LongTensor([101]*batch_size).to(device).view(batch_size,-1)),above_inputs),dim=1)
        above_orig = above_inputs[above_mask_idx]
        above_inputs[above_mask_idx] = self.bert.embeddings.word_embeddings(torch.LongTensor([103]).to(device)) #[MASK] token idx
        above_outputs = self.bert(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=None,
            inputs_embeds=above_inputs
        )
        above_sequence_output = above_outputs[0]
        above_new = above_sequence_output[above_mask_idx]
        loss_fct = torch.nn.CosineEmbeddingLoss(reduction='mean')
        above_new = self.linear(above_new)
        if above_new.size(0)!=0:
            above_loss = loss_fct(above_new,above_orig,target=torch.LongTensor([1]*above_new.size(0)).to(device).view(-1))
            loss = under_loss
        else:
            loss = under_loss
        return {
            "loss":loss,
            "output":above_sequence_output,
            "embedding_predict":above_new
        }


    # def forward(
    #     self,
    #     all_tag_inputs=None,
    #     attention_mask=None,
    #     token_type_ids=None,
    #     head_mask=None,
    #     masked_idx = None,
    #     inputs_embeds=None,
    #     labels=None,
    # ):
    #     under_bert_mlm_loss = 0
    #     for idx,one_tag_inputs in enumerate(all_tag_inputs):
    #         inputs_id = one_tag_inputs["inputs_id"]
    #         inputs_type_idx = one_tag_inputs["inputs_type_idx"]
    #         token_type_ids = one_tag_inputs["token_type_ids"]
    #         labels = one_tag_inputs["labels"]
    #         one_tag_outputs = self.under_bert(input_ids=input_ids,inputs_type_idx=inputs_type_idx,token_type_ids=token_type_ids
    #                             , labels = labels)
    #         loss = one_tag_outputs['loss']
    #         one_tag_output = one_tag_outputs['output']、
    #         if idx not in masked_idx:
    #             inputs_embeds = torch.cat(inputs_embeds,one_tag_output[:,0,:],dim=1)
    #         under_bert_mlm_loss += loss
    #     above_output = self.bert(
    #         input_ids=None,
    #         attention_mask=attention_mask,
    #         token_type_ids=None,
    #         position_ids=None,
    #         head_mask=None,
    #         inputs_embeds=inputs_embeds)
        
    #     loss_fct = nn.KLDivLoss(reduction='mean')
    #     sequence_output = outputs[0]
    #     sequence_output = self.relu(self.linear(sequence_output))


        


