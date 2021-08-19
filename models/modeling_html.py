import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import sys
from transformers import BertConfig, BertForMaskedLM,BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from models.FraBert import FraBert

class HirachicalBert(FraBert):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self,text_config,node_config,layer_num,num_type=3):
        super().__init__(text_config,num_type=num_type)
        self._keys_to_ignore_on_save = ['node_bert','node_cls']
        self.node_config = node_config
        self.node_bert = BertModel(node_config)
        #self.bert = FraBert(config=config)
        self.node_cls = BertOnlyMLMHead(node_config)
        self.layer_num = layer_num
        self.apply(self._init_weights)
        

    def forward(
            self,
            node_layer_index,
            token_layer_index,
            position,
            waiting_mask,
            node_num,
            seq_num,
            attention_mask = None,
            token_input_ids=None,
            node_input_ids=None,
            inputs_type_idx=None,
            token_type_ids=None,
            node_labels=None,
            token_labels=None,
    ):

        # print("layer_index:"+str(layer_index.size()))
        # print("waiting_mask:"+str(waiting_mask.size()))
        # print("input_ids:"+str(input_ids.size()))
        # print("inputs_type_idx:"+str(inputs_type_idx.size()))
        # print("token_type_ids:"+str(token_type_ids.size()))
        # print("labels:"+str(labels.size()))
        # print(attention_mask.size())
        word_logits = []
        tag_logits = []
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='mean')
        output = []
        batch_size,_,seq_len = token_input_ids.size()
        _,_,node_seq_len = node_input_ids.size()
        for layer in range(self.layer_num):
            cur_layer = self.layer_num - layer
            node_layer_mask = node_layer_index.eq(cur_layer)  #(b,n_num)

            token_layer_mask = token_layer_index.eq(cur_layer) #(b,seq_num)
            if not len(torch.nonzero(node_layer_mask,as_tuple=False)) and not len(torch.nonzero(token_layer_mask,as_tuple=False)):
                continue
            layer_position = position[:,cur_layer-1,:].view(batch_size,-1)     #(b,l_num,l_sum_num)
            if len(output) != 0:
                last_layer_output = output[-1]
            # node_layer_labels = node_labels[node_layer_mask] #(b,n_num,n_len)
            # token_layer_labels = token_labels[token_layer_mask] #(b,seq_num,seq_len)
            if len(torch.nonzero(node_layer_mask,as_tuple=False)):
                layer_waiting_mask = waiting_mask[node_layer_mask].type(torch.bool) #(b,lay_n_num,n_len)
                layer_node_input_ids = node_input_ids[node_layer_mask]  # (b,lay_n_num,n_len)
                layer_node_embeds = self.bert.embeddings.word_embeddings(layer_node_input_ids)  # (b,lay_n_num,n_len,h_dim)
                if len(output) != 0:
                    layer_node_embeds[layer_waiting_mask] = last_layer_output
                layer_node_outputs = self.node_bert(
                    input_ids=None,
                    attention_mask=None,
                    token_type_ids=None,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=layer_node_embeds
                )
                layer_node_output = layer_node_outputs[0]
                layer_node_labels = node_labels[node_layer_mask]
                if len(tag_logits) :
                    tag_logits = torch.cat((tag_logits,self.node_cls(layer_node_output).view(-1,self.node_config.vocab_size)),dim = 0)
                    tag_labels = torch.cat((tag_labels,layer_node_labels),dim=0)
                else:
                    tag_logits = self.node_cls(layer_node_output).view(-1,self.node_config.vocab_size)
                    tag_labels = layer_node_labels
                node_output = layer_node_output[:, 0, :].view(-1, self.config.hidden_size)
            #layer_token_type_ids = token_type_ids[token_layer_mask] #(b,lay_seq_num,seq_len)
            if len(torch.nonzero(token_layer_mask,as_tuple=False)):
                layer_token_input_ids = token_input_ids[token_layer_mask] #(b,lay_seq_num,seq_len)
                layer_inputs_type_idx = inputs_type_idx[token_layer_mask] #(b,lay_n_num,n_len)
                layer_token_embeds = self.bert.embeddings.word_embeddings(layer_token_input_ids)#(b,lay_seq_num,seq_len,h_dim)
                layer_token_embeds += self.type_embedding(layer_inputs_type_idx)#(b,lay_seq_num,seq_len,h_dim)
                layer_text_outputs = self.bert(
                    input_ids = None,
                    attention_mask = None,
                    token_type_ids = None,
                    position_ids = None,
                    head_mask = None,
                    inputs_embeds = layer_token_embeds
                )
            # layer_node_output = layer_node_outputs[0]
                layer_text_output = layer_text_outputs[0]
                layer_toten_labels = token_labels[token_layer_mask]
                if len(word_logits) :
                    word_logits = torch.cat((word_logits,self.cls(layer_text_output).view(-1,self.config.vocab_size)),dim = 0)
                    new_token_labels = torch.cat((new_token_labels,layer_toten_labels),dim=0)
                else:
                    word_logits = self.cls(layer_text_output).view(-1,self.config.vocab_size)
                    new_token_labels = layer_toten_labels
                text_output = layer_text_output[:, 0, :].view(-1, self.config.hidden_size)
                
            layer_node_num = node_num[:,cur_layer-1].view(-1)
            layer_seq_num = seq_num[:,cur_layer-1].view(-1)
            layer_output = []
            x_label = 0
            y_label = 0
            for i in range(batch_size):
                x = layer_node_num[i]
                y = layer_seq_num[i]
                one_layer_position = layer_position[i,:].view(-1)
                if x>0 and y == 0:
                    one_layer_output = node_output[x_label:x_label+x,:]
                    x_label += x
                elif y>0 and x == 0:
                    one_layer_output = text_output[y_label:y+y_label,:]
                    y_label += y
                elif x>0 and y>0:
                    output_node = node_output[x_label:x_label+x,:]
                    output_text = text_output[y_label:y+y_label,:]
                    x_label += x
                    y_label += y
                    one_layer_output = torch.cat((output_node,output_text),dim = 0)
                    output_len = one_layer_output.size(0)
                    one_layer_output = one_layer_output[one_layer_position][:output_len]
                else:
                    continue
                if len(layer_output):
                    layer_output = torch.cat((layer_output,one_layer_output),dim=0)
                else:
                    layer_output = one_layer_output
            output.append(layer_output)
        #print(word_logits.size(),labels.size())
        text_mlm_loss = loss_fct(word_logits, new_token_labels.view(-1))
        node_mlm_loss = loss_fct(tag_logits, tag_labels.view(-1))

        
            
        return {'loss': text_mlm_loss+node_mlm_loss,
                'text_mlm_loss':text_mlm_loss,
                'node_mlm_loss':node_mlm_loss,
                'output':output
                } 
