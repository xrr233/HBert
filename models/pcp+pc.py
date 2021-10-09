import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.autograd import Variable
import sys
from transformers import BertConfig, BertForMaskedLM, BertModel
from transformers.models.bert.modeling_bert import BertOnlyMLMHead
from models.FraBert import FraBert
import random



class MLP(nn.Module):
    def __init__(self, hidden_size, mid_size, out_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hidden_size, mid_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(mid_size, out_size)
        self.modules()

    def forward(self, input):
        return torch.nn.functional.softmax(self.fc2(self.relu(self.fc1(input))))



class HirachicalBert(FraBert):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self, text_config, node_config, layer_num, num_type=3):
        super().__init__(text_config, num_type=num_type)
        self._keys_to_ignore_on_save = ['node_bert', 'node_cls']
        self.node_config = node_config
        self.node_bert = BertModel(node_config)
        # self.bert = FraBert(config=config)
        # self.node_cls = BertOnlyMLMHead(node_config)
        self.layer_num = layer_num
        self.hidden_size = text_config.hidden_size
        self.linear = nn.Linear(text_config.hidden_size, text_config.hidden_size)
        self.apply(self._init_weights)
        #self.linear = MLP(text_config.hidden_size,128,2)

    def get_embedding(self, output, waiting_mask, node_input_ids):
        batch_size, tag_len, node_len = waiting_mask.size()
        all_embeddings = self.bert.embeddings.word_embeddings(node_input_ids)
        mask_embeddings = self.bert.embeddings.word_embeddings(node_input_ids)
        neg_embeddings = self.bert.embeddings.word_embeddings(node_input_ids)
        output.reverse()
        output = output[1:]
        res = torch.cat(output, dim=0)
        all_embeddings[waiting_mask.type(torch.bool)] = res
        neg_embeddings[waiting_mask.type(torch.bool)] = res
        mask_embeddings[waiting_mask.type(torch.bool)] = res

        return all_embeddings, neg_embeddings, mask_embeddings


    def get_task_embeddings(self, embeddings, negetive_embeddings, mask_embeddings,waiting_mask):
        device = embeddings.device
        batch_size, tag_len, node_len, hidden_size = embeddings.size()
        tag_labels = torch.zeros(batch_size,tag_len,node_len).to(device)
        for i in range(batch_size):
            for j in range(tag_len):
                tag = 0
                node_num = len(torch.nonzero(waiting_mask[i,j],as_tuple=False))
                if node_num==0:
                    continue
                mask_num = max(1,int(node_num*0.15))
                temp = list(range(2,node_num+2))
                random.shuffle(temp)
                for k in range(mask_num):
                    if waiting_mask[i, j, temp[k]]:
                        # negetive_embeddings[i][j][k] = self.sample_one_embedding(waiting_mask, negetive_embeddings, i,j)
                        #
                        negetive_embeddings[i][j][temp[k]] = self.sample_one_embedding(waiting_mask, negetive_embeddings, i,j)
                        mask_embeddings[i][j][temp[k]] = self.bert.embeddings.word_embeddings(torch.tensor(103).to(device))
                        tag_labels[i,j,temp[k]] = 1
        children_num = torch.sum(waiting_mask, dim=-1)  # (batch_size,tag_len)
        candidates_mask = children_num.gt(1)
        embeddings = embeddings[candidates_mask]
        mask_embeddings = mask_embeddings[candidates_mask]
        negetive_embeddings = negetive_embeddings[candidates_mask]
        tag_labels = tag_labels[candidates_mask]
        candidates_len = len(embeddings)
        train_embeddings = torch.cat((embeddings, mask_embeddings), dim=0)
        # labels= torch.cat((torch.ones(candidates_len).long(), torch.zeros(candidates_len).long()), dim=0).to(device)
        return train_embeddings, negetive_embeddings,tag_labels.bool()

    # [tag_len,node_len,hidden_size]

    def sample_one_embedding(self, waiting_mask, embeddings, i, j):
        batch_size, _, _ = waiting_mask.size()

        temp = list(range(batch_size))
        random.shuffle(temp)
        for x in temp:
            if x == i :
                continue
            else:
                sample_batch = waiting_mask[x].clone().detach()
                candidates = torch.nonzero(sample_batch, as_tuple=False)
                random.shuffle(candidates)
                # if len(candidates):
                sample_index = candidates[0]
                return embeddings[x, sample_index[0], sample_index[1]]
        #
        # sample_batch = waiting_mask[i].clone().detach()
        # sample_batch[j] = 0
        # candidates = torch.nonzero(sample_batch, as_tuple=False)
        # random.shuffle(candidates)
        # # if len(candidates):
        # sample_index = candidates[0]
        # return embeddings[i, sample_index[0], sample_index[1]]
        # # else :
        #     return 0
    def masked_item_prediction(self,sequence_output,target_item):
        sequence_output = self.linear(sequence_output)
        sequence_output = torch.nn.functional.normalize(sequence_output,p=2,dim=1)
        sequence_output = sequence_output.view([-1, self.hidden_size])
        target_item = target_item.view([-1, self.hidden_size])
        score = torch.mul(sequence_output, target_item)
        #print(torch.sum(score, -1)[task1_labels.flatten()])
        return torch.sum(score, -1)
    # def sample_one_embedding(self,waiting_mask,embeddings,i):
    #     batch_size,_,_ = waiting_mask.size()
    #     temp = list(range(batch_size))
    #     random.shuffle(temp)
    #     for j in temp:
    #         if j==i:
    #             continue
    #         else:
    #             sample_batch = waiting_mask[j]
    #             candidates = torch.nonzero(sample_batch,as_tuple=False)
    #             random.shuffle(candidates)
    #             sample_index = candidates[0]
    #             return embeddings[j,sample_index[0],sample_index[1]]

    def forward(
            self,
            node_layer_index,
            token_layer_index,
            position,
            waiting_mask,
            node_num,  # (batch_size,layer_num,1)
            seq_num,  # (batch_size,layer_num,1)
            attention_mask=None,
            token_input_ids=None,
            node_input_ids=None,  # (batch_size,node_num,node_len)
            inputs_type_idx=None,
            token_type_ids=None,
            node_labels=None,  # (batch_size,node_num,node_len)
            token_labels=None,
    ):
        node_input_ids[node_labels.ne(-100)] = node_labels[node_labels.ne(-100)]
        # print("layer_index:"+str(layer_index.size()))
        # print("waiting_mask:"+str(waiting_mask.size()))
        # print("input_ids:"+str(input_ids.size()))
        # print("inputs_type_idx:"+str(inputs_type_idx.size()))
        # print("token_type_ids:"+str(token_type_ids.size()))
        # print("labels:"+str(labels.size()))
        # print(attention_mask.size())
        word_logits = []
        # tag_logits = []
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='mean')
        task1_loss_fct = torch.nn.CosineEmbeddingLoss(margin=0.0,size_average=None,reduction='mean')
        pc_loss_fct = CrossEntropyLoss(reduction='mean')
        output = []
        batch_size, _, seq_len = token_input_ids.size()
        _, _, node_seq_len = node_input_ids.size()
        for layer in range(self.layer_num):
            cur_layer = self.layer_num - layer
            node_layer_mask = node_layer_index.eq(cur_layer)  # (b,n_num)

            token_layer_mask = token_layer_index.eq(cur_layer)  # (b,seq_num)
            if not len(torch.nonzero(node_layer_mask, as_tuple=False)) and not len(
                    torch.nonzero(token_layer_mask, as_tuple=False)):
                continue
            layer_position = position[:, cur_layer - 1, :].view(batch_size, -1)  # (b,l_num,l_sum_num)
            if len(output) != 0:
                last_layer_output = output[-1]
            # node_layer_labels = node_labels[node_layer_mask] #(b,n_num,n_len)
            # token_layer_labels = token_labels[token_layer_mask] #(b,seq_num,seq_len)
            if len(torch.nonzero(node_layer_mask, as_tuple=False)):
                layer_waiting_mask = waiting_mask[node_layer_mask].type(torch.bool)  # (b,lay_n_num,n_len)
                layer_node_input_ids = node_input_ids[node_layer_mask]  # (b,lay_n_num,n_len)
                layer_node_embeds = self.bert.embeddings.word_embeddings(
                    layer_node_input_ids)  # (b,lay_n_num,n_len,h_dim)
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
                # layer_node_labels = node_labels[node_layer_mask]
                # if len(tag_logits) :
                #     tag_logits = torch.cat((tag_logits,self.node_cls(layer_node_output).view(-1,self.node_config.vocab_size)),dim = 0)
                #     tag_labels = torch.cat((tag_labels,layer_node_labels),dim=0)
                # else:
                #     tag_logits = self.node_cls(layer_node_output).view(-1,self.node_config.vocab_size)
                #     tag_labels = layer_node_labels
                node_output = layer_node_output[:, 0, :].view(-1, self.config.hidden_size)
            # layer_token_type_ids = token_type_ids[token_layer_mask] #(b,lay_seq_num,seq_len)
            if len(torch.nonzero(token_layer_mask, as_tuple=False)):
                layer_token_input_ids = token_input_ids[token_layer_mask]  # (b,lay_seq_num,seq_len)
                layer_inputs_type_idx = inputs_type_idx[token_layer_mask]  # (b,lay_n_num,n_len)
                layer_token_embeds = self.bert.embeddings.word_embeddings(
                    layer_token_input_ids)  # (b,lay_seq_num,seq_len,h_dim)
                layer_token_embeds += self.type_embedding(layer_inputs_type_idx)  # (b,lay_seq_num,seq_len,h_dim)
                layer_text_outputs = self.bert(
                    input_ids=None,
                    attention_mask=None,
                    token_type_ids=None,
                    position_ids=None,
                    head_mask=None,
                    inputs_embeds=layer_token_embeds
                )
                # layer_node_output = layer_node_outputs[0]
                layer_text_output = layer_text_outputs[0]
                layer_toten_labels = token_labels[token_layer_mask]
                if len(word_logits):
                    word_logits = torch.cat((word_logits, self.cls(layer_text_output).view(-1, self.config.vocab_size)),
                                            dim=0)
                    new_token_labels = torch.cat((new_token_labels, layer_toten_labels), dim=0)
                else:
                    word_logits = self.cls(layer_text_output).view(-1, self.config.vocab_size)
                    new_token_labels = layer_toten_labels
                text_output = layer_text_output[:, 0, :].view(-1, self.config.hidden_size)

            layer_node_num = node_num[:, cur_layer - 1].view(-1)
            layer_seq_num = seq_num[:, cur_layer - 1].view(-1)
            layer_output = []
            x_label = 0
            y_label = 0
            for i in range(batch_size):
                x = layer_node_num[i]
                y = layer_seq_num[i]
                one_layer_position = layer_position[i, :].view(-1)
                if x > 0 and y == 0:
                    one_layer_output = node_output[x_label:x_label + x, :]
                    x_label += x
                elif y > 0 and x == 0:
                    one_layer_output = text_output[y_label:y + y_label, :]
                    y_label += y
                elif x > 0 and y > 0:
                    output_node = node_output[x_label:x_label + x, :]
                    output_text = text_output[y_label:y + y_label, :]
                    x_label += x
                    y_label += y
                    one_layer_output = torch.cat((output_node, output_text), dim=0)
                    output_len = one_layer_output.size(0)
                    one_layer_output = one_layer_output[one_layer_position][:output_len]
                else:
                    continue
                if len(layer_output):
                    layer_output = torch.cat((layer_output, one_layer_output), dim=0)
                else:
                    layer_output = one_layer_output
            output.append(layer_output)

        all_embeddings,neg_embeddings,mask_embeddings = self.get_embedding(output,waiting_mask,node_input_ids)
        train_embeddings,negetive_embeddings,task1_labels = self.get_task_embeddings(all_embeddings,neg_embeddings,mask_embeddings,waiting_mask)
        cls_len = int(len(negetive_embeddings))
        pc_positive = train_embeddings[:cls_len][task1_labels]
        pc_negetive = negetive_embeddings[task1_labels]


        all_node_outputs = self.node_bert(
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=train_embeddings,
        )
        node_output = all_node_outputs[0]
        device = node_output.device
        positive = node_output[:cls_len]
        positive = positive[task1_labels]
        mask = node_output[cls_len:]
        #print(mask.size())
        pc_parent = mask[:,0,:]
        pos_score = self.masked_item_prediction(pc_parent,pc_positive).view(1,-1)
        neg_score = self.masked_item_prediction(pc_parent,pc_negetive).view(1,-1)
        scores = torch.cat((pos_score, neg_score), dim=0).permute(1, 0)

        pc_loss = pc_loss_fct(scores, torch.ones(pos_score.size(-1)).to(device).long())
        mask = mask[task1_labels]
        #cls = node_output[:,0,:].view(-1,self.config.hidden_size)
        #cls_logits = self.linear(cls)
        # cls_score = cls_logits[:,1]
        task1_loss = task1_loss_fct(positive,mask,torch.ones(len(positive)).to(device))
        # if torch.isnan(task1_loss):
        #     print("????????",positive,negetive)
        text_mlm_loss = loss_fct(word_logits, new_token_labels.view(-1))
        # node_mlm_loss = loss_fct(tag_logits, tag_labels.view(-1))
        print(text_mlm_loss,task1_loss.double(),pc_loss)
        return {
            'loss': text_mlm_loss+0*task1_loss+pc_loss,
            'text_mlm_loss': text_mlm_loss,
            # 'node_mlm_loss':node_mlm_loss,
            'task1_loss':task1_loss,
            'pc_loss':pc_loss,
            'output': output
        }


'''
    Task1:兄弟节点之间
    CLS tag tag1 text1 tag2   tag3    text2 text3 
    替换为
    CLS tag tag1 text1 tag2 tag_wrong text2 text3
    用CLS预测是否有替换
    Task2:父子节点之间
    Task3:文档level
    CLS为文档表示
    CLS，本文档中任意一个句子的embedding
    CLS，同一batch中其他文档中任意一个句子的embedding
'''

