import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import BertConfig, BertForMaskedLM, BertPreTrainedModel, BertModel
from models.FraBert import FraBert
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"
_TOKENIZER_FOR_DOC = "BertTokenizer"
BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using :class:`~transformers.BertTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.
            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`({0})`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in ``[0,
            1]``:
            - 0 corresponds to a `sentence A` token,
            - 1 corresponds to a `sentence B` token.
            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`({0})`, `optional`):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range ``[0,
            config.max_position_embeddings - 1]``.
            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in ``[0, 1]``:
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`({0}, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""

class HirachicalBert(BertForMaskedLM):
    config_class = BertConfig
    base_model_prefix = "bert"

    def __init__(self,config,layer_num,num_type=3):
        super().__init__(config)
        self.type_embedding = nn.Embedding(num_type,self.config.hidden_size)
        self.HBert = nn.ModuleList([BertModel(config,add_pooling_layer=False) for  i in range(layer_num)])
        #self.bert = FraBert(config=config)
        self.layer_num = layer_num
        self.apply(self._init_weights)

    # def get_input_embeddings(self):
    #     return self.embeddings.word_embeddings
    #
    # def set_input_embeddings(self, value):
    #     self.embeddings.word_embeddings = value
    #
    # def _prune_heads(self, heads_to_prune):
    #     """
    #     Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
    #     class PreTrainedModel
    #     """
    #     for layer, heads in heads_to_prune.items():
    #         self.encoder.layer[layer].attention.prune_heads(heads)
    #
    # @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    def forward(
            self,
            layer_index,
            waiting_mask,
            attention_mask = None,
            input_ids=None,
            inputs_type_idx=None,
            token_type_ids=None,
            labels=None,
    ):
        # print("layer_index:"+str(layer_index.size()))
        # print("waiting_mask:"+str(waiting_mask.size()))
        # print("input_ids:"+str(input_ids.size()))
        # print("inputs_type_idx:"+str(inputs_type_idx.size()))
        # print("token_type_ids:"+str(token_type_ids.size()))
        # print("labels:"+str(labels.size()))
       # print(attention_mask.size())
        word_logits = []
        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='mean')
        output = []
        batch_size,tag_len,seq_len = input_ids.size()
        bert_layer = 0
        for layer in range(self.layer_num):
            cur_layer = self.layer_num - layer
            if len(output) != 0:
                last_layer_output = output[-1]
            layer_mask = layer_index.eq(cur_layer)

            layer_labels = labels[layer_mask]
            if len(layer_labels):
                layer_waiting_mask = waiting_mask[layer_mask].type(torch.bool) #(batch_size,layer_tag_len,layer_seq_len)
                layer_token_type_ids = token_type_ids[layer_mask]
                layer_input_ids = input_ids[layer_mask]
                layer_inputs_embeds = self.bert.embeddings.word_embeddings(layer_input_ids) #(batch_size,layer_tag_len,layer_seq_len,hidden_dim)
                layer_inputs_embeds += self.type_embedding(inputs_type_idx[layer_mask])
                if len(output) != 0:
                    #print(layer_inputs_embeds.size())
                    # print(layer_index)
                    # print(cur_layer)
                    # print(layer_waiting_mask)
                   # print(last_layer_output.size())
                    layer_inputs_embeds[layer_waiting_mask] = last_layer_output
                layer_outputs = self.HBert[bert_layer](
                    input_ids = None,
                    attention_mask = None,
                    token_type_ids = layer_token_type_ids,
                    position_ids = None,
                    head_mask = None,
                    inputs_embeds = layer_inputs_embeds
                )
                layer_sequence_output = layer_outputs[0]
                #print(layer_sequence_output.size())
                if len(word_logits) :
                    word_logits = torch.cat((word_logits,self.cls(layer_sequence_output).view(-1,self.config.vocab_size)),dim = 0)
                else:
                    word_logits = self.cls(layer_sequence_output).view(-1,self.config.vocab_size)
                cls_output = layer_sequence_output[:,0,:].view(-1,self.config.hidden_size)
                output.append(cls_output)
                bert_layer += 1
            else:
               # print(layer_labels)
                continue
        labels = labels[layer_index.ne(0)]
        #print(word_logits.size(),labels.size())
        mlm_loss = loss_fct(word_logits, labels.view(-1))
        
            
        return {'loss': mlm_loss,
                'output':output
                } 
