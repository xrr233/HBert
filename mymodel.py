import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from pretrain.large_emb import LargeEmbedding

from transformers import RobertaConfig, RobertaForMaskedLM, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP,BertConfig,BertForMaskedLM,BERT_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.modeling_bert import BertLayerNorm, gelu


class FraBert(RobertaForMaskedLM):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"
    
    def __init__(self,config,num_tag):
        super.__init__(config)
        self.tag_embedding = nn.Embedding(num_tag,config.hidden_size,padding_idx=1)
        self.tag_lm_head = TagLMHead(config, num_tag)
        self.apply(self._init_weights)

    def extend_embedding(self,token_type=3):
        self.roberta.embeddings.token_type_embeddings = nn.Embedding(token_type,self.config.hidden_size,_weight=torch.zeros(
                                                                         (token_type, self.config.hidden_size)))

    def tie_tag_weights(self):
        self.tag_lm_head.decoder.weight = self.tag_embeddings.weight
        if getattr(self.tag_lm_head.decoder, "bias", None) is not None:
            self.tag_lm_head.decoder.bias.data = torch.nn.functional.pad(
                self.tag_lm_head.decoder.bias.data,
                (0, self.tag_lm_head.decoder.weight.shape[0] - self.tag_lm_head.decoder.bias.shape[0],),
                "constant",
                0,
            )       

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            tag_masked_lm_labels=None, 
            n_word_nodes=None, 
            ent_index=None 
    ):
        n_word_nodes = n_word_nodes[0]
        
        word_embeddings = self.roberta.embeddings.word_embeddings(
            input_ids[:, :n_word_nodes])  # batch x n_word_nodes x hidden_size

        tag_embeddings = self.tag_embeddings(
            input_ids[:, n_word_nodes:])

        inputs_embeds = torch.cat([word_embeddings,tag_embeddings],
                                  dim=1)  # batch x seq_len x hidden_size

        outputs = self.roberta(
            input_ids=None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]  # batch x seq_len x hidden_size

        loss_fct = CrossEntropyLoss(ignore_index=-1, reduction='mean')
        word_logits = self.lm_head(sequence_output[:, :n_word_nodes, :])
        word_predict = torch.argmax(word_logits, dim=-1)
        masked_lm_loss = loss_fct(word_logits.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))

        tag_logits = self.tag_lm_head(sequence_output[:, n_word_nodes:, :])
        tag_predict = torch.argmax(tag_logits, dim=-1)
        tag_masked_lm_loss = loss_fct(tag_logits.view(-1, tag_logits.size(-1)), tag_masked_lm_labels.view(-1))
        loss = masked_lm_loss + tag_masked_lm_loss
        return {'loss': loss,
                'word_pred': word_predict,
                'tag_pred': tag_predict}                                                                                           



class TagLMHead(nn.Module):
    def __init__(self, config, num_tag):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, num_tag, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_tag), requires_grad=True)
        # self.dropout = nn.Dropout(p=dropout)

        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        # x = self.dropout(x)
        x = self.layer_norm(x)

        x = self.decoder(x)

        return x