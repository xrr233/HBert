import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

from transformers import BertConfig, BertForMaskedLM,BertModel,BertForPreTraining

class PageBert(BertForPreTraining):
    def __init__(self, config, list_length = 5):
        super().__init__(config)
        self.list_length = list_length

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        listwise_label=None,
        next_sentence_label=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        anchor_attention=None,
    ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        total_loss = None

        if labels is not None and next_sentence_label is not None:
            seq_relationship_logits = seq_relationship_score # bs, 2
            seq_relationship_scores = seq_relationship_logits[:,1] # bs, 1

            batch_size = seq_relationship_logits.size(0)
            softmax = nn.Softmax(dim=1)
            log_softmax= nn.LogSoftmax(dim=1)
            kl_loss = torch.nn.KLDivLoss(reduction='mean')
            logits = seq_relationship_scores.reshape(batch_size//self.list_length, self.list_length) # bs/2, 2
            logits = logits.masked_fill(mask=anchor_attention,value=torch.tensor(-1e9))
            logits = torch.nn.log_softmax(logits)
            
            if listwise_label is None:
                listwise_label = torch.Tensor([self.list_length-(i+1) for i in range(self.list_length)]).long().repeat(batch_size//self.list_length,1)
            listwise_label = listwise_label.masked_fill(mask=anchor_attention,value=torch.tensor(-1e9))
            listwise_label_logits = softmax(listwise_label)

            listwise_loss = kl_loss(logits,listwise_label_logits)
            

            # pos_logits = logits[:, 0] #bs/2, 1
            # neg_logits = logits[:, 1] # bs/2, 1
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

            total_loss = listwise_loss + masked_lm_loss
            
        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output        

        return BertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
