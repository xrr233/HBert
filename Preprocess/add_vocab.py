from transformers import BertTokenizer, BertModel
BertPath = '/home/yu_guo/huggingface_transformers/examples/pytorch/language-modeling/bert_base_uncased'
tag_vocab_path = '/home/yu_guo/DataPreProcess/data/tag.vocab'

def load_vocab(path):
    vocab = []
    with open(path,'r')as f:
        for one_tag in f:
            vocab.append(one_tag.strip())
    return vocab
# model
bert_tokenizer = BertTokenizer.from_pretrained(BertPath)
bert_model = BertModel.from_pretrained(BertPath)

ADDITIONAL_SPECIAL_TOKENS = load_vocab(tag_vocab_path)
print(ADDITIONAL_SPECIAL_TOKENS)
bert_tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})

#bert_tokenizer.add_special_tokens({"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS})
bert_model.resize_token_embeddings(len(bert_tokenizer))
bert_param_ids = list(map(id, bert_model.parameters()))
print(bert_tokenizer.vocab['<div>'])