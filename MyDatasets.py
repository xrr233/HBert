import random
from dataclasses import dataclass
import math
import datasets
from typing import Union, List, Tuple, Dict
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import Dataset

# from .arguments import DataArguments, RerankerTrainingArguments
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers import DataCollatorWithPadding
import numpy as np
from tqdm import tqdm
import os

import random
from dataclasses import dataclass
import math
import datasets
from typing import Union, List, Tuple, Dict
from torch.nn.utils.rnn import pad_sequence
import torch
from torch.utils.data import Dataset

# from .arguments import DataArguments, RerankerTrainingArguments
from transformers import PreTrainedTokenizer, BatchEncoding
from transformers import DataCollatorWithPadding
import numpy as np
from tqdm import tqdm
import os

class HBERTPretrainedPointWiseDataset(Dataset):
    def __init__(
            self,
            args,
            tokenizer: PreTrainedTokenizer,
            dataset_cache_dir,
            dataset_script_dir,
            max_seq_len,
    ):
        self.max_seq_len = max_seq_len
        train_file = args.train_file
        if os.path.isdir(train_file):
            filenames = os.listdir(train_file)
            train_files = [os.path.join(train_file, fn) for fn in filenames]
        else:
            train_files = train_file
        block_size_10MB = 10<<20
        print("start loading datasets, train_files: ", train_files)
        print(dataset_script_dir)
        self.nlp_dataset = datasets.load_dataset(
            f'{dataset_script_dir}/json.py',
            data_files=train_files,
            ignore_verifications=False,
            cache_dir=dataset_cache_dir,
            features=datasets.Features({
                "node_texts_idx":[datasets.Value("int32")],
                "text_tokens_idx":[datasets.Value("int32")],
                "node_tokens_idx":[datasets.Value("int32")],
                "inputs_type_idx":[datasets.Value("int32")],
                "text_labels":[datasets.Value("int32")],
                "node_text_labels":[datasets.Value("int32")],
                "text_layer_index":[datasets.Value("int32")],
                "node_layer_index":[datasets.Value("int32")],
                "text_num":[datasets.Value("int32")],
                "node_num":[datasets.Value("int32")],
                "waiting_mask":[datasets.Value("int32")],
                "position":[datasets.Value("int32")],
            }),
            block_size = block_size_10MB
        )['train']

        
        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)

        print("loading dataset ok! len of dataset,", self.total_len)
    
    def __len__(self):
        return self.total_len

    
    def __getitem__(self, item):
        data = self.nlp_dataset[item]
        #token_type_ids = np.array([0]*len(data['tokens_idx']))
        #waiting_mask = data['waiting_mask']
        max_seq_len = self.max_seq_len
        #max_tag_len = 32
        #waiting_mask = torch.BoolTensor(waiting_mask).view(-1,max_seq_len)
        #tag_len,_ = waiting_mask.size()
        #data['layer_index'] = data['layer_index']+(max_tag_len-len(data['layer_index']))*[-1]
        # waiting_mask = torch.cat((torch.zeros(tag_len,1),waiting_mask),dim=1)[:,:max_seq_len].contiguous().view(-1)
        # data = {
        #     "input_ids": list(data['tokens_idx']),
        #     "token_type_ids": list(token_type_ids),
        #     "inputs_type_idx":list(data['type_idx']),
        #     "labels": list(data['labels']),
        #     "layer_index":torch.LongTensor(data['layer_index']),
        #     "waiting_mask":waiting_mask,
        # }
        text_num = data['text_num']
        node_num = data['node_num']
        position_len = max([text_num[i]+node_num[i] for i in range(len(text_num))])

        data = {
                "node_text_input_ids":torch.LongTensor(data['node_texts_idx']),
                "token_input_ids":torch.LongTensor(data['text_tokens_idx']),
                "node_input_ids":torch.LongTensor(data['node_tokens_idx']),
                "inputs_type_idx":torch.LongTensor(data['inputs_type_idx']),
                "token_labels":torch.LongTensor(data['text_labels']),
                "node_text_labels":torch.LongTensor(data['node_text_labels']),
                "token_layer_index":torch.LongTensor(data['text_layer_index']),
                "node_layer_index":torch.LongTensor(data['node_layer_index']),
                "seq_num":list(data['text_num']),
                "node_num":list(data['node_num']),
                "waiting_mask":torch.LongTensor(data['waiting_mask']),
                "position":list(data['position']),
                "position_len":position_len,
        }
        return BatchEncoding(data)

@dataclass
class HBERTPointCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    def __call__(
            self, features
    ):
        
        max_seq_len = 256
        max_node_len = 10
        # print(features)
        batch_size = len(features)
        mlm_labels = []
        inputs_type_idx = []
        layer_index = []
        waiting_mask = []
        position = []
        layer_num = len(features[0]['seq_num'])
        token_input_ids = []
        node_texts_idx = []
        node_input_ids = []
        inputs_type_idx = []
        token_labels= []
        node_text_labels= []
        token_layer_index= []
        node_layer_index= []
        seq_num= []
        node_num= []
        waiting_mask= []
        for i in range(batch_size):
            one_position = features[i]['position']
            position_len = features[i]['position_len']
            one_position = [torch.LongTensor(one_position[j:j+position_len]) for j in range(0, len(one_position),position_len )]
            assert len(one_position) == layer_num
            position.extend(one_position)
            token_input_ids.append(features[i]['token_input_ids'])
            node_input_ids.append(features[i]['node_input_ids'])
            inputs_type_idx.append(features[i]['inputs_type_idx'])
            token_labels.append(features[i]['token_labels'])
            token_layer_index.append(features[i]['token_layer_index'])
            node_layer_index.append(features[i]['node_layer_index'])
            seq_num.append(features[i]['seq_num'])
            node_num.append(features[i]['node_num'])
            waiting_mask.append(features[i]['waiting_mask'])
            node_text_labels.append(features[i]['node_text_labels'])
            node_texts_idx.append(features[i]['node_texts_idx'])
            del features[i]['token_input_ids']
            del features[i]['node_input_ids']
            del features[i]['inputs_type_idx']
            del features[i]['token_labels']
            del features[i]['node_text_labels']
            del features[i]['token_layer_index']
            del features[i]['node_layer_index']
            del features[i]['seq_num']
            del features[i]['node_num']
            del features[i]['waiting_mask']
            del features[i]['position']
            del features[i]['position_len']
            del features[i]['node_texts_idx']
        features = {}
        features['node_texts_idx'] = pad_sequence(node_texts_idx,batch_size=True).view(batch_size,-1,max_seq_len)
        features["token_input_ids"] = pad_sequence(token_input_ids, batch_first=True).view(batch_size,-1,max_seq_len)
        features["node_input_ids"] = pad_sequence(node_input_ids, batch_first=True).view(batch_size,-1,max_node_len)
        features["inputs_type_idx"] = pad_sequence(inputs_type_idx, batch_first=True).view(batch_size,-1,max_seq_len)
        features["token_labels"] = pad_sequence(token_labels, batch_first=True).view(batch_size,-1,max_seq_len)
        features["node_text_labels"] = pad_sequence(node_text_labels, batch_first=True).view(batch_size,-1,max_seq_len)
        features["token_layer_index"] = pad_sequence(token_layer_index, batch_first=True)
        features["node_layer_index"] = pad_sequence(node_layer_index, batch_first=True)
        features["seq_num"] = torch.LongTensor(seq_num)
        features["node_num"] = torch.LongTensor(node_num)
        features["waiting_mask"] = pad_sequence(waiting_mask, batch_first=True).view(batch_size,-1,max_node_len)
        features["position"] = pad_sequence(position, batch_first=True).view(batch_size,layer_num,-1)

        # features['input_ids'] = features['input_ids'].view(batch_size,-1,max_seq_len)
        # features['token_type_ids']=features['token_type_ids'].view(batch_size,-1,max_seq_len)
        # features['labels'] = torch.LongTensor(mlm_labels_matrix).view(batch_size,-1,max_seq_len)
        # features['inputs_type_idx'] = torch.LongTensor(inputs_type_idx_matrix).view(batch_size,-1,max_seq_len)
        # features['layer_index'] = pad_sequence(layer_index).transpose(0,1)
        # print(features['layer_index'].size())
        # features['waiting_mask'] = pad_sequence(waiting_mask).transpose(0,1).view(batch_size,-1,max_seq_len)
        
        # print("gen features")
        # print("---------------------------")
        # print(features)
        # print("#####################")
        return features
# class HBERTPretrainedPointWiseDataset(Dataset):
#     def __init__(
#             self,
#             args,
#             tokenizer: PreTrainedTokenizer,
#             dataset_cache_dir,
#             dataset_script_dir,
#             max_seq_len,
#     ):
#         self.max_seq_len = max_seq_len
#         train_file = args.train_file
#         if os.path.isdir(train_file):
#             filenames = os.listdir(train_file)
#             train_files = [os.path.join(train_file, fn) for fn in filenames]
#         else:
#             train_files = train_file
#         block_size_10MB = 10<<20
#         print("start loading datasets, train_files: ", train_files)
#         print(dataset_script_dir)
#         self.nlp_dataset = datasets.load_dataset(
#             f'{dataset_script_dir}/json.py',
#             data_files=train_files,
#             ignore_verifications=False,
#             cache_dir=dataset_cache_dir,
#             features=datasets.Features({

#                 "text_tokens_idx":[datasets.Value("int32")],
#                 "node_tokens_idx":[datasets.Value("int32")],
#                 "inputs_type_idx":[datasets.Value("int32")],
#                 "text_labels":[datasets.Value("int32")],
#                 "node_labels":[datasets.Value("int32")],
#                 "text_layer_index":[datasets.Value("int32")],
#                 "node_layer_index":[datasets.Value("int32")],
#                 "text_num":[datasets.Value("int32")],
#                 "node_num":[datasets.Value("int32")],
#                 "waiting_mask":[datasets.Value("int32")],
#                 "position":[datasets.Value("int32")],
#             }),
#             block_size = block_size_10MB
#         )['train']

        
#         self.tok = tokenizer
#         self.SEP = [self.tok.sep_token_id]
#         self.args = args
#         self.total_len = len(self.nlp_dataset)

#         print("loading dataset ok! len of dataset,", self.total_len)
    
#     def __len__(self):
#         return self.total_len

    
#     def __getitem__(self, item):
#         data = self.nlp_dataset[item]
#         #token_type_ids = np.array([0]*len(data['tokens_idx']))
#         #waiting_mask = data['waiting_mask']
#         max_seq_len = self.max_seq_len
#         #max_tag_len = 32
#         #waiting_mask = torch.BoolTensor(waiting_mask).view(-1,max_seq_len)
#         #tag_len,_ = waiting_mask.size()
#         #data['layer_index'] = data['layer_index']+(max_tag_len-len(data['layer_index']))*[-1]
#         # waiting_mask = torch.cat((torch.zeros(tag_len,1),waiting_mask),dim=1)[:,:max_seq_len].contiguous().view(-1)
#         # data = {
#         #     "input_ids": list(data['tokens_idx']),
#         #     "token_type_ids": list(token_type_ids),
#         #     "inputs_type_idx":list(data['type_idx']),
#         #     "labels": list(data['labels']),
#         #     "layer_index":torch.LongTensor(data['layer_index']),
#         #     "waiting_mask":waiting_mask,
#         # }
#         text_num = data['text_num']
#         node_num = data['node_num']
#         position_len = max([text_num[i]+node_num[i] for i in range(len(text_num))])

#         data = {
#                 "token_input_ids":torch.LongTensor(data['text_tokens_idx']),
#                 "node_input_ids":torch.LongTensor(data['node_tokens_idx']),
#                 "inputs_type_idx":torch.LongTensor(data['inputs_type_idx']),
#                 "token_labels":torch.LongTensor(data['text_labels']),
#                 "node_labels":torch.LongTensor(data['node_labels']),
#                 "token_layer_index":torch.LongTensor(data['text_layer_index']),
#                 "node_layer_index":torch.LongTensor(data['node_layer_index']),
#                 "seq_num":list(data['text_num']),
#                 "node_num":list(data['node_num']),
#                 "waiting_mask":torch.LongTensor(data['waiting_mask']),
#                 "position":list(data['position']),
#                 "position_len":position_len,
#         }
#         return BatchEncoding(data)

# @dataclass
# class HBERTPointCollator(DataCollatorWithPadding):
#     """
#     Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
#     and pass batch separately to the actual collator.
#     Abstract out data detail for the model.
#     """

#     def __call__(
#             self, features
#     ):
        
#         max_seq_len = 256
#         max_node_len = 10
#         # print(features)
#         batch_size = len(features)
#         mlm_labels = []
#         inputs_type_idx = []
#         layer_index = []
#         waiting_mask = []
#         position = []
#         layer_num = len(features[0]['seq_num'])
#         token_input_ids = []
#         node_input_ids = []
#         inputs_type_idx = []
#         token_labels= []
#         node_labels= []
#         token_layer_index= []
#         node_layer_index= []
#         seq_num= []
#         node_num= []
#         waiting_mask= []
#         for i in range(batch_size):
#             one_position = features[i]['position']
#             position_len = features[i]['position_len']
#             one_position = [torch.LongTensor(one_position[j:j+position_len]) for j in range(0, len(one_position),position_len )]
#             assert len(one_position) == layer_num
#             position.extend(one_position)
#             token_input_ids.append(features[i]['token_input_ids'])
#             node_input_ids.append(features[i]['node_input_ids'])
#             inputs_type_idx.append(features[i]['inputs_type_idx'])
#             token_labels.append(features[i]['token_labels'])
#             token_layer_index.append(features[i]['token_layer_index'])
#             node_layer_index.append(features[i]['node_layer_index'])
#             seq_num.append(features[i]['seq_num'])
#             node_num.append(features[i]['node_num'])
#             waiting_mask.append(features[i]['waiting_mask'])
#             node_labels.append(features[i]['node_labels'])

#             del features[i]['token_input_ids']
#             del features[i]['node_input_ids']
#             del features[i]['inputs_type_idx']
#             del features[i]['token_labels']
#             del features[i]['node_labels']
#             del features[i]['token_layer_index']
#             del features[i]['node_layer_index']
#             del features[i]['seq_num']
#             del features[i]['node_num']
#             del features[i]['waiting_mask']
#             del features[i]['position']
#             del features[i]['position_len']
#         features = {}
#         features["token_input_ids"] = pad_sequence(token_input_ids, batch_first=True).view(batch_size,-1,max_seq_len)
#         features["node_input_ids"] = pad_sequence(node_input_ids, batch_first=True).view(batch_size,-1,max_node_len)
#         features["inputs_type_idx"] = pad_sequence(inputs_type_idx, batch_first=True).view(batch_size,-1,max_seq_len)
#         features["token_labels"] = pad_sequence(token_labels, batch_first=True).view(batch_size,-1,max_seq_len)
#         features["node_labels"] = pad_sequence(node_labels, batch_first=True).view(batch_size,-1,max_node_len)
#         features["token_layer_index"] = pad_sequence(token_layer_index, batch_first=True)
#         features["node_layer_index"] = pad_sequence(node_layer_index, batch_first=True)
#         features["seq_num"] = torch.LongTensor(seq_num)
#         features["node_num"] = torch.LongTensor(node_num)
#         features["waiting_mask"] = pad_sequence(waiting_mask, batch_first=True).view(batch_size,-1,max_node_len)
#         features["position"] = pad_sequence(position, batch_first=True).view(batch_size,layer_num,-1)

#         # features['input_ids'] = features['input_ids'].view(batch_size,-1,max_seq_len)
#         # features['token_type_ids']=features['token_type_ids'].view(batch_size,-1,max_seq_len)
#         # features['labels'] = torch.LongTensor(mlm_labels_matrix).view(batch_size,-1,max_seq_len)
#         # features['inputs_type_idx'] = torch.LongTensor(inputs_type_idx_matrix).view(batch_size,-1,max_seq_len)
#         # features['layer_index'] = pad_sequence(layer_index).transpose(0,1)
#         # print(features['layer_index'].size())
#         # features['waiting_mask'] = pad_sequence(waiting_mask).transpose(0,1).view(batch_size,-1,max_seq_len)
        
#         # print("gen features")
#         # print("---------------------------")
#         # print(features)
#         # print("#####################")
#         return features

@dataclass
class HirachicalBERTPointCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    def __call__(
            self, features
    ):
        
        max_seq_len = 32
        # print(features)
        batch_size = len(features)
        mlm_labels = []
        inputs_type_idx = []
        layer_index = []
        waiting_mask = []
        for i in range(batch_size):
            mlm_labels.append(features[i]['labels'])
            inputs_type_idx.append(features[i]['inputs_type_idx'])
            layer_index.append(features[i]['layer_index'])
            waiting_mask.append(features[i]['waiting_mask'])
            del features[i]['labels']
            del features[i]['inputs_type_idx']
            del features[i]['layer_index']
            del features[i]['waiting_mask']
        # print("++++++++++++++++++++")
        #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        features = super().__call__(features)
        max_len = features['input_ids'].size()[1]
        # print("max_len", max_len)
        mlm_labels_matrix = np.ones([batch_size, max_len]) * -100
        inputs_type_idx_matrix = np.zeros([batch_size,max_len])
        for i in range(batch_size):
            mlm_labels_matrix[i][:len(mlm_labels[i])] = mlm_labels[i]
            inputs_type_idx_matrix[i][:len(inputs_type_idx[i])]=inputs_type_idx[i]
        features['input_ids'] = features['input_ids'].view(batch_size,-1,max_seq_len)
        features['token_type_ids']=features['token_type_ids'].view(batch_size,-1,max_seq_len)
        features['labels'] = torch.LongTensor(mlm_labels_matrix).view(batch_size,-1,max_seq_len)
        features['inputs_type_idx'] = torch.LongTensor(inputs_type_idx_matrix).view(batch_size,-1,max_seq_len)
        features['layer_index'] = pad_sequence(layer_index).transpose(0,1)
        print(features['layer_index'].size())
        features['waiting_mask'] = pad_sequence(waiting_mask).transpose(0,1).view(batch_size,-1,max_seq_len)
        
        # print("gen features")
        # print("---------------------------")
        # print(features)
        # print("#####################")
        return features

class HirachicalBERTPretrainedPointWiseDataset(Dataset):
    def __init__(
            self,
            args,
            tokenizer: PreTrainedTokenizer,
            dataset_cache_dir,
            dataset_script_dir,
    ):
        train_file = args.train_file
        if os.path.isdir(train_file):
            filenames = os.listdir(train_file)
            train_files = [os.path.join(train_file, fn) for fn in filenames]
        else:
            train_files = train_file
        block_size_10MB = 10<<20
        print("start loading datasets, train_files: ", train_files)
        print(dataset_script_dir)
        self.nlp_dataset = datasets.load_dataset(
            f'{dataset_script_dir}/json.py',
            data_files=train_files,
            ignore_verifications=False,
            cache_dir=dataset_cache_dir,
            features=datasets.Features({
                "tokens_idx":[datasets.Value("int32")],
                "type_idx":[datasets.Value("int32")],
                "labels":[datasets.Value("int32")],
                "layer_index":[datasets.Value("int32")],
                "waiting_mask":[datasets.Value("int32")],
            }),
            block_size = block_size_10MB
        )['train']

        
        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)

        print("loading dataset ok! len of dataset,", self.total_len)
    
    def __len__(self):
        return self.total_len

    
    def __getitem__(self, item):
        data = self.nlp_dataset[item]
        token_type_ids = np.array([0]*len(data['tokens_idx']))
        waiting_mask = data['waiting_mask']
        max_seq_len = 32
        #max_tag_len = 32
        waiting_mask = torch.BoolTensor(waiting_mask).view(-1,max_seq_len)
        tag_len,_ = waiting_mask.size()
        #data['layer_index'] = data['layer_index']+(max_tag_len-len(data['layer_index']))*[-1]
        waiting_mask = torch.cat((torch.zeros(tag_len,1),waiting_mask),dim=1)[:,:max_seq_len].contiguous().view(-1)
        data = {
            "input_ids": list(data['tokens_idx']),
            "token_type_ids": list(token_type_ids),
            "inputs_type_idx":list(data['type_idx']),
            "labels": list(data['labels']),
            "layer_index":torch.LongTensor(data['layer_index']),
            "waiting_mask":waiting_mask,
        }
        return BatchEncoding(data)

class BERTGATPretrainedPointWiseDataset(Dataset):
    def __init__(
            self,
            args,
            tokenizer: PreTrainedTokenizer,
            dataset_cache_dir,
            dataset_script_dir,
    ):
        train_file = args.train_file
        if os.path.isdir(train_file):
            filenames = os.listdir(train_file)
            train_files = [os.path.join(train_file, fn) for fn in filenames]
        else:
            train_files = train_file
        block_size_10MB = 10<<20
        print("start loading datasets, train_files: ", train_files)
        print(dataset_script_dir)
        self.nlp_dataset = datasets.load_dataset(
            f'{dataset_script_dir}/json.py',
            data_files=train_files,
            ignore_verifications=False,
            cache_dir=dataset_cache_dir,
            features=datasets.Features({
                "segment_ids": [datasets.Value("int32")],
                "tokens_idx":[datasets.Value("int32")],
                "type_idx":[datasets.Value("int32")],
                "labels":[datasets.Value("int32")],
                "attention_mask":[datasets.Value("int32")],
                "mask_tag_idx":[datasets.Value("int32")],
            }),
            block_size = block_size_10MB
        )['train']

        
        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)

        print("loading dataset ok! len of dataset,", self.total_len)
    
    def __len__(self):
        return self.total_len

    
    def __getitem__(self, item):
        data = self.nlp_dataset[item]
        mask_tag_idx = data['mask_tag_idx']
        above_mask = np.array([0]*int(len(data['tokens_idx'])/128))
        above_mask[mask_tag_idx]=1
        above_mask = np.insert(above_mask,0,[0])
        token_type_ids = np.array([0]*len(data['tokens_idx']))
        data = {
            "input_ids": list(data['tokens_idx']),
            "token_type_ids": list(token_type_ids),
            "inputs_type_idx":list(data['type_idx']),
            "labels": list(data['labels']),
            "attention_mask":list(data['attention_mask']),
            "above_mask_idx":torch.BoolTensor(above_mask),
        }
        return BatchEncoding(data)

class BERTPretrainedPointWiseDataset(Dataset):

    def __init__(
            self,
            args,
            tokenizer: PreTrainedTokenizer,
            dataset_cache_dir,
            dataset_script_dir,
    ):
        train_file = args.train_file
        if os.path.isdir(train_file):
            filenames = os.listdir(train_file)
            train_files = [os.path.join(train_file, fn) for fn in filenames]
        else:
            train_files = train_file
        
        print("start loading datasets, train_files: ", train_files)
            
        self.nlp_dataset = datasets.load_dataset(
            f'{dataset_script_dir}/json.py',
            data_files=train_files,
            ignore_verifications=False,
            cache_dir=dataset_cache_dir,
            features=datasets.Features({
                #"masked_lm_labels": [datasets.Value("string")],
                "masked_lm_positions": [datasets.Value("int32")],
                "segment_ids": [datasets.Value("int32")],
                #"tokens":[datasets.Value("string")],
                "tokens_idx":[datasets.Value("int32")],
                "type_idx":[datasets.Value("int32")],
                "masked_lm_labels_idxs":[datasets.Value("int32")],
            })
        )['train']

        
        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)

        print("loading dataset ok! len of dataset,", self.total_len)
    
    def __len__(self):
        return self.total_len

    
    def __getitem__(self, item):
        data = self.nlp_dataset[item]

        labels = np.array([-100] * len(data['tokens_idx']))
        masked_lm_positions = data['masked_lm_positions']

        masked_lm_labels = data['masked_lm_labels_idxs']
        type_idx = data['type_idx']
        labels[masked_lm_positions] = masked_lm_labels
        token_type_ids = np.array([0]*len(data['tokens_idx']))
        data = {
            "input_ids": list(data['tokens_idx'])[:512],
            "token_type_ids": list(token_type_ids)[:512],
           # "inputs_type_idx":list(data['type_idx']),
            "labels": list(labels)[:512]
        }

        return BatchEncoding(data)

class BERTPretrainedPairWiseDataset(Dataset):
    def __init__(
            self,
            args,
            tokenizer: PreTrainedTokenizer,
            dataset_cache_dir,
            dataset_script_dir,
    ):
        train_file = args.train_file
        if os.path.isdir(train_file):
            filenames = os.listdir(train_file)
            train_files = [os.path.join(train_file, fn) for fn in filenames]
        else:
            train_files = train_file
        
        print("start loading datasets, train_files: ", train_files)
        mymydatasets = []
        for i in range(10):
            print("start loading dataset", i)
            nlp_dataset = datasets.load_dataset(
                f'{dataset_script_dir}/json.py',
                data_files=train_files,
                ignore_verifications=False,
                cache_dir=dataset_cache_dir + str(i),
                features=datasets.Features({
                    "pos":{
                        'label': datasets.Value("int32"),
                        "masked_lm_positions": [datasets.Value("int32")],
                        "segment_ids": [datasets.Value("int32")],
                        "tokens_idx":[datasets.Value("int32")],
                        "masked_lm_labels_idxs":[datasets.Value("int32")],
                    },
                    "neg":{
                        'label': datasets.Value("int32"),
                        "masked_lm_positions": [datasets.Value("int32")],
                        "segment_ids": [datasets.Value("int32")],
                        "tokens_idx":[datasets.Value("int32")],
                        "masked_lm_labels_idxs":[datasets.Value("int32")],                    
                    }
                })
            )['train']
            mymydatasets.append(nlp_dataset)

        # nlp_dataset = nlp_dataset.shuffle(2021)
        # nlp_dataset = nlp_dataset[:args.limit]
        self.nlp_dataset = nlp_dataset
        
        self.tok = tokenizer
        self.SEP = [self.tok.sep_token_id]
        self.args = args
        self.total_len = len(self.nlp_dataset)    

        print("loading dataset ok! len of dataset,", self.total_len)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item):
        pairdata = self.nlp_dataset[item]
        examples = [pairdata['pos'], pairdata['neg']]
        group_batch = []

        for e in examples:
            labels = np.array([-100] * len(e['tokens_idx']))
            masked_lm_positions = e['masked_lm_positions']
            masked_lm_labels = e['masked_lm_labels_idxs']
            labels[masked_lm_positions] = masked_lm_labels

            data = {
                "input_ids": list(e['tokens_idx']),
                "token_type_ids": list(e['segment_ids']),
                "labels": list(labels),
                "next_sentence_label": e['label']
            }

            group_batch.append(BatchEncoding(data))
        
        return group_batch
@dataclass
class PointCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    def __call__(
            self, features
    ):
        
        # print(features)
        batch_size = len(features)
        mlm_labels = []
        #inputs_type_idx = []
        for i in range(batch_size):
            mlm_labels.append(features[i]['labels'])
            #inputs_type_idx.append(features[i]['inputs_type_idx'])
            del features[i]['labels']
           # del features[i]['inputs_type_idx']

        # print("++++++++++++++++++++")
        #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        features = super().__call__(features)
        max_len = features['input_ids'].size()[1]
        # print("max_len", max_len)
        mlm_labels_matrix = np.ones([batch_size, max_len]) * -100
        inputs_type_idx_matrix = np.zeros([batch_size,max_len])
        
        for i in range(batch_size):
            mlm_labels_matrix[i][:len(mlm_labels[i])] = mlm_labels[i]
            #inputs_type_idx_matrix[i][:len(inputs_type_idx[i])]=inputs_type_idx[i]
        features['labels'] = torch.LongTensor(mlm_labels_matrix)

        #features['inputs_type_idx'] = torch.LongTensor(inputs_type_idx_matrix)
        # print("gen features")
        # print("---------------------------")
        # print(features)
        # print("#####################")
        return features
@dataclass
class PairCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    def __call__(
            self, features
    ):
        
        # print(features)
        features_flattened = []
        for f in features:
            features_flattened += [f[0], f[1]]

        features = features_flattened
        batch_size = len(features)
        mlm_labels = []
        

        for i in range(batch_size):
            mlm_labels.append(features[i]['labels'])
            del features[i]['labels']

        # print("++++++++++++++++++++")
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        features = super().__call__(features)
        max_len = features['input_ids'].size()[1]
        # print("max_len", max_len)
        mlm_labels_matrix = np.ones([batch_size, max_len]) * -100
        for i in range(batch_size):
            mlm_labels_matrix[i][:len(mlm_labels[i])] = mlm_labels[i]
        features['labels'] = torch.LongTensor(mlm_labels_matrix)
        # print("gen features")
        # print("---------------------------")
        # print(features)
        # print("#####################")
        # print(features)
        return features


@dataclass
class BERTGATPointCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """

    def __call__(
            self, features
    ):
        
        max_seq_len = 128
        max_tag_len = 32
        # print(features)
        batch_size = len(features)
        mlm_labels = []
        inputs_type_idx = []
        attention_mask = []
        above_mask_idx = []
        for i in range(batch_size):
            mlm_labels.append(features[i]['labels'])
            inputs_type_idx.append(features[i]['inputs_type_idx'])
            temp = features[i]['attention_mask']
            temp = [temp[i:i+max_tag_len] for i in range(0,len(temp),max_tag_len)]
            temp = [([0]+temp[i]) for i in range(len(temp))]
            temp = [[1]*(max_tag_len+1)] + temp
            attention_mask.append(temp)
            above_mask_idx.append(features[i]['above_mask_idx'])
            del features[i]['labels']
            del features[i]['inputs_type_idx']
            del features[i]['attention_mask']
            del features[i]['above_mask_idx']
        # print("++++++++++++++++++++")
        #print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        features = super().__call__(features)
        max_len = features['input_ids'].size()[1]
        # print("max_len", max_len)
        mlm_labels_matrix = np.ones([batch_size, max_len]) * -100
        inputs_type_idx_matrix = np.zeros([batch_size,max_len])
        for i in range(batch_size):
            mlm_labels_matrix[i][:len(mlm_labels[i])] = mlm_labels[i]
            inputs_type_idx_matrix[i][:len(inputs_type_idx[i])]=inputs_type_idx[i]
        features['input_ids'] = features['input_ids'].view(batch_size,-1,max_seq_len)
        features['token_type_ids']=features['token_type_ids'].view(batch_size,-1,max_seq_len)
        features['labels'] = torch.LongTensor(mlm_labels_matrix).view(batch_size,-1,max_seq_len)
        features['inputs_type_idx'] = torch.LongTensor(inputs_type_idx_matrix).view(batch_size,-1,max_seq_len)
        features['attention_mask'] = torch.LongTensor(attention_mask)
        features['above_mask_idx'] = torch.nn.utils.rnn.pad_sequence(above_mask_idx,padding_value=0).transpose(0,1)
        
        # print("gen features")
        # print("---------------------------")
        # print(features)
        # print("#####################")
        return features