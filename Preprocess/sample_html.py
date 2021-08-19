import os
from argparse import ArgumentParser
import sys
sys.path.append('./')
from tqdm import tqdm
import json
from transformers import BertTokenizer
import numpy as np
import random
from random import  shuffle, choice, sample
import collections
import traceback
from multiprocessing import Pool, Value, Lock
from tempfile import TemporaryDirectory
from pathlib import Path
import shelve
from collections import Counter
import math
import copy
bert_tokenizer = BertTokenizer.from_pretrained('/home/yu_guo/bert/pretrained_model', do_lower_case=True)

def get_text(html,i):
    depth = 0
    word = []
    type_idx = []
    tag_name = html[i]["name"]
    tag_children = html[i]["children"]
    tag_text = html[i]["text"]
    tag_idx = html[i]["id"]
    if tag_name == "textnode":
        res = bert_tokenizer.tokenize(tag_text)
        return res,[0]*len(res),depth
    else:
        for child_idx in tag_children:
            inner_word,inner_type_idx,tag_depth = get_text(html,int(child_idx))
            word += inner_word
            type_idx += inner_type_idx
            depth = max(depth,tag_depth)
        depth += 1
        assert len(type_idx)==len(word)
        return ["<"+tag_name+">"]+word+["<"+tag_name+">"],[1]+type_idx+[2],depth

def get_train_set(line,min_length = 10,max_length=512):
    text = []
    type_idx = []
    index = {}
    text,type_idx = get_html_text(line,min_length,max_length)
    return text,type_idx

def cut_text(line):
    text = []
    input_ids=[]
    all_tokens,input_type_ids,_ = get_text(line,0)
    while(len(all_tokens)>500):
        one_text = all_tokens[:500]
        input_id = input_type_ids[:500]
        all_tokens = all_tokens[500:]
        input_type_ids=input_type_ids[500:]
        text.append(one_text)
        input_ids.append(input_id)
    text.append(all_tokens)
    input_ids.append(input_type_ids)
    return text,input_ids
def get_html_text(line,index,depth,min_length = 10,max_length=512):

#    line = json.loads(line)
    res_text = []
    candidates = {}
    res_type_idx = []
    tag_num = len(line)
    for i in range(tag_num):
        if i in candidates:
            continue
        text,type_idx,depth = get_text(line,i)
        if len(text) > max_length or len(text)<min_length:
            candidates[i] = 1
            continue
        else:
            res_text.append(text)
            res_type_idx.append(type_idx)
            if depth>1:
                children_ids = line['chilren']
                sample_num = int(0.2*len(children_ids))
                sample_ids = random.sample(children_ids, len(children_ids)-sample_num)
                candidates[item] = 1 for item in sample_ids
    return res_text,res_type_idx

             

            


def sample_one_text(line,index,min_depth=2,min_length = 20,max_length=512):
#    line = json.loads(line)
    tag_num = len(line)
    idx = 0
    while(1):
        idx+=1
        if idx>100:
            return None,None,index
            break
        num = random.randint(0,tag_num-1)
        if num not in index:
            index[num] = 1
            one_text,one_type_idx,depth = get_text(line,num)
            if len(one_text) < max_length and len(one_text)>=min_length and depth>=min_depth:
                break
    return one_text,one_type_idx,index

test_file = '/home/yu_guo/DataPreProcess/data/wiki_html.json'
data_file = '/home/yu_guo/DataPreProcess/data/wiki_html_sample.json'
cut_data_file = '/home/yu_guo/DataPreProcess/data/wiki_html_cut.json'
if __name__ == "__main__":
    line = []
    with open(test_file,'r')as f,open(cut_data_file,'w')as g:
        for line in tqdm(f):
            line = json.loads(line.strip())
            text,type_idx = cut_text(line)
            for one_text,one_type_idx in zip(text,type_idx):
                new_line = {"text":one_text,"type_idx":one_type_idx}
                g.write(json.dumps(new_line,ensure_ascii=False)+'\n')
