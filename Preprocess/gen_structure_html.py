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



def get_structure_info(nodes,max_node_num,node_max_length):
    node_queue=[]
    res = {}
    nodes_info = []
    attention_mask = []
    node_num = min(max_node_num,len(nodes))
    node_queue.append(nodes[0])
    i = 0
    while(len(node_queue)<node_num):
        visual_idx = [0]*node_num
        visual_idx[i] = 1
        node_info={}
        tag_name = node_queue[i]["name"]
        tag_children = node_queue[i]["children"]
        children_num = len(tag_children)
        tag_text = node_queue[i]["text"]
        tag_idx = node_queue[i]["id"]
        if tag_name=='textnode':
            text,type_idx,_ = get_text(nodes,tag_idx)
            node_info['type']='text'
            node_info['text'] = text[:node_max_length]
            node_info['type_idx']=type_idx[:node_max_length]
            nodes_info.append(node_info)
            attention_mask.append(visual_idx)
            i+=1
            continue
        if(len(node_queue)+children_num>node_num):
            text,type_idx,_ = get_text(nodes,tag_idx)
            node_info['type']='text'
            node_info['text'] = text[:node_max_length]
            node_info['type_idx']=type_idx[:node_max_length]
            nodes_info.append(node_info)
            attention_mask.append(visual_idx)
            break
        for item in tag_children:
            node_queue.append(nodes[item])
            child_idx = len(node_queue)-1
            visual_idx[child_idx] = 1
        text=['<'+tag_name+'>']
        type_idx=[1]
        node_info['type']='tag'
        node_info['text'] = text[:node_max_length]
        node_info['type_idx']=type_idx[:node_max_length]
        nodes_info.append(node_info)
        attention_mask.append(visual_idx)
        i+=1
    assert len(nodes_info) == len(attention_mask)
    assert len(nodes_info)<=max_node_num
    return nodes_info,attention_mask


            

    # for i in range(node_num):
    #     visual_idx = [0]*node_num
    #     visual_idx[i] = 1
    #     node_info={}
    #     tag_name = nodes[i]["name"]
    #     tag_children = nodes[i]["children"]
    #     for item in tag_children:
    #         visual_idx[item] = 1
    #     tag_text = nodes[i]["text"]
    #     tag_idx = nodes[i]["id"]
    #     last_chidren = int(tag_children[-1])
    #     if last_chidren>=node_num:
    #         text,type_idx = get_text(nodes,i)
    #         visual_idx = [0]*node_num
    #     else:
    #         if tag_name=='text_node':
    #             text = tag_text
    #             type_idx = [0] * len(text)
    #         else:
    #             text = []
    #             type_idx=[]
    #     text = text[:node_max_length]
    #     type_idx = type_idx[:node_max_length]
    #     node_info['text'] = text
    #     node_info['type_idx'] = type_idx
    #     nodes_info.append(node_info)
    #     attention_mask.append(visual_idx)
    # res['nodes_info'] = nodes_info
    # res['attention_mask'] = attention_mask





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

def sample_k_text(line,min_depth=2,min_length = 10,max_length=512,k=15):
    text = []
    type_idx = []
    index = {}
    for i in range(k):
        one_text,one_type_idx,index = sample_one_text(line,index,min_depth,min_length,max_length)
        if one_text:
            text.append(one_text)
            type_idx.append(one_type_idx)
        else:break
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

test_file = '/home/yu_guo/DataPreProcess/data/html.json'
data_file = '/home/yu_guo/DataPreProcess/data/html_struture.json'
if __name__ == "__main__":
    line = []
    with open(test_file,'r')as f,open(data_file,'w')as g:
        for line in tqdm(f):
            line = json.loads(line.strip())
            nodes_info,attention_mask = get_structure_info(line,max_node_num=32,node_max_length=64)
            new_line = {"nodes_info":nodes_info,"attention_mask":attention_mask}
            g.write(json.dumps(new_line,ensure_ascii=False)+'\n')
            # text,type_idx = cut_text(line)
            # for one_text,one_type_idx in zip(text,type_idx):
            #     new_line = {"text":one_text,"type_idx":one_type_idx}
            #     g.write(json.dumps(new_line,ensure_ascii=False)+'\n')
