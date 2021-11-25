import os
from argparse import ArgumentParser
import sys
sys.path.append('../')
from tqdm import tqdm
import json
from transformers import BertTokenizer
import numpy as np
from random import random, shuffle, choice, sample
import collections
import traceback
from multiprocessing import Pool, Value, Lock
from tempfile import TemporaryDirectory
from pathlib import Path
import shelve
from collections import Counter
import math
import copy
import sys
from transformers import BertTokenizer, BertModel

def load_data(file):
    graph = {}
    with open(file,'r')as f:
        for line in tqdm(f):
            line = json.loads(line.strip())
            graph[line['title']] = line
    return graph


def sample_task1_data(graph,keys,src_title,max_length,weight_threshold):
    anchor_list = []
    existing_page = [src_title]
    pre = src_title
    anchor_list=[]
    sample_page = graph[choice(keys)]['passage']
    for i in range(max_length):
        anchors = graph[pre]['anchors']
        shuffle(anchors)
        flag = 0
        for item in anchors:
            title = item['anchor_title']
            anchor_weight = item['anchor_weight']
            if title not in existing_page and title in graph and anchor_weight >= weight_threshold:
                if i==0:
                    sentence = item['sentence']
                flag=1
                item['anchor_passage']=graph[title]['passage']
                anchor_list.append(item)
                existing_page.append(title)
                pre = title
                break
        if flag==0:
            break

    instance={
        'src_title':src_title,
        'sentence':sentence,
        'anchor_list':anchor_list,
        'sample_passage':passage,
        'length':len(anchor_list)
    }
    return instance

def main():
    graph = load_data('./data/anchor_data_with_weight.json')
    keys = list(graph.keys())

    with open('./data/task1_data.json','w')as f:
        for src_title in tqdm(graph):
            for i in range(5):
                f.write(json.dumps(sample_task1_data(graph,keys,src_title,3,0.01))+'\n')

if __name__ == "__main__":
    main()
