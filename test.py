import os
from argparse import ArgumentParser
import sys
sys.path.append('./')
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

def get_QA_data(filename,bert_tokenizer,output_file):
	data = {}
	with open(filename,'r')as f,open(output_file,'w')as g:
		for idx,line in enumerate(f):
			if idx%3 == 2:
				g.write(json.dumps(data)+'\n')
				continue
			elif idx%3 == 0:
				data['post'] =bert_tokenizer.tokenize(line.strip().lower())
			elif idx%3 == 1:
				data['response'] = bert_tokenizer.tokenize(line.strip().lower())

if __name__ == "__main__":
    input_file = '../data/test_data_train.txt'
    output_file = '../data/test_data_train.json'
    bert_tokenizer = BertTokenizer.from_pretrained('../bert_base_uncased', do_lower_case = True)
    bert_vocab_list = list(bert_tokenizer.vocab.keys())
    get_QA_data(input_file,bert_tokenizer,output_file)