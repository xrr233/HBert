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

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
										  ["index", "label"])
lock = Lock()
num_instances = Value('i', 0)
num_docs = Value('i', 0)
num_words = Value('i', 0)

TEMP_DIR = './'

class DocumentDatabase:
	def __init__(self, reduce_memory=False):
		if reduce_memory:
			self.temp_dir = TemporaryDirectory(dir=TEMP_DIR)
			self.working_dir = Path(self.temp_dir.name)
			self.document_shelf_filepath = self.working_dir / 'shelf.db'
			self.document_shelf = shelve.open(str(self.document_shelf_filepath),
											  flag='n', protocol=-1)
			self.documents = None
		else:
			self.documents = []
			self.document_shelf = None
			self.document_shelf_filepath = None
			self.temp_dir = None
		self.doc_lengths = []
		self.doc_cumsum = None
		self.cumsum_max = None
		self.reduce_memory = reduce_memory

	def add_document(self, document):
		if not document:
			return
		if self.reduce_memory:
			current_idx = len(self.doc_lengths)
			self.document_shelf[str(current_idx)] = document
		else:
			self.documents.append(document)
		self.doc_lengths.append(len(document))

	def __len__(self):
		return len(self.doc_lengths)

	def __getitem__(self, item):
		if self.reduce_memory:
			return self.document_shelf[str(item)]
		else:
			return self.documents[item]

	def __contains__(self, item):
		if str(item) in self.document_shelf:
			return True
		else:
			return False

	def __enter__(self):
		return self

	def __exit__(self, exc_type, exc_val, traceback):
		if self.document_shelf is not None:
			self.document_shelf.close()
		if self.temp_dir is not None:
			self.temp_dir.cleanup()

def store_QA_data(filename,bert_tokenizer,output_file):
	data = {}
	with open(filename,'r')as f,open(output_file,'w')as g:
		for idx,line in enumerate(f):
			if idx%3 == 2:
				json.dump(data,f)
				continue
			elif idx%3 == 0:
				data['post'] =bert_tokenizer.tokenize(line.strip().lower())
			elif idx%3 == 1:
				data['response'] = bert_tokenizer.tokenize(line.strip().lower())

def get_QA_data(filename):
	post = []
	resp = []
	with open(filename,'r')as f:
		for idx,line in enumerate(f):
			data = json.loads(line.strip())
			post.append(data['post'])
			resp.append(data['resp'])
	assert len(post) == len(response)
	return post,resp


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, whole_word_mask, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""
    cand_indices = []
    # [MASK] word from DOC, not the query
    START_DOC = False
    for (i, token) in enumerate(tokens):
        if token == "[SEP]":
            START_DOC = True
            continue
        if token == "[CLS]":
            continue
        if not START_DOC:
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (whole_word_mask and len(cand_indices) >= 1 and token.startswith("##")):
            cand_indices[-1].append(i)
        else:
            cand_indices.append([i])

    num_to_mask = min(max_predictions_per_seq, max(1, int(round(len(cand_indices) * masked_lm_prob))))
    shuffle(cand_indices)
    # print(tokens)
    # print(cand_indices, num_to_mask)
    mask_indices = sorted(sample(cand_indices, num_to_mask))
    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indices:
        if len(masked_lms) >= num_to_mask:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_mask:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = choice(vocab_list)
            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
            tokens[index] = masked_token

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]

    return tokens, mask_indices, masked_token_labels

def select_neg_resps(resps_list, post, resp):
	cc = 0
	while True and cc < 100:
		rneg = choice(resps_list)
		if rneg != post and rneg != resp:
			break
		cc += 1
	return rneg


def construct_pointwise_data(examples,chunk_indexs,max_seq_len,bert_tokenizer,masked_lm_prob,
            max_predictions_per_seq,bert_vocab_list,epoch_filename,resps_list):
	with open(epoch_filename,'w')as g:
		num_examples = len(examples)
		print("num_examples", num_examples)
		num_instance = 0
		num_instances_value = 0
		for doc_idx in tqdm(chunk_indexs):
		# print(doc_idx)
			if doc_idx % 100 == 0:
				print(doc_idx)
			example = examples[doc_idx]

			instances = [] 
			post_tokens = example['post']
			resp_tokens = example['response']
			neg_resp_tokens = select_neg_resps(resps_list,post_tokens,resp_tokens)

			pos_post = copy.deepcopy(post_tokens)[:250]
			pos_resp = copy.deepcopy(resp_tokens)[:250]
			neg_post = copy.deepcopy(post_tokens)[:250]
			neg_resp = copy.deepcopy(neg_resp_tokens)[:250]
			try:
				truncate_seq_pair(pos_post, pos_resp, max_seq_len - 3)
				truncate_seq_pair(neg_post, neg_resp, max_seq_len - 3)
			except:
				break

			pos_tokens = ["[CLS]"] + pos_post + ["[SEP]"] + pos_resp + ["[SEP]"]
			pos_segment_ids = [0 for _ in range(len(pos_post) + 2)] + [1 for _ in range(len(pos_resp) + 1)]
			neg_tokens = ["[CLS]"] + neg_post + ["[SEP]"] + neg_resp + ["[SEP]"]
			neg_segment_ids = [0 for _ in range(len(neg_post) + 2)] + [1 for _ in range(len(neg_resp) + 1)]
			pos_tokens, pos_masked_lm_positions, pos_masked_lm_labels = create_masked_lm_predictions(
				pos_tokens, masked_lm_prob, max_predictions_per_seq, True, bert_vocab_list)
			neg_tokens, neg_masked_lm_positions, neg_masked_lm_labels = create_masked_lm_predictions(
				neg_tokens, masked_lm_prob, max_predictions_per_seq, True, bert_vocab_list)
			pos_tokens_idx = bert_tokenizer.convert_tokens_to_ids(pos_tokens)
			pos_tokens_idx_labels = bert_tokenizer.convert_tokens_to_ids(pos_masked_lm_labels)
			neg_tokens_idx = bert_tokenizer.convert_tokens_to_ids(neg_tokens)
			neg_tokens_idx_labels = bert_tokenizer.convert_tokens_to_ids(neg_masked_lm_labels)
			pos_instance = {
				"tokens_idx":pos_tokens_idx,
				"tokens":pos_tokens,
				"segment_ids": pos_segment_ids,
				"label": 1,
				"masked_lm_positions": pos_masked_lm_positions,
				"masked_lm_labels_idxs": pos_tokens_idx_labels,
				}
			neg_instance = {
				"tokens_idx":neg_tokens_idx,
				"tokens":neg_tokens,
				"segment_ids": neg_segment_ids,
				"label": 0,
				"masked_lm_positions": neg_masked_lm_positions,
				"masked_lm_labels_idxs": neg_tokens_idx_labels,
				}
			g.write(json.dumps(pos_instance, ensure_ascii=False)+'\n')
			g.write(json.dumps(neg_instance, ensure_ascii=False)+'\n')
			num_instances_value += 1

	return num_instances_value
			
		




def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
	"""Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_num_tokens:
			break
		# truncate from the doc side
		tokens_b.pop()

def sample_neg_from_passage(anchor_passage_tokenized, pos_query, p_word_weights):
	len_pos_query = len(pos_query)
	len_anchor_passage = len(anchor_passage_tokenized)

	# word_set = []
	i = 0
	if len_anchor_passage == 0:
		return []
	

	words = list(p_word_weights.keys())
	if len(words) == 0:
		return []
	word_weights = [p_word_weights[w] for w in words]
	# print("ww", word_weights)
	normalized_word_weights = softmax(np.array(word_weights))

	# print("nww",normalized_word_weights)
	samples = np.random.choice(a=len(words), size=len_pos_query, replace=True, p=normalized_word_weights)
	word_set = [words[s] for s in samples]

	# while len(word_set) < len_pos_query and i < 1000:
	# 	idx = np.random.choice(list(range(len_anchor_passage)),size=1)[0]
	# 	wordidx = anchor_passage_tokenized[idx]
	# 	if wordidx not in pos_query:
	# 		word_set.append(wordidx)
	# 	i += 1

	return word_set

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--do_lower_case", action="store_true")
	parser.add_argument("--bert_model", type=str, default='../bert_base_uncased')
	parser.add_argument("--reduce_memory", action="store_true",
						help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")
	parser.add_argument("--epochs_to_generate", type=int, default=1,
						help="Number of epochs of data to pregenerate")
	# parser.add_argument("--output_dir", type=str, required=True)
	parser.add_argument("--max_seq_len", type=int, default=128)
	parser.add_argument("--mlm", action="store_true")
	parser.add_argument("--masked_lm_prob", type=float, default=0.15,
						help="Probability of masking each token for the LM task")
	parser.add_argument("--max_predictions_per_seq", type=int, default=60,
						help="Maximum number of tokens to mask in each sequence")
	parser.add_argument("--rop_num_per_doc", type=int, default=1,
						help="How many samples for each document")
	parser.add_argument("--pairnum_per_doc", type=int, default=2,
						help="How many samples for each document")
	parser.add_argument("--num_workers", type=int, default=16,
						help="The number of workers to use to write the files")
	parser.add_argument("--mu", type=int, default=512,
						help="The number of workers to use to write the files")
	parser.add_argument('--output_dir', type=str, required=True)
	parser.add_argument('--sentence_file_path', type=str, required=True)
	#parser.add_argument('--stop_words_file', type=str, required=True)
	#parser.add_argument('--anchor_file', type=str, required=True)
	#parser.add_argument('--passage_file', type=str, required=True)

	args = parser.parse_args()

	bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
	bert_vocab_list = list(bert_tokenizer.vocab.keys())

	examples = []
	resps_list = []
	with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
		with open(args.sentence_file_path) as f:
			for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
				example = json.loads(line.strip())
				resps_list.append(example['response'])
				# examples.append(example)
				docs.add_document(example)
		print('Reading file is done! Total doc num:{}'.format(len(docs)))

		
		for epoch in range(args.epochs_to_generate):
			epoch_filename =  f"{args.output_dir}/epoch_{epoch}.json"
			if os.path.exists(epoch_filename):
				with open(epoch_filename, "w") as ef:
					print(f"start generating {epoch_filename}")
			# num_processors = args.num_workers
			# processors = Pool(num_processors)
			cand_idxs = list(range(0, len(docs)))
			chunk_size = int(len(cand_idxs) / 1)
			chunk_indexs = cand_idxs[0*chunk_size:(0+1)*chunk_size]
			num_instances_value = construct_pointwise_data(docs, chunk_indexs, args.max_seq_len, bert_tokenizer, args.masked_lm_prob, \
				args.max_predictions_per_seq, bert_vocab_list, epoch_filename, resps_list)


			# for i in range(num_processors):
			# 	chunk_size = int(len(cand_idxs) / num_processors)
			# 	chunk_indexs = cand_idxs[i*chunk_size:(i+1)*chunk_size]
			# 	# print("?")
			# 	r = processors.apply_async(construct_pairwise_examples, (docs, chunk_indexs, args.max_seq_len, args.mlm, bert_tokenizer, args.masked_lm_prob, \
			# 	args.max_predictions_per_seq, bert_vocab_list, epoch_filename, args.pairnum_per_doc, word2df, args.mu, len(docs), \
			# 		stopwords, anchors, passages), error_callback=error_callback)
			# processors.close()
			# processors.join()

			metrics_file =  f"{args.output_dir}/epoch_{epoch}_metrics.json"
			with open(metrics_file, 'w') as metrics_file:
				metrics = {
					"num_training_examples": num_instances_value,
					"max_seq_len": args.max_seq_len
				} 
				metrics_file.write(json.dumps(metrics))     



# python gen_my_data.py --output_dir ../data/QA_data --sentence_file_path ../data/test_data_dev.json