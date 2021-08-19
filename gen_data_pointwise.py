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


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
	"""Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
	while True:
		total_length = len(tokens_a) + len(tokens_b)
		if total_length <= max_num_tokens:
			break
		# truncate from the doc side
		tokens_b.pop()

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

def sample_neg_from_passage(anchor_passage_tokenized, pos_query):
	len_pos_query = len(pos_query)
	len_anchor_passage = len(anchor_passage_tokenized)

	word_set = []
	i = 0
	if len_anchor_passage == 0:
		return []
	while len(word_set) < len_pos_query and i < 1000:
		idx = np.random.choice(list(range(len_anchor_passage)),size=1)[0]
		wordidx = anchor_passage_tokenized[idx]
		if wordidx not in pos_query:
			word_set.append(wordidx)
		i += 1
	return word_set



def construct_pairwise_examples(examples, chunk_indexs, max_seq_len, mlm, bert_tokenizer, masked_lm_prob, 
			max_predictions_per_seq, bert_vocab_list, epoch_filename, pairnum_per_doc, word2df, mu, total_doc_cnt):
	# print(examples)
	num_examples = len(examples)
	num_instance = 0
	for doc_idx in tqdm(chunk_indexs):
		example = examples[doc_idx]

		sentence = example['sentence']
		sentence_tokenized = bert_tokenizer.tokenize(sentence)

		anchors = example['anchors']
		instances = []
		for i, anchor in enumerate(anchors):
			# print(anchor)
			assert len(anchor) == 1
			anchor = anchor[0]
			text = anchor['text']
			idx_start, idx_end = anchor['pos']
			anchor_passage = anchor['passage']
			anchor_passage_tokenized = bert_tokenizer.tokenize(anchor_passage)
			
			pos_query = sentence_tokenized
			neg_query = sample_neg_from_passage(anchor_passage_tokenized, pos_query)
			if len(neg_query) <= 1:
				continue
			if len(pos_query) <= 1:
				continue
			if len(anchor_passage_tokenized) <= 5:
				continue
			
			pair_wise_instances = []
			for j, query in enumerate([pos_query, neg_query]):

				if j == 0:
					label = 1
				else:
					label = 0
				
				anchor_passage_tokenized_copied = copy.deepcopy(anchor_passage_tokenized)
				# 不然会truncate两次，就不对了。
				try:
					truncate_seq_pair(query, anchor_passage_tokenized_copied, max_seq_len - 3)
				except:
					break
				tokens = ["[CLS]"] + query + ["[SEP]"] + anchor_passage_tokenized_copied + ["[SEP]"]
				segment_ids = [0 for _ in range(len(query) + 2)] + [1 for _ in range(len(anchor_passage_tokenized_copied) + 1)]
				

				if mlm:
				    tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
				        tokens, masked_lm_prob, max_predictions_per_seq, True, bert_vocab_list)
				else:
				    masked_lm_positions, masked_lm_labels = [], []
				
		
				tokens_idx = bert_tokenizer.convert_tokens_to_ids(tokens)
				tokens_idx_labels = bert_tokenizer.convert_tokens_to_ids(masked_lm_labels)

				instance = {
				    "tokens_idx":tokens_idx,
				    "segment_ids": segment_ids,
				    "label": label,
				    "masked_lm_positions": masked_lm_positions,
					"masked_lm_labels_idxs": tokens_idx_labels,
				    }

				# 确保每次是pair-wise的！
				pair_wise_instances.append(instance)

			if len(pair_wise_instances) == 2:
				instances += pair_wise_instances


		doc_instances = [json.dumps(instance, ensure_ascii=False) for instance in instances]
		lock.acquire()
		with open(epoch_filename,'a+') as epoch_file:
			for i, instance in enumerate(doc_instances):
				epoch_file.write(instance + '\n')
				num_instances.value += 1
		lock.release()





def error_callback(e):
	print('error')
	print(dir(e), "\n")
	traceback.print_exception(type(e), e, e.__traceback__)

		

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--train_corpus', type=str, required=True)
	parser.add_argument("--do_lower_case", action="store_true")
	parser.add_argument("--bert_model", type=str, default='bert-base-uncased')
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
	
	args = parser.parse_args()


	bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
	bert_vocab_list = list(bert_tokenizer.vocab.keys())

	# word2df = {}
	# with open(os.path.join(args.output_dir, f"word2df.json"), 'r') as wdf:
	#     word2df = json.loads(wdf.read().strip())
	word2df = None
	


	examples = []
	with DocumentDatabase(reduce_memory=args.reduce_memory) as docs:
		with open(args.train_corpus) as f:
			for line in tqdm(f, desc="Loading Dataset", unit=" lines"):
				example = line.strip().lower()
				example = bert_tokenizer.tokenize(example)
				print(example)
				#example = json.loads(line)
				# examples.append(example)
				docs.add_document(example)
		print('Reading file is done! Total doc num:{}'.format(len(docs)))

		
		for epoch in range(args.epochs_to_generate):
			epoch_filename =  f"{args.output_dir}/epoch_{epoch}.json"
			if os.path.exists(epoch_filename):
				with open(epoch_filename, "w") as ef:
					print(f"start generating {epoch_filename}")
			num_processors = args.num_workers
			processors = Pool(num_processors)
			cand_idxs = list(range(0, len(docs)))

			for i in range(num_processors):
				chunk_size = int(len(cand_idxs) / num_processors)
				chunk_indexs = cand_idxs[i*chunk_size:(i+1)*chunk_size]
				# print("?")
				r = processors.apply_async(construct_pairwise_examples, (docs, chunk_indexs, args.max_seq_len, args.mlm, bert_tokenizer, args.masked_lm_prob, \
				args.max_predictions_per_seq, bert_vocab_list, epoch_filename, args.pairnum_per_doc, word2df, args.mu, len(docs)), error_callback=error_callback)
			processors.close()
			processors.join()


			# metrics = construct_pairwise_examples(examples, , args.max_seq_len, args.mlm, bert_tokenizer, args.masked_lm_prob, \
			#     args.max_predictions_per_seq, bert_vocab_list, epoch_filename, args.pairnum_per_doc)
			
			metrics_file =  f"{args.output_dir}/epoch_{epoch}_metrics.json"
			with open(metrics_file, 'w') as metrics_file:
				metrics = {
					"num_training_examples": num_instances.value,
					"max_seq_len": args.max_seq_len
				} 
				metrics_file.write(json.dumps(metrics))       


			
			


