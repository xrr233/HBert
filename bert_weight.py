import torch
import torch.nn as nn
import transformers
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from argparse import ArgumentParser
import json
import math
import os
from multiprocessing import Process
import numpy as np


class Worker():
    def __init__(self, src_fp, tgt_fp, func):
        self.src_fp = src_fp
        self.tgt_fp = tgt_fp
        self.parse_line = func
        self.postfix = '.pid.'
        # self.tokenizer = AutoTokenizer.from_pretrained('./bert_base_uncased')
        # self.model = AutoModel.from_pretrained('./bert_base_uncased')

    def run(self, pid, p_num,model,tokenizer):
        pid_file_fp = self.tgt_fp + self.postfix + str(pid)
        with open(self.src_fp, 'r') as f_in, open(pid_file_fp, 'w') as f_out:
            for idx, line in enumerate(f_in):
                if idx % p_num != pid: continue
                out_string = self.parse_line(json.loads(line),model,tokenizer)
                if out_string: f_out.write(json.dumps(out_string)+ '\n')

    def merge_result(self, keep_pid_file=False):
        os.system('cat %s%s* > %s' % (self.tgt_fp, self.postfix, self.tgt_fp))
        if not keep_pid_file:
            os.system('rm %s%s*' % (self.tgt_fp, self.postfix))

class MultiProcessor():
    def __init__(self, worker, pid_num):
        self.worker = worker
        self.pid_num = pid_num
        self.tokenizer = AutoTokenizer.from_pretrained('./bert_base_uncased')
        self.model = AutoModel.from_pretrained('./bert_base_uncased')

    def run(self):
        for pid in range(self.pid_num):
            p = Process(target= self.worker.run, args = (pid, self.pid_num,self.model,self.tokenizer))
            p.start()
            
        for pid in range(self.pid_num):
            p.join()

def get_anchor_attention(anchor_offset,offsets,attentions):
    left = 0
    right = 0
    left_flag = 0
    seq_len = len(offsets)
    for offset in offsets:
        if left_flag==0:
            if offset[0] == anchor_offset[0]:
                left_flag = 1
                right=left
            else:
                left+=1
        else:
            if offset[1] == anchor_offset[1]:  
                break
            else:
                right+=1
    return float(sum(attentions[left+1:right+1]))



def get_text_weight(passages,anchorses,batch_size,model,tokenizer,device,max_length=256):
    input_ids = []
    offsets  = []
    attention_masks = []
    for passage in passages:
        #print(type(passage))
        new_tokens = tokenizer.encode_plus(passage, max_length=max_length,
                                        truncation=True,
                                        padding=True,
                                        return_offsets_mapping = True,
                                        return_overflowing_tokens = True,
                                        return_tensors='pt')
        offset = new_tokens['offset_mapping'][0]
        input_id = new_tokens['input_ids'][0].view(-1).to(device)
        attention_mask = new_tokens['attention_mask'][0].view(-1).to(device)
        offsets.append(offset)
        input_ids.append(input_id)
        attention_masks.append(attention_mask)
    input_ids = nn.utils.rnn.pad_sequence(input_ids,batch_first=True,padding_value=0)
    attention_masks = nn.utils.rnn.pad_sequence(attention_masks,batch_first=True,padding_value=0)
    # We process these tokens through our model:
    outputs = model(input_ids=input_ids,
                    attention_mask=attention_masks,
                    return_dict=True,
                    output_attentions = True
                    )
    attentions = torch.mean(outputs['attentions'][-1][:,:,0,:],dim=1)
    for j in range(len(anchorses)):
        for i in range(len(anchorses[j])):
            anchor_offset = anchorses[j][i]['anchor_passage_pos']
            #print(anchor_offset,offsets[j])
            anchor_attention = get_anchor_attention(anchor_offset,offsets[j],attentions[j])
            anchorses[j][i]['anchor_weight'] = anchor_attention
            #print(new_tokens['input_ids'].device)
    #print(input_ids.size())
    return input_ids


                                     
def parse_line(lines,batch_size,model,tokenizer,device):
    passages = []
    anchorses = []
    for line in lines:
        passage = line['passage']
        passages.append(passage)
        anchors = line['anchors']
        anchorses.append(anchors)

    tokens = get_text_weight(passages,anchorses,batch_size,model,tokenizer,device)
    for i in range(batch_size):
        lines[i]['anchors'] = anchorses[i]
        lines[i]['tokens_id'] = tokens[i].cpu().numpy().tolist() 
    return lines

def handle_file(input_file,output_file,pid,batch_size,model,tokenizer,device):
    with open(input_file,'r')as f,open(output_file,'w') as g:
        num = 3000000
        start = pid*num
        end = (pid+1)*num
        i = 0
        lines = []
        for idx,line in tqdm(enumerate(f)):
            if idx<start or idx>=end:
                continue
            line  = json.loads(line.strip())
            lines.append(line)
            i+=1
            if i==batch_size:
                output_lines = parse_line(lines,batch_size,model,tokenizer,device)
                i=0
                lines=[]
                for output_line in output_lines:
                    g.write(json.dumps(output_line)+'\n')
            else:
                continue
 
def main():
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--pid",type=str,required=True)
    args = parser.parse_args()
    # worker = Worker(args.input_file,args.output_file,parse_line)
    tokenizer = AutoTokenizer.from_pretrained('./bert_base_uncased')
    model = AutoModel.from_pretrained('./bert_base_uncased')
    model = model.to(device)
    handle_file(args.input_file,args.output_file+'pid.'+args.pid,int(args.pid),args.batch_size,model,tokenizer,device)
    # mp = MultiProcessor(worker, 10)
    # mp.run()
    # mp.merge_result()
    # print("All Processes Done.")
if __name__ == "__main__":
    main()





# reformat list of tensors into single tensor
