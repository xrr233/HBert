import os
import glob
import time
import argparse
os.system('pip install dataclasses')
os.system("yum install -y unzip zip")
os.system("apt-get install zip")
import moxing as mox
import sys
import torch
import moxing as mox
mox.file.shift('os', 'mox')
mox.file.copy_parallel('s3://obs-app-2020042019121301221/SEaaKM/m50017495/lcemodel/', '/cache/lcemodel/')
mox.file.copy_parallel('s3://obs-app-2020042019121301221/SEaaKM/m50017495/code/anchors/', '/cache/anchors/')
os.system('pip install -r /cache/lcemodel/requirements.txt')
os.system('pip install dgl-cu101')

os.system('cd /cache/lcemodel/Reranker && pip install .')
os.system('nvidia-smi')



s3_model_path = 's3://obs-app-2020042019121301221/SEaaKM/m50017495/output/anchor_output/HBert_ablation1/'
s3_processed_path = 's3://obs-app-2020042019121301221/SEaaKM/z00562934/data/lce_data/json_output_20neg_shuffle_maxP'
s3_processed_rerank_path = 's3://obs-app-2020042019121301221/SEaaKM/m50017495/data/anchor_data/reranker/processed_rerank_n20/'

s3_inference_path = 's3://obs-app-2020042019121301221/SEaaKM/z00562934/data/lce_data/inference_ance'
s3_inference_rerank_path = 's3://obs-app-2020042019121301221/SEaaKM/m50017495/data/anchor_data/reranker/inference_output_rerank_dev/'


s3_inference_trec_path = 's3://obs-app-2020042019121301221/SEaaKM/m50017495/data/anchor_data/reranker/processed_trecdl_fullrank_eval_r/'
s3_inference_trec_rerank_path = 's3://obs-app-2020042019121301221/SEaaKM/m50017495/data/anchor_data/reranker/processed_trecdl_rerank_eval_r/'

s3_req_path = "s3://obs-app-2020042019121301221/SEaaKM/m50017495/data/requirement/"
s3_output_path = 's3://obs-app-2020042019121301221/SEaaKM/m50017495/output/anchor_output/HBert_document_ranking_ablation1'

# 16 2000
# 8 4000
# 4 8000
# 2 16000
ml = 512
bs = 8
save_step = 5000
epoch_num = 2
eval_bs_singlecard = 512
print("s3_model_path", s3_model_path)
print("s3_processed_path", s3_processed_path)
print("s3_inference_path", s3_inference_path)
print("s3_output_path", s3_output_path)
os.system('pip install transformers')

os.system('pip install datasets')
mox.file.copy_parallel("s3://obs-app-2020042019121301221/SEaaKM/g50020960/datasets/HBert_new/","/cache/")
os.system('mkdir /home/work/user-job-dir/Pretrain/HBert_fine_tune/')
os.system(f'python /home/work/user-job-dir/Pretrain/run_mlm_ft.py \
        --fp16 \
        --model_name_or_path /home/work/user-job-dir/Pretrain/HBert/FraBert_scrach_v1 \
        --node_config_name /home/work/user-job-dir/Pretrain/node_bert/bert_base_uncased \
        --train_file /cache/HBert_30_20W.json \
        --do_train \
        --ignore_data_skip \
        --learning_rate 5e-5 \
        --per_device_train_batch_size 2 \
        --num_train_epochs 1 \
        --dataloader_num_workers 8 \
        --save_steps 10000 \
        --output_dir /home/work/user-job-dir/Pretrain/HBert_fine_tune \
        --dataset_script_dir /home/work/user-job-dir/Pretrain/data_scripts \
        --dataset_cache_dir /home/work/user-job-dir/Pretrain/cache \
        --limit 50000000 \
        --overwrite_output_dir ')

mox.file.copy_parallel("/home/work/user-job-dir/Pretrain/HBert_fine_tune/",s3_model_path)




# need input and output folder on s3

def extract_data():

    os.makedirs('/home/work/mymodel')
    mox.file.copy_parallel(s3_model_path, '/home/work/mymodel') 
    os.system("rm /home/work/mymodel/trainer_state.json")
    os.makedirs('/cache/data')
    mox.file.copy_parallel(s3_processed_path, '/cache/data/processed')
    mox.file.copy_parallel(s3_processed_rerank_path, '/cache/data/processed_rerank')
    # obs-app-2020042019121301221/SEaaKM/m50017495/data/msmarco/

    os.makedirs('/cache/msmarco')
    mox.file.copy_parallel('s3://obs-app-2020042019121301221/SEaaKM/m50017495/data/anchor_data/msmarco/', '/cache/msmarco')


    # /cache/inference_output
    # obs-app-2020042019121301221/SEaaKM/m50017495/data/inference_hdct/
    mox.file.copy_parallel(s3_inference_path, '/cache/inference')
    mox.file.copy_parallel(s3_inference_rerank_path, '/cache/inference_rerank')


    mox.file.copy_parallel(s3_inference_trec_path, '/cache/inference_trec')
    mox.file.copy_parallel(s3_inference_trec_rerank_path, '/cache/inference_trec_rerank')

    

    # os.system('cd /cache/data && unzip processed.zip')

    os.makedirs('/cache/score')
    os.makedirs('/cache/output')
    # # os.makedirs('/cache/data_mlm')

def install_package():
    os.makedirs('/cache/mypackages/')
    mox.file.copy_parallel(s3_req_path, '/cache/mypackages/')   
    os.system("pip install sentencepiece==0.1.90")
    print("begin pytrec")
    os.system("cd /cache/mypackages/pytrec_eval-0.5 && python setup.py install")
    print("pytrec ok")

def parse_args():
    parser = argparse.ArgumentParser(description='Process Reader Data')
    # to ignore
    parser.add_argument('--data_url', default='s3://bucket-857/h00574873/test/model_save/',
                        help='data_url for yundao')
    parser.add_argument('--init_method', default='',
                        help='init_method for yundao')
    parser.add_argument('--train_url', default='s3://bucket-857/h00574873/test/model_save/',
                        help='train_url for yundao')
    parser.add_argument("--s3_path_dir", type=str,
                        default='s3://bucket-852/f00574594/data/HGN_data/train_data_with_tfidf30_bert_large_aug/path_data/',
                        help='define path directory')
    parser.add_argument("--s3_HGN_data_dir", type=str,
                        default='s3://bucket-852/f00574594/data/KFB_data/reader_data_no_sep/',
                        help='define output directory')
    parser.add_argument("--my_output_dir", type=str,
                        default='s3://bucket-852/m50017495/replearn/output_train/',
                        help='define output directory')
    return parser.parse_args()

def main():
    extract_data()
    args = parse_args()
    install_package()

    os.makedirs("/cache/fullrank")

    print(f"[fullrank] finetune....")
    
    os.system(f'cd /cache/lcemodel/Reranker/examples/msmarco-doc/ && python -m torch.distributed.launch --nproc_per_node 8 run_marco.py \
        --output_dir /cache/fullrank/reranker_output \
        --model_name_or_path /home/work/mymodel \
        --do_train \
        --save_steps {save_step} \
        --train_dir /cache/data/processed \
        --max_len {ml} \
        --fp16 \
        --per_device_train_batch_size {bs} \
        --train_group_size 8 \
        --gradient_accumulation_steps 1 \
        --per_device_eval_batch_size 64 \
        --warmup_ratio 0.1 \
        --weight_decay 0.01 \
        --learning_rate 1e-5 \
        --num_train_epochs {epoch_num} \
        --overwrite_output_dir \
        --dataloader_num_workers 8 \
        --collaborative')

    # mox.file.copy_parallel('/cache/fullrank/reranker_output', s3_output_path)

    print(f"[fullrank] msmarco predicting....")
    os.system(f'cd /cache/lcemodel/Reranker/examples/msmarco-doc/ && CUDA_VISIBLE_DEVICES=0 python run_marco.py \
        --output_dir /cache/fullrank/reranker_output \
        --model_name_or_path /cache/fullrank/reranker_output \
        --tokenizer_name /cache/fullrank/reranker_output \
        --do_predict \
        --max_len {ml} \
        --fp16 \
        --per_device_eval_batch_size {eval_bs_singlecard} \
        --dataloader_num_workers 8 \
        --pred_path /cache/inference/all.json \
        --pred_id_file  /cache/inference/ids.tsv \
        --rank_score_path /cache/score/score.txt')

    print(f"[fullrank] msmarco evaluating....")
    
    os.system("cd /cache/lcemodel/Reranker && python evaluate.py --qrels_path /cache/msmarco/msmarco-docdev-qrels.tsv --score_path /cache/score/score.txt")

    print(f"[fullrank] trec evaluating....")

    os.system(f'cd /cache/lcemodel/Reranker/examples/msmarco-doc/ && CUDA_VISIBLE_DEVICES=0 python run_marco.py \
        --output_dir /cache/fullrank/reranker_output \
        --model_name_or_path /cache/fullrank/reranker_output \
        --tokenizer_name /cache/fullrank/reranker_output \
        --do_predict \
        --max_len {ml} \
        --fp16 \
        --per_device_eval_batch_size {eval_bs_singlecard} \
        --dataloader_num_workers 8 \
        --pred_path /cache/inference_trec/all.json \
        --pred_id_file  /cache/inference_trec/ids.tsv \
        --rank_score_path /cache/score/score2.txt')
    
    os.system(f"cd /cache/anchors/finetune/ && python evaluate_rerank_trec.py --score_file /cache/score/score2.txt --qrel_file /cache/msmarco/trec-2019qrels-docs.txt ")
    
    os.system(f"mv /cache/fullrank /cache/output/")
    mox.file.copy_parallel('/cache/output', s3_output_path) 

    # mox.file.copy_parallel('/cache/score', s3_output_path)
    


    print(f"[rerank] finetune....")

    os.makedirs("/cache/rerank")
    os.system(f'cd /cache/lcemodel/Reranker/examples/msmarco-doc/ && python -m torch.distributed.launch --nproc_per_node 8 run_marco.py \
        --output_dir /cache/rerank/reranker_output \
        --model_name_or_path /home/work/mymodel \
        --do_train \
        --save_steps {save_step} \
        --train_dir /cache/data/processed_rerank/ \
        --max_len {ml} \
        --fp16 \
        --per_device_train_batch_size {bs} \
        --train_group_size 8 \
        --gradient_accumulation_steps 1 \
        --per_device_eval_batch_size 64 \
        --warmup_ratio 0.1 \
        --weight_decay 0.01 \
        --learning_rate 1e-5 \
        --num_train_epochs {epoch_num} \
        --overwrite_output_dir \
        --dataloader_num_workers 8 \
        --collaborative')

    print(f"[rerank] msmarco predicting....")

    os.system(f'cd /cache/lcemodel/Reranker/examples/msmarco-doc/ && CUDA_VISIBLE_DEVICES=0 python run_marco.py \
        --output_dir /cache/rerank/reranker_output \
        --model_name_or_path /cache/rerank/reranker_output \
        --tokenizer_name /cache/rerank/reranker_output \
        --do_predict \
        --max_len {ml} \
        --fp16 \
        --per_device_eval_batch_size {eval_bs_singlecard} \
        --dataloader_num_workers 8 \
        --pred_path /cache/inference_rerank/all.json \
        --pred_id_file  /cache/inference_rerank/ids.tsv \
        --rank_score_path /cache/score/score3.txt') 

    print(f"[rerank] msmarco evaluating....")

    os.system("cd /cache/lcemodel/Reranker && python evaluate.py --qrels_path /cache/msmarco/msmarco-docdev-qrels.tsv --score_path /cache/score/score3.txt")
    
    
    print(f"[rerank] trec predicting....")
    os.system(f'cd /cache/lcemodel/Reranker/examples/msmarco-doc/ && CUDA_VISIBLE_DEVICES=0 python run_marco.py \
        --output_dir /cache/rerank/reranker_output \
        --model_name_or_path /cache/rerank/reranker_output \
        --tokenizer_name /cache/rerank/reranker_output \
        --do_predict \
        --max_len {ml} \
        --fp16 \
        --per_device_eval_batch_size {eval_bs_singlecard} \
        --dataloader_num_workers 8 \
        --pred_path /cache/inference_trec_rerank/all.json \
        --pred_id_file  /cache/inference_trec_rerank/ids.tsv \
        --rank_score_path /cache/score/score4.txt')
    

    print(f"[rerank] trec evaluating....")
    os.system(f"cd /cache/anchors/finetune/ && python evaluate_rerank_trec.py --score_file /cache/score/score4.txt --qrel_file /cache/msmarco/trec-2019qrels-docs.txt ")


    print(f"[copying] ....")
    os.system(f"mv /cache/rerank /cache/output/")
    
    mox.file.copy_parallel('/cache/output', s3_output_path) 





if __name__ == '__main__':
    main()

# --deepspeed /home/work/user-job-dir/FraBert/pretrain/ds_config.json \
# python /home/work/user-job-dir/FraBert/pretrain/run_mlm_my.py \
#       --config_name /home/work/user-job-dir/FraBert/bert_base_uncased \
#       --train_file /cache/test.json \
#       --do_train \
#       --per_device_train_batch_size 20 \
#       --num_train_epochs 5 \
#       --dataloader_num_workers 8 \
#       --save_steps 50000 \
#       --output_dir /home/work/user-job-dir/FraBert/FraBert_scrach \
#       --dataset_script_dir /home/work/user-job-dir/FraBert/data_scripts \
#       --dataset_cache_dir /home/work/user-job-dir/FraBert/cache \
#       --limit 50000000 \
#       --overwrite_output_dir \
#       --tokenizer_name /home/work/user-job-dir/FraBert/bert_base_uncased \
# --model_name_or_path /home/yu_guo/huggingface_transformers/examples/pytorch/language-modeling/html_output_cut

