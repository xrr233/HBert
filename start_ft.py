# python run_mlm_my.py \
# 		--config_name /home/yu_guo/huggingface_transformers/examples/pytorch/language-modeling/bert_base_uncased \
# 		--train_file /home/yu_guo/DataPreProcess/data/hirachical_html_data_revised/test_100W.json \
# 		--do_train \
# 		--per_device_train_batch_size 1 \
# 		--num_train_epochs 1 \
# 		--dataloader_num_workers 8 \
# 		--save_steps 100000 \
# 		--output_dir /home/yu_guo/huggingface_transformers/examples/pytorch/language-modeling/HirachicalBert_scrach \
# 		--dataset_script_dir /home/yu_guo/huggingface_transformers/examples/pytorch/language-modeling/data_scripts \
# 		--dataset_cache_dir /home/yu_guo/huggingface_transformers/examples/pytorch/language-modeling/cache \
# 		--limit 50000000 \
# 		--overwrite_output_dir \
# 		--tokenizer_name /home/yu_guo/huggingface_transformers/examples/pytorch/language-modeling/bert_base_uncased \
# --model_name_or_path /home/yu_guo/huggingface_transformers/examples/pytorch/language-modeling/html_output_cut
import os
import sys
import torch
import moxing as mox
os.system('pip install transformers')
os.system( 'mv /home/work/user-job-dir/Pretrain/training_args.py /root/miniconda3/lib/python3.6/site-packages/transformers/training_args.py')
os.system( 'mv /home/work/user-job-dir/Pretrain/hf_argparser.py /root/miniconda3/lib/python3.6/site-packages/transformers/hf_argparser.py')

os.system('pip install datasets')
INIT_METHOD = "tcp://" + os.environ.get( 'BATCH_CUSTOM0_HOSTS')  # BATCH_CUSTOM0_HOSTS 为选定的init机器地址和端口号，需手动加入tcp://即可等价于init_method
WORLD_SIZE = os.environ.get('DLS_TASK_NUMBER')  # DLS_TASK_NUMBER 等价于初始 world_size
RANK = os.environ.get('DLS_TASK_INDEX')  # DLS_TASK_INDEX 等价于初始 rank
WORLD_SIZES = 32
mox.file.copy_parallel("s3://obs-app-2020042019121301221/SEaaKM/g50020960/datasets/HBert/","/cache/")
os.system(f'python -m torch.distributed.launch  --nproc_per_node=8 --nnodes=4 --node_rank={int(RANK)} /home/work/user-job-dir/Pretrain/run_mlm_ft.py \
		--fp16 \
		--world_sizes {WORLD_SIZES} \
		--init_method {INIT_METHOD} --rank {int(RANK)} \
		--model_name_or_path /home/work/user-job-dir/Pretrain/HBert/FraBert_scrach \
		--node_config_name /home/work/user-job-dir/Pretrain/node_bert/bert_base_uncased \
		--train_file /cache/epoch_200W.json \
		--do_train \
		--ignore_data_skip \
		--learning_rate 5e-5 \
		--per_device_train_batch_size 2 \
		--num_train_epochs 2 \
		--dataloader_num_workers 8 \
		--save_steps 10000 \
		--output_dir /home/work/user-job-dir/Pretrain/HBert_fine_tune \
		--dataset_script_dir /home/work/user-job-dir/Pretrain/data_scripts \
		--dataset_cache_dir /home/work/user-job-dir/Pretrain/cache \
		--limit 50000000 \
		--ddp_find_unused_parameters True \
		--overwrite_output_dir ')
mox.file.copy_parallel("/home/work/user-job-dir/Pretrain/HBert_fine_tune/","s3://obs-app-2020042019121301221/SEaaKM/g50020960/code/Pretrain/HBert_fine_tune/")
# --deepspeed /home/work/user-job-dir/FraBert/pretrain/ds_config.json \
# python /home/work/user-job-dir/FraBert/pretrain/run_mlm_my.py \
#		--config_name /home/work/user-job-dir/FraBert/bert_base_uncased \
#		--train_file /cache/test.json \
#		--do_train \
#		--per_device_train_batch_size 20 \
#		--num_train_epochs 5 \
#		--dataloader_num_workers 8 \
#		--save_steps 50000 \
#		--output_dir /home/work/user-job-dir/FraBert/FraBert_scrach \
#		--dataset_script_dir /home/work/user-job-dir/FraBert/data_scripts \
#		--dataset_cache_dir /home/work/user-job-dir/FraBert/cache \
#		--limit 50000000 \
#		--overwrite_output_dir \
#		--tokenizer_name /home/work/user-job-dir/FraBert/bert_base_uncased \
# --model_name_or_path /home/yu_guo/huggingface_transformers/examples/pytorch/language-modeling/html_output_cut

