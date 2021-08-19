import os
import sys
import torch
import moxing as mox
os.system('pip install transformers')

mox.file.copy_parallel("s3://obs-app-2020042019121301221/SEaaKM/g50020960/datasets","/cache")

os.system(
          f' python /home/work/user-job-dir/Pretrain/Preprocess/gen_data.py  \
          --train_corpus /cache/wiki_html_all.json \
          --do_lower_case  \
          --output_dir /home/work/user-job-dir/Pretrain/HBert   \
          --epochs_to_generate 1      \
          --mlm     \
          --reduce_memory \
          --num_workers 40 ')

mox.file.copy_parallel('/home/work/user-job-dir/Pretrain/HBert',"s3://obs-app-2020042019121301221/SEaaKM/g50020960/Pretrain/HBert")
# INIT_METHOD = "tcp://" + os.environ.get( 'BATCH_CUSTOM0_HOSTS')  # BATCH_CUSTOM0_HOSTS 为选定的init机器地址和端口号，需手动加入tcp://即可等价于init_method
# WORLD_SIZE = os.environ.get('DLS_TASK_NUMBER')  # DLS_TASK_NUMBER 等价于初始 world_size
# RANK = os.environ.get('DLS_TASK_INDEX')  # DLS_TASK_INDEX 等价于初始 rank
# WORLD_SIZES = 32
# os.system('python /home/work/user-job-dir/FraBert/pretrain/download_data.py')

# os.system(f'python -m torch.distributed.launch  --nproc_per_node=8 --nnodes=4 --node_rank={int(RANK)} /home/work/user-job-dir/FraBert/pretrain/run_mlm_my.py \
# 		--config_name /home/work/user-job-dir/FraBert/bert_base_uncased \
# 		--world_sizes {WORLD_SIZES} \
# 		--init_method {INIT_METHOD} --rank {int(RANK)} \
# 		--train_file /cache/FraBert.json \
# 		--do_train \
# 		--per_device_train_batch_size 20 \
# 		--num_train_epochs 1 \
# 		--dataloader_num_workers 8 \
# 		--save_steps 50000 \
# 		--output_dir /home/work/user-job-dir/FraBert/FraBert_scrach \
# 		--dataset_script_dir /home/work/user-job-dir/FraBert/data_scripts \
# 		--dataset_cache_dir /home/work/user-job-dir/FraBert/cache \
# 		--limit 50000000 \
# 		--ddp_find_unused_parameters True \
# 		--overwrite_output_dir \
# 		--tokenizer_name /home/work/user-job-dir/FraBert/bert_base_uncased')

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

