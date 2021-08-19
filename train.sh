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
		#--model_name_or_path /home/yu_guo/huggingface_transformers/examples/pytorch/language-modeling/html_output_cut
python run_mlm_my.py --config_name bert_base_uncased --node_config_name node_bert/bert_base_uncased --train_file HBert/epoch_0.json --do_train --per_device_train_batch_size 1  --num_train_epochs 1 --dataloader_num_workers 8 --save_steps 1000 --output_dir experiment --dataset_script_dir data_scripts --dataset_cache_dir cache --limit 50000000 --overwrite_output_dir --tokenizer_name bert_base_uncased

python run_mlm_my.py 	--text_config_name bert_base_uncased 	--node_config_name node_bert/bert_base_uncased 	--train_file HBert/test.json 		--do_train 		--per_device_train_batch_size 1 		--num_train_epochs 1 		--dataloader_num_workers 8 		--save_steps 1000 		--output_dir experiment 		--dataset_script_dir data_scripts 		--dataset_cache_dir cache 		--limit 50000000 		--overwrite_output_dir 		--tokenizer_name bert_base_uncased
		    #--deepspeed /home/work/user-job-dir/FraBert/pretrain/ds_config.json \
#python /home/work/user-job-dir/FraBert/pretrain/run_mlm_my.py \
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
		#--model_name_or_path /home/yu_guo/huggingface_transformers/examples/pytorch/language-modeling/html_output_cut 