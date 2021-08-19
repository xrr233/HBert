
python gen_data_addpid.py  --train_corpus /home/dou/replearn/anchors/data/sent_anchors_output_1w.txt \
    --bert_model /home/dou/replearn/transformers_models/bert    --do_lower_case   \
   --output_dir /home/dou/replearn/anchors/data     --epochs_to_generate 1     --rop_num_per_doc 4   \
    --mlm     --max_seq_len 512     --reduce_memory

# task 1:
python gen_task1_weights.py \
--per_gpu_batch_size 2 --per_gpu_test_batch_size 2 --task bertgen \
--model_path /home/dou/replearn/transformers_models/bert \
--passage_file_path /home/dou/replearn/anchors/data/anchor_passages.txt \
--dataset_script_dir /home/dou/replearn/anchors/data_scripts \
--dataset_cache_dir /tmp/negs_cache \
--log_path ./log.txt \
--output_file_path /home/dou/replearn/anchors/output/passage_cls_weight.txt 

python gen_task1_pairdata.py  --train_corpus /home/dou/replearn/anchors/data/anchor_corpus.addid.txt \
    --bert_model  /home/dou/replearn/transformers_models/bert --passage_cls_weight_file  /home/dou/replearn/anchors/output/passage_cls_weight.txt \
    --do_lower_case  --output_dir /home/dou/replearn/anchors/output/task1     --epochs_to_generate 1     --rop_num_per_doc 4   \
    --mlm     --max_seq_len 512     --reduce_memory --num_workers 2 --stop_words_file /home/dou/replearn/anchors/data/stopwords.txt

# task2:

python gen_task2_filter.py --passage_file /home/dou/replearn/anchors/data/anchor_passages.txt \
--sentence_file /home/dou/replearn/anchors/data/anchor_sentences.txt --sap_file /home/dou/replearn/anchors/data/anchor_sap_triples.txt \
--anchor_file /home/dou/replearn/anchors/data/anchor_anchors.txt --bert_model /home/dou/replearn/transformers_models/bert \
 --output_file /home/dou/replearn/anchors/output/task2/task2_sentence_passages.txt

python gen_task2_weights.py \
--per_gpu_batch_size 2 --per_gpu_test_batch_size 2 --task bertgen \
--model_path /home/dou/replearn/transformers_models/bert \
--sentence_file_path /home/dou/replearn/anchors/output/task2/task2_sentence_passages.txt \
--dataset_script_dir /home/dou/replearn/anchors/data_scripts \
--dataset_cache_dir /tmp/negs_cache \
--log_path ./log.txt \
--output_file_path /home/dou/replearn/anchors/output/sentence_cls_weight.txt 

python gen_task2_pairdata.py --bert_model  /home/dou/replearn/transformers_models/bert \
    --do_lower_case  --output_dir /home/dou/replearn/anchors/output/task2     --epochs_to_generate 1     --rop_num_per_doc 4   \
    --mlm     --max_seq_len 512     --reduce_memory --num_workers 1 --stop_words_file /home/dou/replearn/anchors/data/stopwords.txt \
    --anchor_file /home/dou/replearn/anchors/data/anchor_anchors.txt --passage_file /home/dou/replearn/anchors/data/anchor_passages.txt \
    --sentence_cls_weight_file /home/dou/replearn/anchors/output/sentence_cls_weight.txt

# task3:

python gen_task3_pairdata.py --passage_file /home/dou/replearn/anchors/data/anchor_passages.txt \
--sentence_file /home/dou/replearn/anchors/data/anchor_sentences.txt --sap_file /home/dou/replearn/anchors/data/anchor_sap_triples.txt \
--anchor_file /home/dou/replearn/anchors/data/anchor_anchors.txt --bert_model /home/dou/replearn/transformers_models/bert \
--max_pair_perquery 10 --max_seq_len 512 --output_dir /home/dou/replearn/anchors/output/task3 --mlm 


# task4

python gen_task4_filter.py --passage_file /home/dou/replearn/anchors/data/anchor_passages.txt \
--sentence_file /home/dou/replearn/anchors/data/anchor_sentences.txt --sap_file /home/dou/replearn/anchors/data/anchor_sap_triples.txt \
--anchor_file /home/dou/replearn/anchors/data/anchor_anchors.txt --bert_model /home/dou/replearn/transformers_models/bert \
 --output_file /home/dou/replearn/anchors/output/task4/task4_sentence_passages.txt


python gen_task4_pairdata.py --bert_model  /home/dou/replearn/transformers_models/bert \
    --do_lower_case  --output_dir /home/dou/replearn/anchors/output/task4     --epochs_to_generate 1     --rop_num_per_doc 4   \
    --mlm     --max_seq_len 512     --reduce_memory --num_workers 1 --stop_words_file /home/dou/replearn/anchors/data/stopwords.txt \
    --anchor_file /home/dou/replearn/anchors/data/anchor_anchors.txt --passage_file /home/dou/replearn/anchors/data/anchor_passages.txt \
    --sentence_file_path /home/dou/replearn/anchors/output/task4/task4_sentence_passages.txt 

# gen baseline data

for i in $(seq -f "%03g" 0 10)
do
python gen_baseline_point_datas.py \
    --tokenizer_name /home/dou/replearn/transformers_models/bert \
    --rank_file /home/dou/replearn/data/lce_data/ance-marco-train-openmatch-100/${i}.txt \
    --json_dir /home/dou/replearn/anchors/output/finetune/processed \
    --n_sample 10 \
    --sample_from_top 100 \
    --random \
    --truncate 512 \
    --qrel /home/dou/msmarco/msmarco-doctrain-qrels.tsv.gz \
    --query_collection /home/dou/msmarco/msmarco-doctrain-queries.tsv \
    --doc_collection /home/dou/msmarco/msmarco-docs.tsv
done

# baseline train full

python runBert.py \
--is_training \
--per_gpu_batch_size 8 --per_gpu_test_batch_size 128 --task msmarco \
--bert_model /home/dou/replearn/transformers_models/bert \
--dataset_script_dir /home/dou/replearn/anchors/data_scripts \
--dataset_cache_dir /tmp/negs_cache \
--log_path ./log.txt \
--train_file /home/dou/replearn/anchors/output/finetune/processed \
--dev_file  /home/dou/replearn/anchors/output/finetune/processed_dev_519queries/all.json \
--dev_id_file /home/dou/replearn/anchors/output/finetune/processed_dev_519queries/ids.tsv \
--msmarco_score_file_path /home/dou/replearn/anchors/output/finetune/score.txt \
--msmarco_dev_qrel_path /home/dou/replearn/data/msmarco/msmarco-docdev-qrels.tsv \
--save_path /home/dou/replearn/anchors/output/finetune/output_bert/pytorch_model.bin

# baseline train sample

python runBert.py \
--is_training \
--per_gpu_batch_size 8 --per_gpu_test_batch_size 128 --task msmarco \
--bert_model /home/dou/replearn/transformers_models/prop \
--dataset_script_dir /home/dou/replearn/anchors/data_scripts \
--dataset_cache_dir /tmp/negs_cache \
--log_path ./log.txt \
--train_file /home/dou/replearn/anchors/output/finetune/processed_sample \
--dev_file  /home/dou/replearn/anchors/output/finetune/processed_dev_51queries/all.json \
--dev_id_file /home/dou/replearn/anchors/output/finetune/processed_dev_51queries/ids.tsv \
--msmarco_score_file_path /home/dou/replearn/anchors/output/finetune/score.txt \
--msmarco_dev_qrel_path /home/dou/replearn/data/msmarco/msmarco-docdev-qrels.tsv \
--save_path /home/dou/replearn/anchors/output/finetune/output_bert/pytorch_model.bin

# baseline test

python runBert.py \
--per_gpu_batch_size 8 --per_gpu_test_batch_size 128 --task msmarco \
--bert_model /home/dou/replearn/transformers_models/bert \
--dataset_script_dir /home/dou/replearn/anchors/data_scripts \
--dataset_cache_dir /tmp/negs_cache \
--log_path ./log.txt \
--train_file /home/dou/replearn/anchors/output/finetune/processed_sample \
--dev_file  /home/dou/replearn/anchors/output/finetune/processed_dev_sample/all.json \
--dev_id_file /home/dou/replearn/anchors/output/finetune/processed_dev_sample/ids.tsv \
--msmarco_score_file_path /home/dou/replearn/anchors/output/finetune/score.txt \
--msmarco_dev_qrel_path /home/dou/replearn/data/msmarco/msmarco-docdev-qrels.tsv \
--save_path /home/dou/replearn/anchors/output/finetune/output_bert/pytorch_model.bin
