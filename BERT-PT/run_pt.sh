export CUDA_VISIBLE_DEVICES=2
export TRAIN_FILE=/users5/yjtian/tyj/LM/BERT-PT/wikitext-2-raw/wiki.train.raw
export TEST_FILE=/users5/yjtian/tyj/LM/BERT-PT/wikitext-2-raw/wiki.test.raw
export BERT_MODEL=/users5/yjtian/tyj/LM/BERT-torch/bert-base-uncased
export OUTPUT_DIR=/users5/yjtian/tyj/LM/BERT-PT/BERT-PT

python run_lm_finetuning.py \
    --output_dir=output \
    --model_type=bert \
    --model_name_or_path=$BERT_MODEL \
    --do_train \
    --train_data_file=$TRAIN_FILE \
    --do_eval \
    --eval_data_file=$TEST_FILE \
    --mlm \
    --block_size 510 \
    --do_lower_case \
    --save_steps -1 \
    --per_gpu_train_batch_size 8