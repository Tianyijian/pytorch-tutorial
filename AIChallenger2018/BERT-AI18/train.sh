export CUDA_VISIBLE_DEVICES=2
export BERT_PATH="/users8/yjtian/tyj/demo/lm/finetuned_lm"
export DATA_DIR="/users8/yjtian/tyj/demo/HAN2/data2"
export OUT_DIR="/users8/yjtian/tyj/demo/BERT-AI18/output"
export TASK_NAME=ai18

python run_ai18.py \
    --model_type bert \
    --model_name_or_path $BERT_PATH \
    --task_name $TASK_NAME \
    --do_train \
    --data_dir $DATA_DIR \
    --max_seq_length 512 \
    --per_gpu_eval_batch_size=32   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 1e-5 \
    --num_train_epochs 2 \
    --output_dir $OUT_DIR \
    --evaluate_during_training \
    --save_steps -1 \
    --logging_steps 200 \
    --max_steps -1 \
    --warmup_proportion 0.1 \
    --weight_decay 0.001 \
    --gradient_accumulation_steps 2 \
    --seed 100 \
    --fp16
