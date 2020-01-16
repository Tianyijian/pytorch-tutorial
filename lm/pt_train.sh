export CUDA_VISIBLE_DEVICES=0
export TRAIN_FILE=/users8/yjtian/tyj/demo/lm/pt_data/
export BERT_PATH=/users8/yjtian/tyj/LM/BERT-torch/bert-base-chinese

python finetune_on_pregenerated.py \
--pregenerated_data $TRAIN_FILE \
--bert_model $BERT_PATH \
--output_dir finetuned_lm/ \
--epochs 3 \
--train_batch_size 32 \
--gradient_accumulation_steps 1
