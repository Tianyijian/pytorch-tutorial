export TRAIN_FILE=/users8/yjtian/tyj/demo/lm/pt_data/pt_train.txt
export OUTPUT=/users8/yjtian/tyj/demo/lm/pt_data

python pregenerate_training_data.py \
--train_corpus $TRAIN_FILE \
--bert_model bert-base-chinese \
--output_dir $OUTPUT \
--epochs_to_generate 3 \
--max_seq_len 128