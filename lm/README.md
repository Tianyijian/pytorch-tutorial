# BERT继续预训练-标准版

在BERT模型的基础上继续预训练，pytorch版，使用 [huggingface/transformers](https://github.com/huggingface/transformers)。

先准备数据 `sh pre_train.sh`，再进行训练 `sh pt_train.sh`。

## 说明

使用 [huggingface/transformers v0.6.2 lm_finetuning](https://github.com/huggingface/transformers/tree/v0.6.2/examples/lm_finetuning)

- pregenerate_training_data.py：代码289行添加ensure_ascii=False，使输出的json文件中文显示正常
- finetune_on_pregenerated.py：无改动
- pre_data.sh：准备数据的执行命令
- pt_train.sh：训练的执行命令
- pytorch_pretrained_bert/ ：transformers v0.6.2 依赖代码
- pt_data/：含示例数据pt_train.txt，**一行一个句子，文档间空行分割**。

## 

