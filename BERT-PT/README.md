# BERT继续预训练-简单版

在BERT模型的基础上继续预训练，pytorch版，使用 [huggingface/transformers](https://github.com/huggingface/transformers)。

**注意：**此代码为简单版，准备数据时未严格按照BERT的NSP（Next Sentence Prediction）任务。

## 依赖

transformers 2.3.0

## 数据集

[WikiText-2 dataset](https://blog.einstein.ai/the-wikitext-long-term-dependency-language-modeling-dataset/)

## 目录

- run_lm_finetuning.py：参考官方代码，基本无改动
- run_pt.sh：运行脚本

## 参考

https://github.com/huggingface/transformers/tree/master/examples#language-model-fine-tuning

https://github.com/huggingface/transformers/blob/master/examples/run_lm_finetuning.py