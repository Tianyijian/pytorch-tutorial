# BERT-AI18

使用BERT进行ABSA，Pytorch版，使用[huggingface/transformers v2.3.0](https://github.com/huggingface/transformers)

### 数据集
AIChallenger2018 数据集，多属性情感分类任务（ABSA）。

**每个属性训练一个模型**，进行情感四分类：积极、中性、消极、未提及。

### 实验结果

##### 参数设置

BASELINE：

```
max_seq_length=512，learning_rate=1e-05，weight_decay=0.001，per_gpu_train_batch_size=8，gradient_accumulation_steps=2，per_gpu_eval_batch_size=32，num_train_epochs=1.0
```

1. baseline 设置
2. baseline设置，在 bert-base-Chinese 模型的基础上，使用训练集继续预训练模型，参考 [pytorch-turorial/lm](https://github.com/Tianyijian/pytorch-tutorial/tree/master/lm)
3. baseline设置，使用模型如2，num_train_epochs=2.0

|      | Aspect 0~2 |
| :--: | :--------: |
|  1   |   0.6404   |
|  2   |   0.6337   |
|  3   |   0.6494   |

##### 实验总结

- 在bert-base模型的基础上，使用领域数据继续预训练，然后 fine-tuning，可以提高性能

