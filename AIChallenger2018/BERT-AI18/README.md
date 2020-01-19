# BERT-AI18

使用BERT进行ABSA

### 数据集
AIChallenger2018 数据集，多属性情感分类任务（ABSA）
**每个属性训练一个模型**，进行情感四分类：积极、中性、消极、未提及。

### 实验结果

##### 参数设置

```
BASELINE：max_seq_length=512，learning_rate=1e-05，weight_decay=0.001，gradient_accumulation_steps=2，per_gpu_train_batch_size=8，per_gpu_eval_batch_size=32，num_train_epochs=1.0,
```

1. baseline 设置
2. 在bert-base-Chinese模型的基础上，使用训练集继续预训练模型
3. 模型如2，num_train_epochs=2.0

|      | Aspect0~2 |
| :--: | :-------: |
|  1   |  0.6404   |
|  2   |  0.6337   |
|  3   |  0.6494   |

