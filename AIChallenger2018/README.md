# AIChallenger2018

### 数据集
AIChallenger2018 数据集，多属性中文情感分类任务（ABSA）。每个属性进行情感四分类：积极、中性、消极、未提及。详细信息可参照数据说明文档。

### 说明

- data：数据文件夹
- HAN-AI18：使用[Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)
- BERT-AI18：使用BERT

### 实验

##### 参数设置
1. Hierarchical Attention Networks
2. bert-base-chinese
3. bert 使用领域数据继续预训练模型

|      | Aspect 0~2 |
| :--: | :--------: |
|  1   |   0.6353   |
|  2   |   0.6404   |
|  3   |   0.6494   |

##### 实验总结

- bert 使用领域数据继续预训练模型可以提高性能

### 冠军分享

- [代码](https://github.com/chenghuige/wenzheng/tree/master/projects/ai2018/sentiment)

- [PPT](https://mp.weixin.qq.com/s/W0PhbE8149nD3Venmy33tw)