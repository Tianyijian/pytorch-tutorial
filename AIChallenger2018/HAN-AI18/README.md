# HAN-AI18

ABSA任务，使用[Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)

### 数据集
AIChallenger2018 数据集，多属性情感分类任务（ABSA）。
由于文本长度较大，**将每条文本视为一个文档**，利用文档级别的分层注意力网络进行情感分类。
**每个属性训练一个模型**，进行情感四分类：积极、中性、消极、未提及。

### 模型实现

参考：[Github: sgrvinod/a-PyTorch-Tutorial-to-Text-Classification](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Text-Classification)

在此基础上进行了以下修改：

- 读取中文数据，**正则实现中文分句**，jieba 中文分词
- 训练过程中在测试集上做评价，记录最佳宏平均结果
- 在每个细粒度属性上单独训练模型，取各属性最佳结果的平均值
- 使用tensorboard进行loss、acc 等指标可视化

### 实验结果

##### 参数设置

1. 交叉熵损失，epoch=8，无三角学习率与L2正则化
2. FocalLoss，epoch=8，三角学习率，weight_decay=0.001
3. FocalLoss，epoch=12，无三角学习率与L2正则化
4. FocalLoss，epoch=12，无三角学习率，weight_decay=0.001
5. FocalLoss，epoch=12，无三角学习率，weight_decay=0.00001
6. FocalLoss，epoch=12，无三角学习率，weight_decay=0.00001

|      | Aspect0~2  | Aspect0~4  | Aspect0~19 |
| :--: | :--------: | :--------: | :--------: |
|  1   |   0.6132   |   0.6353   |   0.6447   |
|  2   |   0.5406   |   0.5685   |   0.5747   |
|  3   |   0.6210   |   0.6380   |            |
|  4   |   0.5509   |   0.5777   |            |
|  5   | **0.6353** | **0.6490** |            |
|  6   |   0.6308   |   0.6476   | **0.6519** |

##### 实验总结

- FocalLoss可一定程度上缓解样本不均衡的问题，提高模型性能。
- L2正则化可提高模型性能，但取值不能太大，否则会导致负作用，并且收敛较慢。

