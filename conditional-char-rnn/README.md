# 使用字符级RNN生成名字

根据语言生成名字，使用只有几层线性层的小型RNN。输入一个类别之后在每一时刻 输出一个字母。循环预测字符以形成语言通常也被称为“语言模型”。

## 数据集

点击[这里](https://download.pytorch.org/tutorial/data.zip)下载数据，在"`data/names`"文件夹下是名称为"[language].txt"的18个文本文件。每个文件的每一行都有一个名字，它们几乎都是罗马化的文本 （但是我们仍需要将其从Unicode转换为ASCII编码）。

## 目录

- data (数据集)
- data.py (读取文件)
- model.py (构造RNN网络)
- train.py (运行训练过程)
- generate.py (生成名字)
- conditional-char-rnn.pt (保存的模型)

## 说明

- 运行`train.py`来训练和保存网络

- 将`generate.py`和一种语言一起运行查看生成的名字 :

  ```
  $ python generate.py English
  Aller
  Bart
  Chel
  ```
  

## 学习总结

- 用RNN进行生成的pytorch简单实现
- Pytorch使用GPU进行训练

## 参考

[Pytorch官方教程中文版-使用字符级RNN生成名字](http://pytorch123.com/FifthSection/CharRNNGeneration/)

[Github: spro/practical-pytorch](https://github.com/spro/practical-pytorch/blob/master/conditional-char-rnn/conditional-char-rnn.ipynb)