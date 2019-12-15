# 使用字符级RNN进行名字分类

构建和训练字符级RNN来对单词进行分类。字符级RNN将单词作为一系列字符读取，在每一步输出预测和“隐藏状态”，将其先前的隐藏 状态输入至下一时刻。我们将最终时刻输出作为预测结果，即表示该词属于哪个类。

## 数据集

点击[这里](https://download.pytorch.org/tutorial/data.zip)下载数据，在"`data/names`"文件夹下是名称为"[language].txt"的18个文本文件。每个文件的每一行都有一个名字，它们几乎都是罗马化的文本 （但是我们仍需要将其从Unicode转换为ASCII编码）。

## 目录

- data (数据集)
- data.py (读取文件)
- model.py (构造RNN网络)
- train.py (运行训练过程)
- predict.py (在命令行中和参数一起运行predict()函数)
- server.py (使用bottle.py构建JSON API的预测服务)
- char-rnn-classification.pt (保存的模型)

## 说明

- 运行`train.py`来训练和保存网络

- 将`predict.py`和一个名字的单词一起运行查看预测结果 :

  ```
  $ python predict.py Hazaki
  (-0.42) Japanese
  (-1.39) Polish
  (-3.51) Czech
  ```
- 运行`server.py`并访问http://localhost:5533/Yourname 得到JSON格式的预测输出

## 参考

[Pytorch官方教程中文版-使用字符级RNN进行名字分类](http://pytorch123.com/FifthSection/CharRNNClassification/)