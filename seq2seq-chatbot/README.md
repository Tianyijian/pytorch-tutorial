# 聊天机器人

使用 Encoder-Attention-Decoder 模型训练一个简单的聊天机器人。

## 数据集

下载 [Cornell Movie-Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)，放入到`data/` 文件夹下。

## 目录

- data (数据集，需下载)
- data.py (加载与预处理数据)
- model.py (构造seq2seq网络)
- run.py (进行训练以及测试)

## 说明

- 运行`data.py`来格式化数据文件

- 运行`python run.py --mode train`进行训练

- 运行`python run.py --mode test`开始聊天 :

  ```
  $ python run.py --mode test
  Ready! Start Chatting!
  > hello ?
  Bot: hello .
  > how are you
  Bot: i m fine .
  > bye
  Bot: bye .
  ```
  

## 学习总结

- Encoder-Attention-Decoder模型的Pytorch实现
- Cornell Movie-Dialogs Corpus 数据集的格式化处理

## 参考

[Pytorch官方教程中文版-聊天机器人](http://pytorch123.com/FifthSection/Chatbot/)