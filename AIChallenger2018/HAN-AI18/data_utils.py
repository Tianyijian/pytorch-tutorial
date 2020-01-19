import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from collections import Counter
import jieba
from tqdm import tqdm
import pandas as pd
import itertools
import os
import json
import gensim
import logging
import time
import re


def read_csv(csv_folder, split, sentence_limit, word_limit):
    """
    Read CSVs containing raw training data, clean documents and labels, and do a word-count.

    :param csv_folder: folder containing the CSV
    :param split: train or test CSV?
    :param sentence_limit: truncate long documents to these many sentences
    :param word_limit: truncate long sentences to these many words
    :return: documents, labels, a word-count
    """
    assert split in {'sentiment_analysis_trainingset',
                     'sentiment_analysis_validationset'}

    docs = []
    labels = []
    ids = []
    word_counter = Counter()
    data = pd.read_csv(os.path.join(csv_folder, split + '.csv'))
    sent_num = 0
    s_l_20 = 0
    word_num = 0
    w_l_60 = 0
    for i in tqdm(range(data.shape[0])):
    # for i in range(1000):
        row = list(data.loc[i, :])

        # sentences = cut_sent(row[1])
        sentences = splitsentence(row[1].replace("\n", ""))
        sent_num += len(sentences)
        if len(sentences) < 20:
            s_l_20 += 1

        words = list()
        for s in sentences[:sentence_limit]:
            # for s in sentences:
            w = list(jieba.cut(s))
            word_num += len(w)
            if len(w) < 60:
                w_l_60 += 1

            w = w[:word_limit]
            # If sentence is empty (due to removing punctuation, digits, etc.)
            if len(w) == 0:
                continue
            words.append(w)
            word_counter.update(w)

        # If all sentences were empty
        if len(words) == 0:
            continue

        labels.append([int(v) + 2 for v in row[2:]])
        ids.append(row[0])
        docs.append(words)

    print("sent_num:{}, avg_sent:{}, s<20:{}, word_num:{}, avg_word:{}, w<60:{}".format(sent_num, float(sent_num) /
                                                                                        data.shape[0], float(s_l_20) / data.shape[0], word_num, float(word_num) / sent_num, float(w_l_60) / sent_num))
    return ids, docs, labels, word_counter


def cut_sent(para):
    """
    https://blog.csdn.net/blmoistawinde/article/details/82379256
    :param para:
    :return:
    """
    para = re.sub(r'([。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub(r'(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub(r'(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub(r'([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")


resentencesp = re.compile('([﹒﹔﹖﹗．；。！？]["’”」』]{0,2}|：(?=["‘“「『]{1,2}|$))')


def splitsentence(sentence):
    """
    https://github.com/fxsjy/jieba/issues/575
    :param sentence:
    :return:
    """
    s = sentence
    slist = []
    for i in resentencesp.split(s):
        if resentencesp.match(i) and slist:
            slist[-1] += i
        elif i:
            slist.append(i)
    return slist


def create_input_files(csv_folder, output_folder, sentence_limit, word_limit, min_word_count=5):
    """
    Create data files to be used for training the model.

    :param csv_folder: folder where the CSVs with the raw data are located
    :param output_folder: folder where files must be created
    :param sentence_limit: truncate long documents to these many sentences
    :param word_limit: truncate long sentences to these many words
    :param min_word_count: discard rare words which occur fewer times than this number
    """
    # Read training data
    print('\nReading and preprocessing training data...\n')
    train_ids, train_docs, train_labels, word_counter = read_csv(
        csv_folder, 'sentiment_analysis_trainingset', sentence_limit, word_limit)

    # Create word map
    word_map = dict()
    word_map['<pad>'] = 0
    for word, count in word_counter.items():
        if count >= min_word_count:
            word_map[word] = len(word_map)
    word_map['<unk>'] = len(word_map)
    print('\nDiscarding words with counts less than %d, the size of the vocabulary is %d.\n' % (
        min_word_count, len(word_map)))

    with open(os.path.join(output_folder, 'word_map.json'), 'w', encoding="utf-8") as j:
        json.dump(word_map, j, ensure_ascii=False, indent=4)
    print('Word map saved to %s.\n' % os.path.abspath(output_folder))

    # Encode and pad
    print('Encoding and padding training data...\n')
    encoded_train_docs = list(map(lambda doc: list(
        map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) + [0] * (word_limit - len(s)),
            doc)) + [[0] * word_limit] * (sentence_limit - len(doc)), train_docs))
    sentences_per_train_document = list(map(lambda doc: len(doc), train_docs))
    words_per_train_sentence = list(
        map(lambda doc: list(map(lambda s: len(s), doc)) + [0] * (sentence_limit - len(doc)), train_docs))

    # Save
    print('Saving...\n')
    assert len(encoded_train_docs) == len(train_labels) == len(sentences_per_train_document) == len(
        words_per_train_sentence)
    # Because of the large data, saving as a JSON can be very slow

    torch.save({'ids': train_ids,
                'docs': encoded_train_docs,
                'labels': train_labels,
                'sentences_per_document': sentences_per_train_document,
                'words_per_sentence': words_per_train_sentence},
               os.path.join(output_folder, 'TRAIN_data.tar'))
    print('Encoded, padded training data saved to %s.\n' %
          os.path.abspath(output_folder))

    # Free some memory
    del train_docs, encoded_train_docs, train_labels, sentences_per_train_document, words_per_train_sentence

    # Read test data
    print('Reading and preprocessing test data...\n')
    test_ids, test_docs, test_labels, _ = read_csv(
        csv_folder, 'sentiment_analysis_validationset', sentence_limit, word_limit)

    # Encode and pad
    print('\nEncoding and padding test data...\n')
    encoded_test_docs = list(map(lambda doc: list(
        map(lambda s: list(map(lambda w: word_map.get(w, word_map['<unk>']), s)) + [0] * (word_limit - len(s)),
            doc)) + [[0] * word_limit] * (sentence_limit - len(doc)), test_docs))
    sentences_per_test_document = list(map(lambda doc: len(doc), test_docs))
    words_per_test_sentence = list(
        map(lambda doc: list(map(lambda s: len(s), doc)) + [0] * (sentence_limit - len(doc)), test_docs))

    # Save
    print('Saving...\n')
    assert len(encoded_test_docs) == len(test_labels) == len(sentences_per_test_document) == len(
        words_per_test_sentence)
    torch.save({'ids': test_ids,
                'docs': encoded_test_docs,
                'labels': test_labels,
                'sentences_per_document': sentences_per_test_document,
                'words_per_sentence': words_per_test_sentence},
               os.path.join(output_folder, 'TEST_data.tar'))
    print('Encoded, padded test data saved to %s.\n' %
          os.path.abspath(output_folder))

    print('All done!\n')


def init_embedding(input_embedding):
    """
    Initialize embedding tensor with values from the uniform distribution.

    :param input_embedding: embedding tensor
    """
    bias = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -bias, bias)


def load_word2vec_embeddings(word2vec_file, word_map):
    """
    Load pre-trained embeddings for words in the word map.

    :param word2vec_file: location of the trained word2vec model
    :param word_map: word map
    :return: embeddings for words in the word map, embedding size
    """
    # Load word2vec model into memory
    w2v = gensim.models.KeyedVectors.load(word2vec_file, mmap='r')

    print("\nEmbedding length is %d.\n" % w2v.vector_size)

    # Create tensor to hold embeddings for words that are in-corpus
    embeddings = torch.FloatTensor(len(word_map), w2v.vector_size)
    init_embedding(embeddings)

    # Read embedding file
    print("Loading embeddings...")
    cnt = 0
    for word in word_map:
        if word in w2v.vocab:
            embeddings[word_map[word]] = torch.FloatTensor(w2v[word])
            cnt += 1
    print("Embedding vocabulary: %d, in w2v: %d(%.4f).\n" %
          (len(word_map), cnt, float(cnt) / len(word_map)))

    return embeddings, w2v.vector_size


def clip_gradient(optimizer, grad_clip):
    """
    Clip gradients computed during backpropagation to prevent gradient explosion.

    :param optimizer: optimized with the gradients to be clipped
    :param grad_clip: gradient clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(epoch, model, optimizer, word_map):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    :param best_acc: best accuracy achieved so far (not necessarily in this checkpoint)
    :param word_map: word map
    :param epochs_since_improvement: number of epochs since last improvement
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer,
             'word_map': word_map}
    filename = './logs/checkpoint_{}.pth.tar'.format(epoch)
    torch.save(state, filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, scale_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rates must be decayed
    :param scale_factor: factor to scale by
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


class WarmupLinearSchedule(LambdaLR):
    """ Linear warmup and then linear decay.
        Linearly increases learning rate from 0 to 1 over `warmup_steps` training steps.
        Linearly decreases learning rate from 1. to 0. over remaining `t_total - warmup_steps` steps.
    """
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)
        
    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))



if __name__ == '__main__':
    create_input_files(csv_folder='./data2',
                       output_folder='./data2',
                       sentence_limit=20,
                       word_limit=60,
                       min_word_count=5)
