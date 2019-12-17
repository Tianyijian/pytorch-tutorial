import glob  # python 内置文件操作模块
import os
import torch
import random


def findFiles(path):
    return glob.glob(path)


# print(findFiles("data/names/*.txt"))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'-"
n_letters = len(all_letters) + 1  # Plus EOS marker


def unicodeToAscii(s):
    """将Unicode字符串转换为纯ASCII"""
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


# print(unicodeToAscii('Ślusàrski'))

# 构建category_lines字典，每种语言的名字列表
category_lines = {}
all_categories = []


# 读取文件并分成几行
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]


for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
                       'from https://download.pytorch.org/tutorial/data.zip and extract it to '
                       'the current directory.')


# 从all_letters中查找字母索引，例如 "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


# 用于输入的从头到尾字母（不包括EOS）的one-hot矩阵
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor


# print(n_letters)
# print(letterToIndex('J'))
# print(letterToTensor('J'))
# print(lineToTensor("Jones").size())

# 列表中的随机项
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]


# 从该类别中获取随机类别和随机行
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line


# 类别的One-hot张量
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor


# 用于目标的第二个结束字母（EOS）的LongTensor
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)
    return torch.LongTensor(letter_indexes)


# 从随机(类别，行)对中创建类别，输入和目标张量
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = lineToTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor

# for i in range(1):
#     category_tensor, input_line_tensor, target_line_tensor = randomTrainingExample()
#     print('category_tensor =', category_tensor, '/ input_line_tensor =', input_line_tensor)
