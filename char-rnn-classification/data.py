import glob  # python 内置文件操作模块
import os
import torch


def findFiles(path):
    return glob.glob(path)


# print(findFiles("data/names/*.txt"))

import unicodedata
import string

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)


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


# 从all_letters中查找字母索引，例如 "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)


def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor


def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

# print(n_letters)
# print(letterToIndex('J'))
# print(letterToTensor('J'))
# print(lineToTensor("Jones").size())
