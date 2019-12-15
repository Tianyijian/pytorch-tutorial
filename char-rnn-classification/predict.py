import torch
import sys
from data import *

rnn = torch.load('char-rnn-classification.pt')


# 只需返回给定一行的输出
def evaluate(line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output


def predict(input_line, n_predictions=3):
    print('\n> %s' % input_line)
    with torch.no_grad():
        output = evaluate(lineToTensor(input_line))

        # 获得前N个类别
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print('(%.2f) %s' % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])
    return predictions


if __name__ == '__main__':
    # predict('Dovesky')
    # predict('Jackson')
    # predict('Satoshi')
    predict(sys.argv[1])
