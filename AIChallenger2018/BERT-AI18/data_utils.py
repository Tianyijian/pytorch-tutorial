import csv
import re


data_dir = "/users5/yjtian/tyj/demo/HAN2/data2/"

def prepare_pt_file():
    raw_file = ["sentiment_analysis_trainingset.csv", "sentiment_analysis_validationset.csv"]
    pt_file = ["./pt_data/pt_train.txt", "./pt_data/pt_dev.txt"]
    for i in range(len(raw_file)):
        with open(data_dir + raw_file[i], "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            docs = []
            for j, line in enumerate(reader):
                if j == 0:
                    continue
                if "\n" in line[1]:
                    docs.append(line[1].replace("\n\n", "\n") + "\n\n")
                else:
                    sentences = splitsentence(line[1])
                    docs.append("\n".join(sentences) + "\n\n")
        with open(pt_file[i], "w", encoding="utf-8") as f:
            f.writelines(docs)


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


def statistic():
    pt_file = ["./pt_data/pt_train.txt", "./pt_data/pt_dev.txt"]
    for i in range(len(pt_file)):
        line_num = 0
        sent_length = 0
        line_l_128 = 0
        with open(pt_file[i], "r", encoding="utf-8") as f:
            for line in f.readlines():
                if line.strip() != "":
                    line_num += 1
                    sent_length += len(line)
                    if len(line) < 128:
                        line_l_128 += 1
        print("line_num:{}, sent_length:{}, avg:{:.4f}, line_l_128:{}({:.4f})".format(line_num, sent_length, float(sent_length) / line_num, line_l_128, float(line_l_128) / line_num))


if __name__ == "__main__":
    # prepare_pt_file()
    statistic()