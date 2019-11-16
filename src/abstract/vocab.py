import pandas as pd
import pkuseg
import re


class Vocab(object):
    """get vocabulary in test and train data
    1. filter voice question and answer
    2. filter picture question and answer
    """
    stop_words = []
    vocabulary = dict()
    percent = 0.00001

    def __init__(self):
        file = open('../../data/stop_words.txt', mode='r')
        for i in file.readlines():
            self.stop_words.append(i.replace('\n', ''))

    def set_vocabulary(self, dataframe):
        #替换中英文标点符号
        # regex = "[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5|]" \
        #         "|[\s+\.\!\/_,$%^*(+\"\']+|[+—\?~@#%&*]"
        seg = pkuseg.pkuseg(model_name='default')
        for index in dataframe.columns:
            for i in dataframe[index]:
                # sentence = re.sub(regex, ' ', i)
                for j in seg.cut(i):
                    if j in self.vocabulary.keys():
                        self.vocabulary[j] += 1
                    elif j not in self.stop_words:
                        self.vocabulary[j] = 1

    def filter_stop_words(self, words_list):
        return list(filter(lambda x: x not in self.stop_words, words_list))


if __name__ == '__main__':
    train_file = pd.read_csv('../../data/AutoMaster_TrainSet.csv', encoding='utf-8')
    test_file = pd.read_csv('../../data/AutoMaster_TestSet.csv', encoding='utf-8')

    train_content = train_file.dropna().drop("QID", axis=1)
    test_content = test_file.dropna().drop("QID", axis=1)

    v = Vocab
    v.set_vocabulary(v, dataframe=train_content)
    v.set_vocabulary(v, dataframe=test_content)
    print(v.vocabulary)
    print(v.vocabulary.keys)
