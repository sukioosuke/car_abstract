import pandas as pd
import pkuseg
import re

from abstract import words_utils


class Vocab(object):
    """get vocabulary in test and train data
    1. filter stopwords
    2. filter punctuation
    3. generate words dictionary
    """
    stop_words = []
    vocabulary = dict()
    percent = 0.0001

    def __init__(self):
        file = open('../../data/stop_words.txt', mode='r')
        for i in file.readlines():
            self.stop_words.append(i.replace('\n', ''))

    def set_vocabulary(self, dataframe):

        for col_name in dataframe.columns:
            dataframe[col_name].apply(self.cut_words)

        less_use = []
        for i in self.vocabulary:
            if self.vocabulary[i] < max(list(self.vocabulary.values())) * self.percent:
                less_use.append(i)
        for i in less_use:
            self.vocabulary.pop(i)

    def cut_words(self, sentence):
        seg = pkuseg.pkuseg(model_name='default', user_dict='../../data/user_dict.txt')
        clean = words_utils.clean_sentence(sentence)
        for j in seg.cut(clean):
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
