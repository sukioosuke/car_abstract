import pandas as pd
import numpy as np
import pkuseg
import re

from multiprocessing import Pool
from abstract import words_utils, utils


class Vocab(object):
    """get vocabulary in test and train data
    1. filter stopwords
    2. filter punctuation
    3. generate words dictionary
    """
    stop_words = []
    less_use = []
    vocabulary = {}
    MIN_COUNT = 5

    def __init__(self):
        file = open('../../data/stop_words.txt', mode='r')
        for i in file.readlines():
            self.stop_words.append(i.replace('\n', ''))

    def set_vocabulary(self, dataframe, partitions=8):
        df = pd.DataFrame
        data_split = np.array_split(dataframe, partitions)
        pool = Pool(partitions)
        data = pd.concat(pool.map(lambda x: self.set_vocabulary(x), data_split))
        pool.close()
        pool.join()
        return data

    def
        for col_name in dataframe.columns:
            df[col_name] = dataframe[col_name].apply(lambda x: self.cut_words(self, x))
        # for i in self.vocabulary:
        #     if self.vocabulary[i] < self.MIN_COUNT:
        #         self.less_use.append(i)
        # for i in self.less_use:
        #     self.vocabulary.pop(i)
        return df

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

    def filter_less_words(self, words_list):
        return list(filter(lambda x: x not in self.less_use, words_list))


if __name__ == '__main__':
    train_file = pd.read_csv('../../data/AutoMaster_TrainSet.csv', encoding='utf-8')
    test_file = pd.read_csv('../../data/AutoMaster_TestSet.csv', encoding='utf-8')

    train_content = train_file.dropna().drop("QID", axis=1)
    test_content = test_file.dropna().drop("QID", axis=1)

    v = Vocab
    df = utils.parallelize_class(train_content, v)
    # v.set_vocabulary(v, dataframe=test_content)
    print(df.head(10))