import pandas as pd
import numpy as np
import pkuseg
import re

from multiprocessing import Pool, Process, Manager
from utils.paths import *


class Vocab(object):
    """get vocabulary in test and train data
    1. filter stopwords
    2. filter punctuation
    3. generate words dictionary
    """
    manager = Manager()
    lock = manager.Lock()
    stop_words = []
    less_use = []
    vocabulary = manager.dict()
    MIN_COUNT = 5

    def __init__(self):
        file = open('../../data/stop_words.txt', mode='r')
        for i in file.readlines():
            self.stop_words.append(i.replace('\n', ''))

    def cut_words(self, sentence):
        cut = []
        seg = pkuseg.pkuseg(model_name='default', user_dict=user_dict_path)
        clean = re.sub(
            r'[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]|[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
            ' ', sentence)
        for j in seg.cut(clean):
            with self.lock:
                if j in self.vocabulary.keys():
                    self.vocabulary[j] += 1
                    cut.append(j)
                elif j not in self.stop_words:
                    self.vocabulary[j] = 1
                    cut.append(j)
        return ' '.join(cut)

    def filter_lessuse(self, sentence):
        ' '.join(sentence.split(' ').filter(lambda x: x in self.less_use))

    def update_lessuse(self):
        for word in self.vocabulary.keys():
            if self.vocabulary.get(word) < self.MIN_COUNT:
                self.less_use.append(word)
        for i in self.less_use: self.vocabulary.pop(i)

    def dataframe_cut(self, df):
        for col_name in df.columns:
            df[col_name] = df[col_name].apply(self.cut_words)
        return df

    def dataframe_filter(self, df):
        for col_name in df.columns:
            df[col_name] = df[col_name].apply(self.filter_lessuse)
        return df

    def set_vocabulary(self, dataframe, partitions=8):
        df = pd.DataFrame
        pool = Pool(partitions)
        data_split = np.array_split(dataframe, partitions)
        data = pd.concat(pool.map(self.dataframe_cut, data_split))
        self.update_lessuse()
        data_split = np.array_split(data, partitions)
        data = pd.concat(pool.map(self.dataframe_filter, data_split))
        pool.close()
        pool.join()
        return data


if __name__ == '__main__':
    v = Vocab()
    train_file = pd.read_csv(train_data_path, encoding='utf-8')
    v.set_vocabulary(train_file.head(24))
    print(v.vocabulary)
