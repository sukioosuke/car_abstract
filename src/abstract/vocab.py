# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import jieba
import pkuseg
import re
import datetime

from multiprocessing import Pool, cpu_count, Manager
from utils.paths import *


class Vocab:
    """get vocabulary in test and train data
    1. filter stopwords
    2. filter words which less occurrences in text
    3. generate words dictionary

    @:param multi: multiple processes running
    @:param package: choose which package to cut words(eg: jieba, pkuseg)
    """
    __manager = None
    __lock = None
    __MIN_COUNT = 5
    __multi = False
    __package = "jieba"
    __partition = 1
    stop_words = []
    less_use = []
    vocabulary = {}

    def __init__(self, multi=False, package="jieba"):
        file = open(stop_words_path, mode='r', encoding='utf-8')
        for i in file.readlines():
            self.stop_words.append(i.replace('\n', ''))
        if multi:
            self.__manager = Manager()
            self.__lock = self.__manager.Lock()
            self.__partition = cpu_count()
            self.vocabulary = self.__manager.dict()
            self.multi = multi
        self.__package = package

    """
    Return words in sentence. Will replace punctuation with space.
    """
    def cut_words(self, sentence):
        cut = []
        seg = None
        clean = re.sub(
            r'[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5]|[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
            ' ', sentence)
        if self.__package == "jieba":
            jieba.load_userdict(user_dict_path)
            seg = jieba.cut(clean)
        elif self.__package == "pkuseg":
            s = pkuseg.pkuseg(model_name='default', user_dict=user_dict_path)
            seg = s.cut(clean)
        for j in seg:
            if self.__multi:
                with self.lock:
                    if j in self.vocabulary.keys():
                        self.vocabulary[j] += 1
                        cut.append(j)
                    elif j not in self.stop_words:
                        self.vocabulary[j] = 1
                        cut.append(j)
            else:
                if j in self.vocabulary.keys():
                    self.vocabulary[j] += 1
                    cut.append(j)
                elif j not in self.stop_words:
                    self.vocabulary[j] = 1
                    cut.append(j)
        return ' '.join(cut)

    """
    Set word dictionary with input dataframe. If init class with param multi which is True, then it will run in multiple processes. 
    """
    def set_vocabulary(self, dataframe):
        if self.__multi:
            pool = Pool(self.__partition)
            data_split = np.array_split(dataframe, self.__partition)
            data = pd.concat(pool.map(self.dataframe_cut, data_split))
            pool.close()
            pool.join()
            return data
        else:
            return self.dataframe_cut(dataframe)

    def filter_lessuse(self, sentence):
        return ' '.join(filter(lambda x: x != '' and x not in self.less_use, sentence.split(' ')))

    def update_lessuse(self):
        for word in self.vocabulary.keys():
            if self.vocabulary.get(word) < self.__MIN_COUNT:
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


if __name__ == '__main__':
    train_file = pd.read_csv(train_data_path, encoding='utf-8')
    test_file = pd.read_csv(test_data_path, encoding='utf-8')

    train_content = train_file.dropna().drop("QID", axis=1)
    test_content = test_file.dropna().drop("QID", axis=1)

    v = Vocab()
    # train_df = v.multi_set_vocabulary(train_content)
    train_df = v.set_vocabulary(train_content.head(8))
    train_df.to_csv(train_clean_path, index=None, header=False)
