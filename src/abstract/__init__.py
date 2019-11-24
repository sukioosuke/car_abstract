import pandas as pd

from abstract.vocab import Vocab
from abstract import utils


def remove_pic_and_sound(line):
    content = filter(lambda x: not x.__contains__('[语音]') and not x.__contains__('[图片]'), line.split('|'))
    return '|'.join(content)


if __name__ == '__main__':
    train_file = pd.read_csv('../../data/AutoMaster_TrainSet.csv', encoding='utf-8')
    test_file = pd.read_csv('../../data/AutoMaster_TestSet.csv', encoding='utf-8')

    train_content = train_file.dropna().drop("QID", axis=1)
    test_content = test_file.dropna().drop("QID", axis=1)

    train_content['Dialogue'] = train_content['Dialogue'].apply(remove_pic_and_sound)
    test_content['Dialogue'] = test_content['Dialogue'].apply(remove_pic_and_sound)

    v = Vocab
    utils.parallelize(train_content, v)
    utils.parallelize(test_content, v)
    train_df = utils.parallelize(train_content, v.cut_words)
    test_df = utils.parallelize(test_content, v.cut_words)
    file = open('../../output/vocabulary.csv', 'w')
    file.write('vocabulary\n')
    file.writelines(map(lambda x: x + '\n', list(v.vocabulary.keys())))
    file.close()
