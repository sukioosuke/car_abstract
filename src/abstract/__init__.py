import pandas as pd

from abstract.vocab import Vocab
from utils.paths import *
from utils import words_utils


def remove_pic_and_sound(line):
    content = filter(lambda x: not x.__contains__('[语音]') and not x.__contains__('[图片]'), line.split('|'))
    return '|'.join(content)


if __name__ == '__main__':
    train_file = pd.read_csv(train_data_path, encoding='utf-8')
    test_file = pd.read_csv(test_data_path, encoding='utf-8')

    train_content = train_file.dropna().drop("QID", axis=1)
    test_content = test_file.dropna().drop("QID", axis=1)

    train_content['Dialogue'] = train_content['Dialogue'].apply(remove_pic_and_sound)
    test_content['Dialogue'] = test_content['Dialogue'].apply(remove_pic_and_sound)

    v = Vocab()
    # train_df = v.multi_set_vocabulary(train_content)
    train_df = v.set_vocabulary(train_content)
    train_df.to_csv(train_clean_path, index=None, header=False)
    # test_df = v.multi_set_vocabulary(test_content)
    test_df = v.set_vocabulary(test_content)
    test_df.to_csv(test_clean_path, index=None, header=False)
    train_df['merged'] = train_df[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)
    test_df['merged'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)
    merged_df = pd.concat([train_df[['merged']], test_df[['merged']]], axis=0)
    merged_df.to_csv(merged_data_path, index=None, header=False)
    words_utils.train_vec_model(merged_data_path, vector_path)