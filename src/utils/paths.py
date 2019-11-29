import os
import pathlib

root = pathlib.Path(os.path.abspath(os.curdir)).parent.parent

train_data_path = os.path.join(root, 'data', 'AutoMaster_TrainSet.csv')
test_data_path = os.path.join(root, 'data', 'AutoMaster_TestSet.csv')
stop_words_path = os.path.join(root, 'data', 'stop_words.txt')
user_dict_path = os.path.join(root, 'data', 'user_dict.txt')

#中间数据
train_clean_path = os.path.join(root, 'output', 'TrainSet.csv')
test_clean_path = os.path.join(root, 'output', 'TestSet.csv')
merged_data_path = os.path.join(root, 'output', 'MergedData.csv')
vector_path = os.path.join(root, 'output', 'VectorModel.model')