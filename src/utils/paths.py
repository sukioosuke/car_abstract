import os
import pathlib

root = pathlib.Path(os.path.abspath(os.curdir)).parent.parent
if root.name != 'car_abstract':
    root = pathlib.Path(os.path.abspath(os.curdir)).parent

# Input data set
train_data_path = os.path.join(root, 'data', 'AutoMaster_TrainSet.csv')
test_data_path = os.path.join(root, 'data', 'AutoMaster_TestSet.csv')
stop_words_path = os.path.join(root, 'data', 'stop_words.txt')
user_dict_path = os.path.join(root, 'data', 'user_dict.txt')

# Intermediate data set
train_clean_path = os.path.join(root, 'output', 'TrainSet.csv')
test_clean_path = os.path.join(root, 'output', 'TestSet.csv')
merged_data_path = os.path.join(root, 'output', 'MergedData.csv')
model_path = os.path.join(root, 'output', 'VectorModel.model')
vector_path = os.path.join(root, 'output', 'Vector.csv')