import re
import gensim
import logging
import numpy as np

from gensim.models.word2vec import LineSentence
from gensim.models import word2vec

"""
Train words to vector model. Input data path and model saved path.
"""
def train_vec_model(path, model_path):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = word2vec.Word2Vec(LineSentence(path), workers=8, min_count=5, size=250)
    model.save(model_path)
    return model


"""
Load words to vector model. Input model path.
"""
def load_vec_model(path):
    word2vec.Word2Vec.load(path)


"""
Get words vector matrix. Same with model.wv.vectors but will save vector to path.
"""
def get_embedding(model, path):
    vocab_size = len(model.wv.vocab)
    embedding_dim = model.wv.vector_size
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for i in range(vocab_size):
        embedding_matrix[i, :] = model.wv[model.wv.index2word[i]]
        embedding_matrix = embedding_matrix.astype('float32')
    assert embedding_matrix.shape == (vocab_size, embedding_dim)
    np.savetxt(path, embedding_matrix, fmt='%0.8f')
    return embedding_matrix
