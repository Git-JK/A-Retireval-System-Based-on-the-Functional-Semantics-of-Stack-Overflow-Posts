import os
import sys

if os.path.exists('D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'):
    base_dir = 'D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'
else:
    base_dir = '/media/dell/disk/jk/Retrieval'
sys.path.append(base_dir)

import gensim
import pandas as pd
from nltk import word_tokenize
from baseline.utils.StopWords import read_EN_stopwords, remove_stopwords

w2v_model_path = base_dir + '/baseline/question_retrieval/word2vec_model/model'
vocab_fpath = base_dir + '/baseline/question_retrieval/IDF_vocabulary/idf_vocab.csv'

def load_w2v_model():
    word2vector_model = gensim.models.Word2Vec.load(w2v_model_path)
    return word2vector_model

def load_idf_vocab():
    vocab_dict = {}
    df_vocab = pd.read_csv(vocab_fpath, low_memory=False)
    for row in df_vocab.itertuples():
        vocab_dict[str(row[1])] = float(row[2])
    return vocab_dict

def preprocessing_for_query(q):
    # basic processing for query
    qw = word_tokenize(q.lower())
    stopwords = read_EN_stopwords()
    qw = remove_stopwords(qw, stopwords)
    return qw

if __name__ == "__main__":
    print(load_idf_vocab())