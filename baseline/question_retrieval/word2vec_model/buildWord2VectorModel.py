import os
import sys

if os.path.exists('D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'):
    base_dir = 'D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'
else:
    base_dir = '/media/dell/disk/jk/Retrieval'
sys.path.append(base_dir)

from gensim.models.word2vec import Word2Vec, LineSentence
from baseline.utils.time_utils import get_current_time

corpus_path = base_dir + '/data/baseline_corpus/corpus.txt'

print('start time: ', get_current_time())
sentences = LineSentence(corpus_path)

model = Word2Vec(sentences, vector_size=200, window=5, min_count=0, workers=4, epochs=100)

model.save(base_dir + '/baseline/question_retrieval/word2vec_model/model')
print('end time: ', get_current_time())