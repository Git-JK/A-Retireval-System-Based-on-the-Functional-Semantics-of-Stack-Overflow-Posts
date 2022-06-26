import os
import time
from utils.config import W2V_CORPUS_STORE_PATH, PRETRAINED_W2V_MODEL_STORE_PATH
from gensim.models.word2vec import Word2Vec, LineSentence
from gensim.models.keyedvectors import KeyedVectors


if __name__ == "__main__":
    corpus_path = os.path.join(W2V_CORPUS_STORE_PATH, 'corpus.txt')
    pretrained_model_path = os.path.join(PRETRAINED_W2V_MODEL_STORE_PATH, 'wikidata_pretrained_model.txt')
    word2vec_model = Word2Vec(size=100, window=5, min_count=0, iter=10)
    pretrained_model = KeyedVectors.load_word2vec_format(pretrained_model_path, binary=False)
    more_sentence = LineSentence(corpus_path)
    word2vec_model.build_vocab(more_sentence)
    word2vec_model.build_vocab(list(pretrained_model.vocab.keys()), update=True)
    word2vec_model.intersect_word2vec_format(pretrained_model_path, binary=False, lockf=1.0)
    print("start time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    word2vec_model.train(more_sentence, total_examples=word2vec_model.corpus_count, epochs=word2vec_model.iter)
    print("end time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    model_save_path = os.path.join(PRETRAINED_W2V_MODEL_STORE_PATH, 'w2v_model.model')
    word2vec_model.save(model_save_path)