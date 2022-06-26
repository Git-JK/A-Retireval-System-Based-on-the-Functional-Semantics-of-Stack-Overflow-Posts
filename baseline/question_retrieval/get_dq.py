import os
import sys

from numpy import matrix

if os.path.exists('D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'):
    base_dir = 'D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'
else:
    base_dir = '/media/dell/disk/jk/Retrieval'
sys.path.append(base_dir)

import operator
import time
from baseline.utils.StopWords import remove_stopwords, read_EN_stopwords
from baseline.utils.db_util import read_all_questions_from_repo, read_specific_question_from_repo
import numpy as np

def init_doc_matrix(doc, w2v):
    # 生成doc的vector matrix，并进行单位化
    matrix = np.zeros((len(doc), 200))
    for i, word in enumerate(doc):
        if word in w2v.wv.key_to_index:
            matrix[i] = np.array(w2v.wv[word])
    
    try:
        norm = np.linalg.norm(matrix, axis=1).reshape(len(doc), 1)
        matrix = np.divide(matrix, norm, out=np.zeros_like(matrix), where=norm!=0)
    except RuntimeWarning:
        print(doc)
    
    return matrix

def init_doc_idf_vector(doc, idf):
    #生成doc的对应idf vector
    idf_vector = np.zeros((1, len(doc)))
    for i, word in enumerate(doc):
        if word in idf:
            idf_vector[0][i] = idf[word]
    
    return idf_vector

def sim_doc_pair(matrix1, matrix2, idf1, idf2):
    # 计算两个doc的idf相似度
    sim12 = (idf1 * (matrix1.dot(matrix2.T).max(axis=1))).sum() / idf1.sum()
    sim21 = (idf2*(matrix2.dot(matrix1.T).max(axis=1))).sum() / idf2.sum()
    return (sim12 + sim21) / 2.0


def calc_wordvec_similarity(vec1, vec2):
    # 计算两个word vector的cosine similarity
    vec1 = vec1.reshape((1, len(vec1)))
    vec2 = vec2.reshape((1, len(vec2)))
    x1_norm = np.sqrt(np.sum(vec1 ** 2, axis=1, keepdims=True))
    x2_norm = np.sqrt(np.sum(vec2 ** 2, axis=1, keepdims=True))
    prod = np.sum(vec1 * vec2, axis=1, keepdims=True)
    cosine_sim = prod / (x1_norm * x2_norm)
    return cosine_sim[0][0]

def calc_similarity(word_list_1, word_list_2, idf_voc, word2vector_model):
    if len(word_list_1) == 0 or len(word_list_2) == 0:
        return 0.0
    sim_up = 0
    sim_down = 0
    for w1 in word_list_1:
        w1_unicode = w1.encode('utf-8', 'ignore').decode('utf-8')
        if w1_unicode in word2vector_model.wv.key_to_index:
            w1_vec = word2vector_model.wv[w1_unicode]
            maxsim = 0.0
            for w2 in word_list_2:
                w2_unicode = w2.encode('utf-8', 'ignore').decode('utf-8')
                if w2_unicode in word2vector_model.wv.key_to_index:
                    w2_vec = word2vector_model.wv[w2_unicode]
                    sim_tmp = calc_wordvec_similarity(w1_vec, w2_vec)
                    maxsim = max(maxsim, sim_tmp)
            # if exist in idf
            if w1 in idf_voc:
                idf = idf_voc[w1]
                sim_up += maxsim * idf
                sim_down += idf
    if sim_down == 0:
        print("sim_down = 0!\n word sent 1 %s\nword sent 2 %s" % (word_list_1, word_list_2))
        return 0.0
    return sim_up / sim_down

def get_dq(query_w, topnum, questions, query_idf, query_matrix):
    rank = []
    count = 0
    for question in questions:
        sim = sim_doc_pair(query_matrix, question.matrix, query_idf, question.idf_vector)
        rank.append([question.id, sim])
        count += 1
        if count % 10000 == 0:
            print(f'Processed {count} questions... {time.strftime("%Y-%m-%d %H:%M:%S")}')
    
    rank.sort(key=operator.itemgetter(1), reverse=True)
    top_dq = []
    for i in range(0, len(rank), 1):
        id = rank[i][0]
        sim = rank[i][1]
        rank.append(id)
        if i < topnum:
            qs = read_specific_question_from_repo(id)
            top_dq.append((qs, sim))
    return top_dq
