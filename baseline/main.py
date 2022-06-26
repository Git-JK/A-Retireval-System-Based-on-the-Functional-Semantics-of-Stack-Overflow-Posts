import os
import sys

if os.path.exists('D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'):
    base_dir = 'D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'
else:
    base_dir = '/media/dell/disk/jk/Retrieval'
sys.path.append(base_dir)

import time
import pandas as pd
from baseline.utils.StopWords import remove_stopwords, read_EN_stopwords
from baseline.question_retrieval.get_dq import get_dq, init_doc_matrix, init_doc_idf_vector
from baseline.sentence_selection.get_ss import get_ss
from baseline.summarization.get_summary import get_summary
from baseline.pathConfig import res_dir
from baseline.utils.csv_utils import write_list_to_csv
from baseline.utils.data_util import load_idf_vocab, load_w2v_model, preprocessing_for_query
from utils.db_util import read_all_questions_from_repo

def get_querylist(list_path):
    query_list = []
    df_query = pd.read_csv(list_path, low_memory=False)
    for line in df_query.itertuples():
        query_list.append(line[2])
    return query_list

def preprocessing_all_questions(questions, idf, w2v, stopword):
    processed_questions = []
    stopwords = stopword
    count = 0
    for question in questions:
        title_words = remove_stopwords(question.title, stopwords)
        if len(title_words) <= 2:
            continue
        if title_words[-1] == '?':
            title_words = title_words[:-1]
        question.title_words = title_words
        question.matrix = init_doc_matrix(question.title_words, w2v)
        question.idf_vector = init_doc_idf_vector(question.title_words, idf)
        processed_questions.append(question)
        count += 1
        if count % 10000 == 0:
            print(f"preprocessing {count} repo questions")
    return processed_questions

if __name__ == "__main__":
    topnum = 10
    # load word2vec model
    print('load textual word2vec_model(): ', time.strftime('%Y-%m-%d %H:%M:%S'))
    w2v_model = load_w2v_model()
    # load repo
    print('load repo:', time.strftime('%Y-%m-%d %H:%M:%S'))
    repo = read_all_questions_from_repo()
    # load idf
    print('load textual voc:', time.strftime('%Y-%m-%d %H:%M:%S'))
    idf_vocab = load_idf_vocab()

    list_path = f'{base_dir}/data/testset_posts/testset_query.csv'
    query_list = get_querylist(list_path)
    query_list = ["how to return the response from an asynchronous call", "how to convert an int to string?", "how can I prevent sql injection in php?", "how do I compare strings in java?"]
    dq_res = []
    stopword = read_EN_stopwords()

    # process questions
    questions = preprocessing_all_questions(repo, idf_vocab, w2v_model, stopword)

    for query in query_list:
        print(f"query: {query}...{time.strftime('%Y-%m-%d %H:%M:%S')}")
        query_word = preprocessing_for_query(query)
        query_matrix = init_doc_matrix(query_word, w2v_model)
        query_idf = init_doc_idf_vector(query_word, idf_vocab)
        top_dq = get_dq(query_word, topnum, questions, query_idf, query_matrix)
        cur_res_dict = []
        for i in range(len(top_dq)):
            q = top_dq[i][0]
            sim = top_dq[i][1]
            cur_res_dict.append((q.id, round(sim, 2)))
        dq_res.append([query, cur_res_dict])
    dq_res_fpath = os.path.join(res_dir, 'rq_res.csv')
    header = ["query", "rq_id_list"]
    write_list_to_csv(dq_res, dq_res_fpath, header)

    # df_dq_res = pd.read_csv(dq_res_fpath, low_memory=False)
    # for line in df_dq_res.itertuples():
    #     dq_res.append([line[1], eval(line[2])])
        
    print('sentence selection...', time.strftime('%Y-%m-%d %H:%M:%S'))
    ss_res = []
    ss_res_save = []
    count = 0
    for query, top_dq_id_and_sim in dq_res:
        query_word = preprocessing_for_query(query)
        top_ss = get_ss(query_word, topnum, top_dq_id_and_sim, stopword)
        ss_res.append((query, top_ss))
        ss_res_save.append([query, top_ss])
        count += 1
        print(f'finished {count} query sentence selection...')
    ss_res_fpath = os.path.join(res_dir, 'ss_res.csv')
    header = ['query', 'top_ss']
    write_list_to_csv(ss_res_save, ss_res_fpath, header)

    # ss_res_fpath = os.path.join(res_dir, 'ss_res.csv')
    # df_ss_res = pd.read_csv(ss_res_fpath, low_memory=False)
    # ss_res = []
    # for line in df_ss_res.itertuples():
    #     ss_res.append((line[1], eval(line[2])))

    print('get summary...', time.strftime('%Y-%m-%d %H:%M:%S'))
    res = []
    count = 0
    for query, ss in ss_res:
        origin_query = query
        query = ' '.join(preprocessing_for_query(query))
        sum = get_summary(query, ss, 5)
        res.append([origin_query, sum])
        count += 1
        print(f'finished {count} summaries...')
    res_fpath = os.path.join(res_dir, 'summary_res.csv')
    header = ['query', 'summary']
    write_list_to_csv(res, res_fpath, header)
