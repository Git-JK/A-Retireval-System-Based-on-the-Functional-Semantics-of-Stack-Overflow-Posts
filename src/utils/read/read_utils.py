from ..config import SO_POSTS_STORE_PATH, TAG_STORE_PATH, STOPWORDS_PATH, TRAINSET_STORE_PATH, TESTSET_STORE_PATH, SEARCHSET_STORE_PATH
from gensim.models.word2vec import Word2Vec
import pickle
import pandas as pd
import os

def read_all_files_from_dir(dir_name):
    file_list = os.listdir(dir_name)
    return file_list

def read_questions_from_posts():
    file_list = read_all_files_from_dir(SO_POSTS_STORE_PATH)
    post_list = []
    count = 0
    for file_name in file_list:
        file_path = os.path.join(SO_POSTS_STORE_PATH, file_name)
        with open(file_path, 'rb') as rbf:
            posts = pickle.load(rbf)
            for post in posts:
                count += 1
                post_list.append(post)
                if len(post_list) % 10000 == 0:
                    print('Load %s questions' % len(post_list))
    return post_list

def read_all_posts_from_trainset():
    file_list = read_all_files_from_dir(TRAINSET_STORE_PATH)
    post_list = []
    # count = 0
    for file_name in file_list:
        file_path = os.path.join(TRAINSET_STORE_PATH, file_name)
        with open(file_path, 'rb') as rbf:
            posts = pickle.load(rbf)
            post_list += posts
    print("Loading train posts completed!")
    return post_list

def read_all_queries_from_testset():
    testset_path = os.path.join(TESTSET_STORE_PATH, 'testset_query.csv')
    query_list = []
    df = pd.read_csv(testset_path)
    for row in df.itertuples():
        query_list.append(row[2])
    return query_list

def read_all_ids_from_testset():
    testset_path = os.path.join(TESTSET_STORE_PATH, 'testset_query.csv')
    id_list = []
    df = pd.read_csv(testset_path)
    for row in df.itertuples():
        id_list.append(row[1])
    return id_list
    

def read_all_from_testset():
    testset_path = os.path.join(TESTSET_STORE_PATH, 'testset_query.csv')
    test_list = []
    df = pd.read_csv(testset_path)
    for row in df.itertuples():
        test_list.append([row[1], row[2]])
    return test_list

def read_all_posts_from_searchset():
    file_list = read_all_files_from_dir(SEARCHSET_STORE_PATH)
    post_list = []
    cnt = 0
    for file_name in file_list:
        if cnt < 5:
            file_path = os.path.join(SEARCHSET_STORE_PATH, file_name)
            with open(file_path, 'rb') as rbf:
                posts = pickle.load(rbf)
                post_list += posts
            cnt += 1
        else:
            break
    print("Loading searchset posts completed!")
    return post_list

def read_all_tag_list():
    tag_path = os.path.join(TAG_STORE_PATH, 'top_tag_list.csv')
    tag_list = []
    df_tag = pd.read_csv(tag_path, low_memory=False)
    for tag_tmp in df_tag.itertuples():
        tag_list.append(tag_tmp[1])
    return tag_list

def read_topk_tag_list(topk):
    tag_path = os.path.join(TAG_STORE_PATH, 'top_tag_list.csv')
    topk_tag_list = []
    df_tag = pd.read_csv(tag_path, low_memory=False)
    count = 0
    for tag_tmp in df_tag.itertuples():
        topk_tag_list.append(tag_tmp[1])
        count += 1
        if count == topk:
            break
    return topk_tag_list


def read_EN_stopwords():
    stopwords_path = os.path.join(STOPWORDS_PATH, 'StopWords_EN.txt')
    sw_set = set()
    with open(stopwords_path, 'r') as f:
        for line in f:
            sw_set.add(line.strip())
        return sw_set

def load_w2v_model(w2v_model_path):
    w2v_model = Word2Vec.load(w2v_model_path)
    return w2v_model

def load_test_posts():
    file_path = os.path.join(SEARCHSET_STORE_PATH, 'search_posts_0.pkl')
    test_posts = []
    with open(file_path, 'rb') as rbf:
        posts = pickle.load(rbf)
        cnt = 0
        for i in range(15):
            search_page_data = []
            for j in range(2):
                search = []
                for k in range(4):
                    search.append(posts[cnt])
                    cnt += 1
                search_page_data.append(search)
            test_posts.append(search_page_data)
    return test_posts


def query2id(query):
    test_list = read_all_from_testset()
    q_id = ""
    for item in test_list:
        # print(item, query)
        if item[1] == query:
            q_id = item[0]
            break
    return str(q_id)
