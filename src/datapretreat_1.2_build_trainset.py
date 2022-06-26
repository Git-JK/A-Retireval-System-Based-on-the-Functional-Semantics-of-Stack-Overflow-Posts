from utils.config import SEARCHSET_STORE_PATH, TESTSET_STORE_PATH, SO_POSTS_STORE_PATH, TRAINSET_STORE_PATH
from utils.read.read_utils import read_topk_tag_list, read_all_files_from_dir, read_all_ids_from_testset
from utils.mysql_access.posts import DBPosts
import pickle
import pandas as pd
import random
import time
import datetime
import os

random.seed(1)


def select_testset_posts():
    tag_list = read_topk_tag_list(20)
    tag_dict = {}
    for tag in tag_list:
        tag_dict[tag] = []
    file_list = read_all_files_from_dir(SO_POSTS_STORE_PATH)
    for file_name in file_list:
        file_path = os.path.join(SO_POSTS_STORE_PATH, file_name)
        with open(file_path, 'rb') as rbf:
            posts = pickle.load(rbf)
            for post in posts:
                if not isinstance(post['Tags'], str):
                    continue
                tag_listr_tmp = post['Tags'].replace('<', ' ').replace('>', ' ').replace('  ', ' ').strip()
                cur_tag = tag_listr_tmp.split(' ')
                for tag in tag_list:
                    if tag in cur_tag:
                        tag_dict[tag].append([str(post['Id']),post['Title']])
    testset_posts = []
    for tag in tag_dict.keys():
        testset_posts += random.sample(tag_dict[tag], 5)
    return testset_posts


def select_trainset_posts(testset_ids):
    train_posts = []
    posts_db = DBPosts()
    cnt = 0
    t0 = time.time()
    for item in posts_db.collect_trainset_posts():
        if str(item['Id']) not in testset_ids:
            train_posts.append(item)
        else:
            continue
        if len(train_posts) >= 50000:
            file_name = f'trainset_posts_{cnt}.pkl'
            file_store_path = os.path.join(TRAINSET_STORE_PATH, file_name)
            with open(file_store_path, 'wb') as wbf:
                pickle.dump(train_posts, wbf)
            del train_posts
            train_posts = []
            cnt += 1
            print("\r", f'{cnt * 50000} trainset posts selected', end="", flush=True)
            print(f" use time: {str(datetime.timedelta(seconds=int(round(time.time() - t0))))}")
    if len(train_posts) > 0:
        file_name = f'trainset_posts_{cnt}.pkl'
        file_store_path = os.path.join(TRAINSET_STORE_PATH, file_name)
        with open(file_store_path, 'wb') as wbf:
            pickle.dump(train_posts, wbf)
        del train_posts
        train_posts = []
        cnt += 1
    print("\r", 'completed!', end="", flush=True)
    

if __name__ == "__main__":
    # testset_path = os.path.join(TESTSET_STORE_PATH, 'testset_query.csv')
    # testset_posts = select_testset_posts()
    # header = ['Id', 'Query']
    # df_testset = pd.DataFrame(testset_posts, columns=header)
    # df_testset.to_csv(testset_path, index=False)
    # searchset_posts = select_searchset_posts(testset_posts)
    # cur_post_list = []
    # cnt = 0
    # for post in searchset_posts:
    #     cur_post_list.append(post)
    #     if len(cur_post_list) >= 50000:
    #         file_name = f'searchset_posts_{cnt}.pkl'
    #         file_store_path = os.path.join(SEARCHSET_STORE_PATH, file_name)
    #         with open(file_store_path, 'wb') as wbf:
    #             pickle.dump(cur_post_list, wbf)
    #         del cur_post_list
    #         cur_post_list = []
    #         cnt += 1
    #         print("\r", f"{cnt * 50000} search posts stored", end="", flush=True)
    # if len(cur_post_list) >= 0:
    #     file_name = f'searchset_posts_{cnt}.pkl'
    #     file_store_path = os.path.join(SEARCHSET_STORE_PATH, file_name)
    #     with open(file_store_path, 'wb') as wbf:
    #         pickle.dump(cur_post_list, wbf)
    #     del cur_post_list
    #     cur_post_list = []
    #     cnt += 1
    # print("\r", "Completed!", end="", flush=True)
    testset_ids = read_all_ids_from_testset()
    select_trainset_posts(testset_ids)




                    
