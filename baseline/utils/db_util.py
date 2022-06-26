import pymysql
import pickle
import pandas as pd
import os
import sys

if os.path.exists('D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'):
    base_dir = 'D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'
else:
    base_dir = '/media/dell/disk/jk/Retrieval'
sys.path.append(base_dir)

from baseline.data_structure.SO_que import SO_Que
from baseline.data_structure.SO_ans import SO_Ans
from baseline.utils.preprocessing_util import preprocessing_for_que, preprocessing_for_ans
from baseline.utils.time_utils import get_current_time

def read_all_files_from_dir(dir_name):
    file_list = os.listdir(dir_name)
    return file_list


def read_q_list_from_posts(id_list):
    # 读取id在id_list中的posts的id,title,body,tags
    posts_path = base_dir + '/data/trainset_posts/'
    file_list = read_all_files_from_dir(posts_path)
    qlist = []
    count = 0
    for file_name in file_list:
        file_path = os.path.join(posts_path, file_name)
        with open(file_path, 'rb') as rbf:
            posts = pickle.load(rbf)
            for post in posts:
                if int(post['Id']) in id_list:
                    count += 1
                    qlist.append(SO_Que(post['Id'], post['Title'], post['Body'], post['Tags']))
                    if len(qlist) % 10000 == 0:
                        print('Load %s questions' % len(qlist))
    return qlist

def read_all_questions_from_repo():
    repo_qs_path = base_dir + '/baseline/data_process/repo_qs.csv'
    df_repo_qs = pd.read_csv(repo_qs_path, low_memory=False)
    repo_qs = []
    for row in df_repo_qs.itertuples():
        repo_qs.append(SO_Que(row[1], row[2], row[3], row[4]))
    return repo_qs
    

def read_specific_question_from_repo(id):
    repo_qs_path = base_dir + '/baseline/data_process/repo_qs.csv'
    df_repo_qs = pd.read_csv(repo_qs_path, low_memory=False)
    for row in df_repo_qs.itertuples():
        if str(id) == str(row[1]):
            return SO_Que(row[1], row[2], row[3], row[4])


def read_correspond_answers_from_all_posts(top_dq_id_and_sim):
    corr_answers = {}
    posts_path = base_dir + '/data/trainset_posts/'
    file_list = read_all_files_from_dir(posts_path)
    id_list = []
    for (qid, sim) in top_dq_id_and_sim:
        id_list.append(str(qid))
    for file_name in file_list:
        file_path = os.path.join(posts_path, file_name)
        with open(file_path, 'rb') as rbf:
            posts = pickle.load(rbf)
            for post in posts:
                if str(post['Id']) in id_list:
                    corr_answer = []
                    for answer in post['Answers']:
                        SO_AnswerUnit_tmp = SO_Ans(answer['Id'], answer['Body'], answer['Score'], str(post['Id']))
                        corr_answer.append(SO_AnswerUnit_tmp)
                    corr_answers[str(post['Id'])] = corr_answer
    return corr_answers
    
