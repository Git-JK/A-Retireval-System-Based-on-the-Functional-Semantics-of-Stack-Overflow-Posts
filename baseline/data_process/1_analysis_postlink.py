import pickle
import os
import sys

if os.path.exists('D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'):
    base_dir = 'D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'
else:
    base_dir = '/media/dell/disk/jk/Retrieval'
sys.path.append(base_dir)

from baseline.utils.time_utils import get_current_time
from baseline.utils.csv_utils import write_list_to_csv
import pandas as pd
import pymysql
from baseline.utils.preprocessing_util import preprocessing_for_que


def read_all_files_from_dir(dir_name):
    file_list = os.listdir(dir_name)
    return file_list

def get_all_qid_set():
    # 创建all_qid_list，包含所有posts questions的id（也就是corpus中所有posts的id）
    posts_path = base_dir + '/data/trainset_posts/'
    file_list = read_all_files_from_dir(posts_path)
    qid_list = []
    count = 0
    for file_name in file_list:
        file_path = os.path.join(posts_path, file_name)
        with open(file_path, 'rb') as rbf:
            posts = pickle.load(rbf)
            for post in posts:
                count += 1
                qid_list.append(int(post['Id']))
                if len(qid_list) % 10000 == 0:
                    print('Load %s Id' % len(qid_list))
    all_qid_list_path = base_dir + "/baseline/data_process/all_qid_list.csv"
    header = ['Id']
    write_list_to_csv(qid_list, all_qid_list_path, header)
    return

def load_all_qid_list(csv_path):
    # corpus中所有的posts的id
    id_set = set()
    df = pd.read_csv(csv_path, header=None, low_memory=False)
    for idx, row in df.iterrows():
        if row[0] == "Id":
            continue
        id_set.add(int(row[0]))
    print('#all questions = %s' % len(id_set), get_current_time())
    return id_set

def extract_relevant_ids_from_post_link(all_id_set):
    # 找到有related关系的postId和related postId都在all_id_set中(corpus posts的id集合)的，将其id保存在id_dict中，sort一下存入related_qid_list.csv
    sql = f"select * from PostLinks"
    db = pymysql.connect(host="162.105.16.191", user="root", password="root", database="sotorrent")
    cursor = db.cursor()
    id_dict = {}
    count = 0
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        for row in results:
            postId = row[2]
            related_postId = row[3]
            if postId in all_id_set and related_postId in all_id_set and (int(postId) >= 4 and int(postId) <= 5816076):
                if postId not in id_dict:
                    id_dict[postId] = True
                if related_postId not in id_dict:
                    id_dict[related_postId] = True
            count += 1
            if count % 10000 == 0:
                print("Processing %s..." % count, get_current_time())
    except Exception as e:
        print(e)
    cursor.close()
    db.close()
    print("relevant qid = %s" % len(id_dict), get_current_time())
    return sorted(list(id_dict.keys()))

if __name__ == "__main__":
    # get_all_qid_set()
    all_qid_set_path = base_dir + "/baseline/data_process/all_qid_list.csv"
    all_id_set = load_all_qid_list(all_qid_set_path)
    related_id_list = extract_relevant_ids_from_post_link(all_id_set)
    # post id list
    related_id_list_path = base_dir + "/baseline/data_process/related_qid_list.csv"
    header = ['Id']
    write_list_to_csv(related_id_list, related_id_list_path, header)


    