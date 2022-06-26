import os
import sys
import pymysql

if os.path.exists('D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'):
    base_dir = 'D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'
else:
    base_dir = '/media/dell/disk/jk/Retrieval'
sys.path.append(base_dir)

from baseline.utils.db_util import read_q_list_from_posts
import pandas as pd
from baseline.utils.preprocessing_util import preprocessing_for_que
from baseline.utils.time_utils import get_current_time

def preprocessing(qlist):
    print('preprocessing...', get_current_time())
    for i in range(len(qlist)):
        qlist[i] = preprocessing_for_que(qlist[i])
        if i % 1000 == 0:
            print('preprocessing %s questions...' % i, get_current_time())
    return qlist

if __name__ == "__main__":
    related_id_list_path = base_dir + "/baseline/data_process/related_qid_list.csv"
    related_id_list = pd.read_csv(related_id_list_path).values
    qlist = read_q_list_from_posts(related_id_list)
    qlist = preprocessing(qlist)
    store_list = []
    for q in qlist:
        store_list.append([q.id, q.title, q.body, ','.join(q.tag)])
    df = pd.DataFrame(store_list, columns=['Id', 'Title', 'Body', 'Tag'])
    repo_qs_path = base_dir + "/baseline/data_process/repo_qs.csv"
    df.to_csv(repo_qs_path, index=False)
    print(df)