import os
import sys
import pickle

if os.path.exists('D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'):
    base_dir = 'D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'
else:
    base_dir = '/media/dell/disk/jk/Retrieval'
sys.path.append(base_dir)

from baseline.data_structure.SO_que import SO_Que
from baseline.utils.preprocessing_util import preprocessing_for_que


def read_all_files_from_dir(dir_name):
    file_list = os.listdir(dir_name)
    return file_list
    

def read_questions_from_posts():
    posts_path = base_dir + '/data/trainset_posts/'
    file_list = read_all_files_from_dir(posts_path)
    qlist = []
    count = 0
    for file_name in file_list:
        file_path = os.path.join(posts_path, file_name)
        with open(file_path, 'rb') as rbf:
            posts = pickle.load(rbf)
            for post in posts:
                count += 1
                q_tmp = SO_Que(post['Id'], post['Title'], post['Body'], post['Tags'])
                q_tmp = preprocessing_for_que(q_tmp)
                qlist.append(q_tmp)
                if len(qlist) % 10000 == 0:
                    print('Load %s questions' % len(qlist))
    return qlist
    
    
if __name__ == "__main__":
    corpus_path = base_dir + '/data/baseline_corpus/corpus.txt'
    qlist = read_questions_from_posts()
    with open(corpus_path, 'w') as f:
        for q in qlist:
            try:
                f.write(str(q.title) + " " + str(q.body) + "\n")
            except Exception as e:
                print(e)
                print(q.id, q.title, q.body)
    print('Compeleted!')