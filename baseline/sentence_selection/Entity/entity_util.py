import pickle
import os
import sys

if os.path.exists('D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'):
    base_dir = 'D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'
else:
    base_dir = '/media/dell/disk/jk/Retrieval'
sys.path.append(base_dir)

from baseline.utils.file_util import write_file
from baseline.question_retrieval.preprocessing.build_corpus import read_all_files_from_dir

dic_path = base_dir + '/baseline/sentence_selection/Entity/entity_dic.txt'

def load_entity_set():
    dic = set()
    for line in open(dic_path):
        line = line.strip()
        dic.add(line)
    return dic

def extract_tag_info_from_all_posts():
    # 抽出所有posts的tag
    posts_path = base_dir + '/data/trainset_posts/'
    file_list = read_all_files_from_dir(posts_path)
    dic = set()
    count = 0
    for file_name in file_list:
        file_path = os.path.join(posts_path, file_name)
        with open(file_path, 'rb') as rbf:
            posts = pickle.load(rbf)
            for post in posts:
                if not isinstance(post['Tags'], str):
                    continue
                tag_listr_tmp = post['Tags'].replace('<', ' ').replace('>', ' ').replace('  ', ' ').strip()
                for tag_tmp in tag_listr_tmp.split(' '):
                    if tag_tmp not in dic:
                        dic.add(tag_tmp)
                count += 1
                if count % 10000 == 0:
                    print(f'processing {count} instances')
    return dic

if __name__ == "__main__":
    # 抽出所有posts的tag，存入entity_dic.txt
    dic = extract_tag_info_from_all_posts()
    dic_str = ''
    for tag_tmp in dic:
        dic_str += (tag_tmp + '\n')
    write_file(dic_path, dic_str)
    print('Completed!')
