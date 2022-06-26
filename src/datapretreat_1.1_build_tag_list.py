from utils.config import SO_POSTS_STORE_PATH, TAG_STORE_PATH
from utils.read.read_utils import read_all_files_from_dir
import pickle
import pandas as pd
import os



def extract_top_tag_from_posts():
    # 抽出所有posts的tag并包含的post num排序
    file_list = read_all_files_from_dir(SO_POSTS_STORE_PATH)
    tag_dict = {}
    count = 0
    for file_name in file_list:
        file_path = os.path.join(SO_POSTS_STORE_PATH, file_name)
        with open(file_path, 'rb') as rbf:
            posts = pickle.load(rbf)
            for post in posts:
                if not isinstance(post['Tags'], str):
                    continue
                tag_listr_tmp = post['Tags'].replace('<', ' ').replace('>', ' ').replace('  ', ' ').strip()
                for tag_tmp in tag_listr_tmp.split(' '):
                    tag_tmp = str(tag_tmp)
                    if tag_tmp not in tag_dict.keys():
                        tag_dict[tag_tmp] = 0
                    tag_dict[tag_tmp] += 1
                count += 1
                if count % 10000 == 0:
                    print(f'processing {count} posts')
    sorted_tag_list = sorted(tag_dict.items(), key= lambda x : x[1], reverse=True)
    return sorted_tag_list
    

if __name__ == "__main__":
    tag_path = os.path.join(TAG_STORE_PATH, 'top_tag_list.csv')
    top_tag_list = extract_top_tag_from_posts()
    header = ['Tag', 'Post_num']
    df_tag = pd.DataFrame(top_tag_list, columns=header)
    df_tag.to_csv(tag_path, index=False)
    print('Completed!')
    