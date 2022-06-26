import os
import pickle
from utils.config import W2V_CORPUS_STORE_PATH, TRAINSET_STORE_PATH
from utils.read.read_utils import read_all_files_from_dir
from utils.text_process.puretext_extract import preprocess_title



def read_questions_from_posts():
    file_list = read_all_files_from_dir(TRAINSET_STORE_PATH)
    qlist = []
    count = 0
    for file_name in file_list:
        file_path = os.path.join(TRAINSET_STORE_PATH, file_name)
        with open(file_path, 'rb') as rbf:
            posts = pickle.load(rbf)
            for post in posts:
                count += 1
                title = preprocess_title(post['Title'])
                qlist.append(title)
                if count % 10000 == 0:
                    print(f'Load {count} questions')
    return qlist


if __name__ == "__main__":
    corpus_path = os.path.join(W2V_CORPUS_STORE_PATH, 'corpus.txt')
    qlist = read_questions_from_posts()
    with open(corpus_path, 'w') as wf:
        for question in qlist:
            try:
                wf.write(str(question) + "\n")
            except Exception as e:
                print(e)
    print('Completed!')

