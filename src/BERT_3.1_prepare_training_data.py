import os
import pickle
from utils.read.read_utils import read_all_posts_from_trainset
from utils.text_process.puretext_extract import puretext_extract, preprocess_title
from utils.config import BERT_Q_A_DATA_STORE_PATH, BERT_Q_QD_DATA_STORE_PATH
import random
import time
import datetime

random.seed(1)



def generate_BERT_data():
    post_list = read_all_posts_from_trainset()
    q_Qd_pair_list = []
    q_a_pair_list = []
    Qd_cnt = 0
    a_cnt = 0
    t0 = time.time()
    for post in post_list:
        # 生成(q, Qd, 非Qd) pair
        neg_sample_num = random.randint(3, 4)
        random_selected_post = random.sample(post_list, neg_sample_num)
        if post in random_selected_post:
            random_selected_post.remove(post)
        else:
            random_selected_post = random.sample(random_selected_post, neg_sample_num - 1)
        for neg_post in random_selected_post:
            q_Qd_pair_list.append((preprocess_title(post['Title']), puretext_extract(post['Body']), puretext_extract(neg_post['Body'])))
        # 生成(q, a, 非a) pair
        neg_sample_num = random.randint(3, 4)
        random_selected_post = random.sample(post_list, neg_sample_num)
        if post in random_selected_post:
            random_selected_post.remove(post)
        else:
            random_selected_post = random.sample(random_selected_post, neg_sample_num - 1)
        post_answers = post['Answers']
        post_answers = sorted(post_answers, key= lambda x:x['Score'])
        post_answer = ''
        for answer in post_answers:
            if answer['Accepted'] == 1:
                post_answer = answer['Body']
                break
        if post_answer == '':
            post_answer = post_answers[0]['Body']
        for neg_post in random_selected_post:
            neg_post_answers = neg_post['Answers']
            neg_post_answers = sorted(neg_post_answers, key=lambda x: x['Score'], reverse=True)
            neg_answer = ''
            for answer in neg_post_answers:
                if answer['Accepted'] == 1:
                    neg_answer = answer['Body']
                    break
            if neg_answer == '':
                neg_answer = neg_post_answers[0]['Body']
            q_a_pair_list.append((preprocess_title(post['Title']), puretext_extract(post_answer), puretext_extract(neg_answer)))
        if len(q_Qd_pair_list) >= 50000:
            file_name = f'q_Qd_pair_{Qd_cnt}.pkl'
            file_path = os.path.join(BERT_Q_QD_DATA_STORE_PATH, file_name)
            with open(file_path, 'wb') as wbf:
                pickle.dump(q_Qd_pair_list[0:50000], wbf)
            q_Qd_pair_list = q_Qd_pair_list[50000:]
            Qd_cnt += 1
            print("\r", f"{Qd_cnt * 50000} q_Qd_pair stored!", end="", flush=True)
            print(f' use time: {str(datetime.timedelta(seconds=int(round(time.time() - t0))))}')
        if len(q_a_pair_list) >= 50000:
            file_name = f'q_a_pair_{a_cnt}.pkl'
            file_path = os.path.join(BERT_Q_A_DATA_STORE_PATH, file_name)
            with open(file_path, 'wb') as wbf:
                pickle.dump(q_a_pair_list[0:50000], wbf)
            q_a_pair_list = q_a_pair_list[50000:]
            a_cnt += 1
            print("\r", f"{a_cnt * 50000} q_a_pair stored!", end="", flush=True)
            print(f' use time: {str(datetime.timedelta(seconds=int(round(time.time() - t0))))}')
    if len(q_Qd_pair_list) > 0:
        file_name = f'q_Qd_pair_{Qd_cnt}.pkl'
        file_path = os.path.join(BERT_Q_QD_DATA_STORE_PATH, file_name)
        with open(file_path, 'wb') as wbf:
            pickle.dump(q_Qd_pair_list, wbf)
    if len(q_a_pair_list) > 0:
        file_name = f'q_a_pair_{a_cnt}.pkl'
        file_path = os.path.join(BERT_Q_A_DATA_STORE_PATH, file_name)
        with open(file_path, 'wb') as wbf:
            pickle.dump(q_a_pair_list, wbf)
    print(f'Completed! {Qd_cnt * 50000} q_Qd_nQd pairs generated, {a_cnt * 50000} q_a_na pairs generated')
            

if __name__ == "__main__":
    generate_BERT_data()
