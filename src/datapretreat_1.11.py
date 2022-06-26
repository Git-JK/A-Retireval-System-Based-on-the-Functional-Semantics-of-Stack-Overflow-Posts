from funcverb_2_calc_score import get_f_catogery_of_sentence, get_p_pattern_of_sentence
from utils.read.read_utils import read_all_ids_from_testset, read_all_files_from_dir
from utils.funcverb.funcverb_utils import find_root_of_dependency_tree, replace_part_of_clause, find_word_in_a_tree
from funcverbnet.funcverbnet import FuncVerbNet
from transformers import BertTokenizer
from utils.mysql_access.posts import DBPosts
from utils.config import SEARCHSET_STORE_PATH
import fasttext
import pickle
import spacy
import torch
import datetime
import time
import os

fasttext.FastText.eprint = lambda x: None

def tensor2list(tensor):
    return tensor.numpy().tolist()


def collect_search_range_posts():
    posts_db = DBPosts()
    cnt = 0
    search_range_posts = []
    testset_ids = read_all_ids_from_testset()
    t0 = time.time()
    for item in posts_db.collect_search_range_posts():
        if str(item['Id']) not in testset_ids:
            search_range_posts.append(item)
        else:
            continue
        if len(search_range_posts) >= 50000:
            file_name = f'search_posts_{cnt}.pkl'
            file_store_path = os.path.join(SEARCHSET_STORE_PATH, file_name)
            with open(file_store_path, 'wb') as wbf:
                pickle.dump(search_range_posts, wbf)
            del search_range_posts
            search_range_posts = []
            cnt += 1
            print("\r", f'{cnt * 50000} searchset posts selected', end="", flush=True)
            print(f" use time: {str(datetime.timedelta(seconds=int(round(time.time() - t0))))}")
    if len(search_range_posts) > 0:
        file_name = f'search_posts_{cnt}.pkl'
        file_store_path = os.path.join(SEARCHSET_STORE_PATH, file_name)
        with open(file_store_path, 'wb') as wbf:
            pickle.dump(search_range_posts, wbf)
        del search_range_posts
        search_range_posts = []
        cnt += 1
    print("\r", 'completed!', end="", flush=True)


def preprocess_search_range_posts(bert_model_type):
    cnt = 160
    cur_posts = []
    tokenizer = BertTokenizer.from_pretrained(bert_model_type)
    classifier = FuncVerbNet()
    nlp= spacy.load('en_core_web_lg')
    t0 = time.time()
    while cnt < 180:
        file_name = f'search_posts_{cnt}.pkl'
        file_path = os.path.join(SEARCHSET_STORE_PATH, file_name)
        # post_cnt = 0
        with open(file_path, 'rb') as rbf:
            posts = pickle.load(rbf)
            for post in posts:
                # post_cnt += 1
                text = post['Title']
                post['f_category'] = get_f_catogery_of_sentence(text, classifier)
                if post['f_category'] != -1:
                    # print(text, post['f_category'])
                    funcverb, funcverb_clause, p_pattern = get_p_pattern_of_sentence(text, post['f_category'], nlp, classifier)
                    if funcverb_clause is None:
                        post['funcverb_clause'] = None
                    else:
                        post['funcverb_clause'] = " ".join([word.text for word in funcverb_clause])
                    post['p_pattern'] = p_pattern
                else:
                    post['funcverb'] = None
                    post['funcverb_clause'] = None
                    post['p_pattern'] = None
                # print(funcverb, funcverb_clause, p_pattern)
                
                Qd_token_trip = tokenizer(post['Body'].lower(), add_special_tokens=False, return_tensors='pt')
                Qd_input_ids = tensor2list(Qd_token_trip['input_ids'])[0]
                Qd_token_type_ids = tensor2list(torch.ones(len(Qd_input_ids)))
                Qd_attention_mask = tensor2list(Qd_token_trip['attention_mask'])[0]
                post['Qd_input_ids'] = Qd_input_ids
                post['Qd_token_type_ids'] = Qd_token_type_ids
                post['Qd_attention_mask'] = Qd_attention_mask
                new_post_answers = []
                for answer in post['Answers']:
                    a_token_trip = tokenizer(answer['Body'].lower(), add_special_tokens=False, return_tensors='pt')
                    a_input_ids = tensor2list(a_token_trip['input_ids'])[0]
                    a_token_type_ids = tensor2list(torch.ones(len(a_input_ids)))
                    a_attention_mask = tensor2list(a_token_trip['attention_mask'])[0]
                    answer['a_input_ids'] = a_input_ids
                    answer['a_token_type_ids'] = a_token_type_ids
                    answer['a_attention_mask'] = a_attention_mask
                    new_post_answers.append(answer)
                post['Answers'] = new_post_answers

                if post['funcverb_clause'] is None or post['p_pattern'] is None:
                    post['semantic_part'] = []
                    cur_posts.append(post)
                    continue
                q_fverb_clause = nlp(post['funcverb_clause'])
                q_p_pattern = post['p_pattern']
                q_clause_root = find_root_of_dependency_tree(q_fverb_clause)
                q_SP_list = replace_part_of_clause(q_fverb_clause, q_clause_root, nlp)
                q_clause_len = len([w for w in q_fverb_clause])
                q_p_pattern_list = q_p_pattern.split(' ')
                q_pattern_pos_list = []
                q_semantic_part_list = []
                for i in range(min(len(q_p_pattern_list), len(q_SP_list))):
                    q_pattern_pos_list.append([q_p_pattern_list[i], min(q_SP_list[i][0], q_clause_len - 1)])
                # print("\n")
                # print(post['Title'])
                # print(" ".join([w.text for w in q_fverb_clause]))
                # print(q_pattern_pos_list)
                for i in range(len(q_pattern_pos_list)):
                    word = q_pattern_pos_list[i][0]
                    if word[0] == '{':
                        semantic_part = word
                        if i > 0 and i < len(q_pattern_pos_list) - 1:
                            semantic_part_pos = [q_pattern_pos_list[i - 1][1] + 1, q_pattern_pos_list[i + 1][1]]
                            # print(semantic_part, semantic_part_pos)
                            q_semantic_part_list.append((semantic_part, [q_fverb_clause[i].text for i in range(semantic_part_pos[0], semantic_part_pos[1])]))
                        elif i == 0:
                            semantic_part_pos = [0, q_pattern_pos_list[i + 1][1]]
                            # print(semantic_part, semantic_part_pos)
                            q_semantic_part_list.append((semantic_part, [q_fverb_clause[i].text for i in range(semantic_part_pos[0], semantic_part_pos[1])]))
                        elif i == len(q_pattern_pos_list) - 1:
                            semantic_part_root = q_fverb_clause[q_pattern_pos_list[i][1]]
                            semantic_part_clause = nlp(' '.join([w.text for w in semantic_part_root.subtree]))
                            root_relative_pos = find_word_in_a_tree(semantic_part_root, semantic_part_clause)
                            semantic_part_pos = [q_pattern_pos_list[i - 1][1] + 1, min(q_pattern_pos_list[i][1] - root_relative_pos + len([w for w in semantic_part_clause]), q_clause_len)]
                            # print(semantic_part, semantic_part_pos,  q_pattern_pos_list[i][1], root_relative_pos, len([w for w in semantic_part_clause]))
                            q_semantic_part_list.append((semantic_part, [q_fverb_clause[i].text for i in range(semantic_part_pos[0], semantic_part_pos[1])]))
                # print(q_semantic_part_list)
                # print('\n')
                post['semantic_part'] = q_semantic_part_list
                cur_posts.append(post)
                # if post_cnt % 1000 == 0:
                #     print("\r", f'{post_cnt} searchset posts processed', end="", flush=True)
                #     print(f" use time: {str(datetime.timedelta(seconds=int(round(time.time() - t0))))}")
        with open(file_path, 'wb') as wbf:
            pickle.dump(cur_posts, wbf)
            del cur_posts
            cur_posts = []
            cnt += 1
            print("\r", f'{cnt * 50000} searchset posts processed', end="", flush=True)
            print(f" use time: {str(datetime.timedelta(seconds=int(round(time.time() - t0))))}")


if __name__ == "__main__":
    # collect_search_range_posts()
    bert_model_type = 'bert-base-uncased'
    preprocess_search_range_posts(bert_model_type)