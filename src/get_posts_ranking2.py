from turtle import pos
from BERT_3_calc_bert_score import generate_bertlets_input, calc_single_bertlets_score
from funcverb_2_calc_score import get_functionality_category_similarity_score, get_semantic_role_similarity_score, get_text_similarity_score, get_f_catogery_of_sentence, get_p_pattern_of_sentence
from utils.read.read_utils import read_EN_stopwords, load_w2v_model, read_all_from_testset
from utils.funcverb.funcverb_utils import find_root_of_dependency_tree, replace_part_of_clause, find_word_in_a_tree
from bert.model import load_bertlets_model
from utils.config import PRETRAINED_W2V_MODEL_STORE_PATH, SEARCHSET_STORE_PATH, BERT_MODEL_SOTRE_PATH, TEST_RESULTS_STORE_PATH
from funcverbnet.funcverbnet import FuncVerbNet
from transformers import BertTokenizer, logging
import spacy
import torch
from tqdm import tqdm
import time
import datetime
import pickle
import os

logging.set_verbosity_warning()
logging.set_verbosity_error()

def tensor2list(tensor):
    return tensor.numpy().tolist()

def get_recommendation_posts(query):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    torch.cuda.set_device(1)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)
    classifier = FuncVerbNet()
    q_Qd_bertlets_model = load_bertlets_model('bert-base-uncased', os.path.join(BERT_MODEL_SOTRE_PATH, 'q_Qd_BertletsModel_params.pkl'))
    q_a_bertlets_model = load_bertlets_model('bert-base-uncased', os.path.join(BERT_MODEL_SOTRE_PATH, 'q_a_BertletsModel_params.pkl'))
    q_Qd_bertlets_model.to(device)
    q_a_bertlets_model.to(device)
    q_Qd_bertlets_model.eval()
    q_a_bertlets_model.eval()
    nlp= spacy.load('en_core_web_lg')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    stopwords = read_EN_stopwords()
    w2v_model = load_w2v_model(os.path.join(PRETRAINED_W2V_MODEL_STORE_PATH, "w2v_model.model"))
    query_f_category = get_f_catogery_of_sentence(query, classifier)
    query_funcverb, query_f_clause, query_p_pattern = get_p_pattern_of_sentence(query, query_f_category, nlp, classifier)
    query_f_clause = " ".join([word.text for word in query_f_clause])
    query_f_clause_doc = nlp(query_f_clause)
    query_trip = tokenizer(query.lower(), add_special_tokens=False, return_tensors='pt')
    query_input_ids = tensor2list(query_trip['input_ids'])[0]
    query_token_type_ids = tensor2list(torch.ones(len(query_input_ids)))
    query_attention_mask = tensor2list(query_trip['attention_mask'])[0]
    if query_f_clause is None or query_p_pattern is None:
        query_semantic_part_list = []
    else:
        query_clause_root = find_root_of_dependency_tree(query_f_clause_doc)
        query_SP_list = replace_part_of_clause(query_f_clause_doc, query_clause_root, nlp)
        query_p_pattern_list = query_p_pattern.split(' ')
        query_pattern_pos_list = []
        query_semantic_part_list = []
        for i in range(len(query_p_pattern_list)):
            query_pattern_pos_list.append([query_p_pattern_list[i], query_SP_list[i][0]])
        for i in range(len(query_pattern_pos_list)):
            word = query_pattern_pos_list[i][0]
            if word[0] == '{':
                semantic_part = word
                if i > 0 and i < len(query_pattern_pos_list) - 1:
                    semantic_part_pos = [query_pattern_pos_list[i - 1][1] + 1, query_pattern_pos_list[i + 1][1]]
                    query_semantic_part_list.append((semantic_part, [query_f_clause_doc[i].text for i in range(semantic_part_pos[0], semantic_part_pos[1])]))
                elif i == 0:
                    semantic_part_pos = [0, query_pattern_pos_list[i + 1][1]]
                    query_semantic_part_list.append((semantic_part, [query_f_clause_doc[i].text for i in range(semantic_part_pos[0], semantic_part_pos[1])]))
                elif i == len(query_pattern_pos_list) - 1:
                    semantic_part_root = query_f_clause_doc[query_pattern_pos_list[i][1]]
                    semantic_part_clause = nlp(' '.join([w.text for w in semantic_part_root.subtree]))
                    root_relative_pos = find_word_in_a_tree(semantic_part_root, semantic_part_clause)
                    semantic_part_pos = [query_pattern_pos_list[i - 1][1] + 1, query_pattern_pos_list[i][1] - root_relative_pos + len([w for w in semantic_part_clause])]
                    query_semantic_part_list.append((semantic_part, [query_f_clause_doc[i].text for i in range(semantic_part_pos[0], semantic_part_pos[1])]))
    posts_ranking_list = []
    max_q_Qd_score = 0
    max_q_a_score = 0
    post_cnt = 0
    t0 = time.time()
    while post_cnt < 191:
        file_name = f"search_posts_{post_cnt}.pkl"
        file_path = os.path.join(SEARCHSET_STORE_PATH, file_name)
        search_posts = []
        with open(file_path, 'rb') as rbf:
            search_posts = pickle.load(rbf)
        # cnt = 0
        for post in tqdm(search_posts):
            # cnt += 1
            f_category_score = get_functionality_category_similarity_score(query_f_category, post['f_category'])
            semantic_role_score = get_semantic_role_similarity_score(query_semantic_part_list, post['semantic_part'], nlp, w2v_model)
            text_score = get_text_similarity_score(query, post['Title'], w2v_model, stopwords, nlp)
            funcverb_score = (f_category_score + semantic_role_score + text_score) / 3
            with torch.no_grad():
                q_Qd_bert_input = generate_bertlets_input(query_input_ids, query_token_type_ids, query_attention_mask, post['Qd_input_ids'], post['Qd_token_type_ids'], post['Qd_attention_mask'])
                q_Qd_bert_score = calc_single_bertlets_score(q_Qd_bert_input, q_Qd_bertlets_model, device).item()
                if q_Qd_bert_score > max_q_Qd_score:
                    max_q_Qd_score = q_Qd_bert_score
            post_q_a_score = 0
            # print(cnt)
            if 'AcceptedAnswer' in post.keys():
                answer = post['AcceptedAnswer']
                with torch.no_grad():
                    q_a_bert_input = generate_bertlets_input(query_input_ids, query_token_type_ids, query_attention_mask, answer['a_input_ids'], answer['a_token_type_ids'], answer['a_attention_mask'])
                    q_a_bert_score = calc_single_bertlets_score(q_a_bert_input, q_a_bertlets_model, device).item()
                    if q_a_bert_score > post_q_a_score:
                        post_q_a_score = q_a_bert_score
            if post_q_a_score > max_q_a_score:
                max_q_a_score = post_q_a_score
            q_a_bertlets_model.zero_grad()
            q_Qd_bertlets_model.zero_grad()
            q_a_bertlets_model.eval()
            q_Qd_bertlets_model.eval()
            posts_ranking_list.append({'Title': post['Title'], 'Body': post['Body'], 'PostLink': post['PostLink'], 'funcverb_score': funcverb_score, 'q_Qd_bert_score': q_Qd_bert_score, 'q_a_bert_score': post_q_a_score})
        post_cnt += 1
        print("\r", f'having searched {post_cnt} pickles', end="", flush=True)
        print(f" use time: {str(datetime.timedelta(seconds=int(round(time.time() - t0))))}")

    for item in posts_ranking_list:
        item['q_Qd_bert_score'] /= max_q_Qd_score
        item['q_a_bert_score'] /= max_q_a_score
        item['score'] = item['funcverb_score'] + item['q_Qd_bert_score'] + item['q_a_bert_score']
    posts_ranking_list = sorted(posts_ranking_list, key= lambda x:x['score'])
    cnt = 0
    recommendation = []
    for i in range(15):
        search_page_data = []
        for j in range(2):
            search = []
            for k in range(4):
                search.append(posts_ranking_list[cnt])
                cnt += 1
            search_page_data.append(search)
        recommendation.append(search_page_data)
    print("\r", f'finished recommendation', end="", flush=True)
    print(f" use time: {str(datetime.timedelta(seconds=int(round(time.time() - t0))))}")
    return recommendation

if __name__ == "__main__":
    test_list = read_all_from_testset()
    for i in range(1, 2):
        print(f"current query:{i}")
        query_item = test_list[i]
        query = query_item[1]
        query_id = query_item[0]
        recommendation = get_recommendation_posts(query)
        file_path = os.path.join(TEST_RESULTS_STORE_PATH, str(query_id) + '.pkl')
        with open(file_path, 'wb') as wbf:
            pickle.dump(recommendation, wbf)

