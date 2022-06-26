from bert.model import load_bertlets_model
from funcverb_2_calc_score import get_semantic_part_text_similarity
import os
import pickle
import spacy
from tqdm import tqdm
import torch
from utils.config import BERT_MODEL_SOTRE_PATH, BERT_Q_A_TEST_TOKEN_STORE_PATH, BERT_Q_QD_TEST_TOKEN_STORE_PATH, PRETRAINED_W2V_MODEL_STORE_PATH, BERT_Q_QD_TEST_DATA_STORE_PATH, BERT_Q_A_TEST_DATA_STORE_PATH
from utils.read.read_utils import load_w2v_model
from rank_bm25 import BM25Okapi
from torch.utils.data import DataLoader
from bert.load_data import BertletsDataset
from transformers import logging


logging.set_verbosity_warning()
logging.set_verbosity_error()

def evaluation(testDataloader, model, device):
    model.eval()
    test_total = 0
    test_correct = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for step, batch in tqdm(enumerate(testDataloader)):
        batch = batch.to(device)
        test_total += 1
        pos_score = model.forward_once(batch[:, 0])
        neg_score = model.forward_once(batch[:, 1])
        if pos_score > neg_score:
            TP += 1
            TN += 1
            test_correct += 1
        else:
            FP += 1
            FN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1score = 2 * precision * recall / (precision + recall)
    return test_correct / test_total, precision, recall, f1score 


def evaluate_bert():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    q_a_bertlets_model = load_bertlets_model('bert-base-uncased', os.path.join(BERT_MODEL_SOTRE_PATH, 'q_a_BertletsModel_params.pkl'))
    q_a_bertlets_model.to(device)
    q_a_BertletsTestDataset = BertletsDataset('bert-base-uncased', BERT_Q_A_TEST_TOKEN_STORE_PATH)
    test_dataloader = DataLoader(q_a_BertletsTestDataset, batch_size=1, shuffle=True, num_workers=4)
    test_accuracy = evaluation(test_dataloader, q_a_bertlets_model, device)
    print(f'test accuracy: {test_accuracy}')

def evalueate_w2v():
    nlp= spacy.load('en_core_web_lg')
    w2v_model = load_w2v_model(os.path.join(PRETRAINED_W2V_MODEL_STORE_PATH, "w2v_model.model"))
    triplets_list = []
    test_total = 0
    test_correct = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for cnt in range(2):
        file_name = f"q_a_pair_{cnt}.pkl"
        file_path = os.path.join(BERT_Q_A_TEST_DATA_STORE_PATH, file_name)
        with open(file_path, 'rb') as rbf:
            triplets = pickle.load(rbf)
            triplets_list += triplets
    for triplet in tqdm(triplets_list):
        test_total += 1
        query = triplet[0]
        pos = triplet[1]
        neg = triplet[2]
        pos_sim = get_semantic_part_text_similarity(query, pos, w2v_model, nlp)
        neg_sim =get_semantic_part_text_similarity(query, neg, w2v_model, nlp)
        if pos_sim > neg_sim:
            TP += 1
            TN += 1
            test_correct += 1
        else:
            FP += 1
            FN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1score = 2 * precision * recall / (precision + recall)
    print(test_correct / test_total, precision, recall, f1score)
    return test_correct / test_total, precision, recall, f1score


def evaluate_bm25():
    corpus = []
    triplets_list = []
    test_total = 0
    test_correct = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    cnt = 0
    for cnt in range(2):
        file_name = f"q_a_pair_{cnt}.pkl"
        file_path = os.path.join(BERT_Q_A_TEST_DATA_STORE_PATH, file_name)
        with open(file_path, 'rb') as rbf:
            triplets = pickle.load(rbf)
            triplets_list += triplets
    for triplet in triplets_list:
        if triplet[1] not in corpus:
            corpus.append(triplet[1])
        if triplet[2] not in corpus:
            corpus.append(triplet[2])
    tok_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tok_corpus)
    for triplet in tqdm(triplets_list):
        query = triplet[0].split(" ")
        test_total += 1
        scores = bm25.get_scores(query)
        # print(scores)
        pos_score = scores[corpus.index(triplet[1])]
        neg_score = scores[corpus.index(triplet[2])]
        if pos_score > neg_score:
            TP += 1
            TN += 1
            test_correct += 1
        else:
            FP += 1
            FN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1score = 2 * precision * recall / (precision + recall)
    print(test_correct / test_total, precision, recall, f1score)



if __name__ == "__main__":
    # evaluate_bert()
    # evalueate_w2v()
    evaluate_bm25()