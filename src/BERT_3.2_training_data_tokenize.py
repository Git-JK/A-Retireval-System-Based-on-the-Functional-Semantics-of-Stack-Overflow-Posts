import os
import torch
import time
import datetime
from transformers import BertTokenizer
import transformers
import pickle
from utils.read.read_utils import read_all_files_from_dir
from utils.config import BERT_Q_A_TOKEN_STORE_PATH, BERT_Q_QD_TOKEN_STORE_PATH, BERT_Q_A_DATA_STORE_PATH, BERT_Q_QD_DATA_STORE_PATH


def tensor2list(tensor):
    return tensor.numpy().tolist()


def training_data_tokenize(bert_model_type, data_dir_path, token_store_path):
    data_file_list = read_all_files_from_dir(data_dir_path)
    tokenizer = BertTokenizer.from_pretrained(bert_model_type)
    triplets = []
    cnt = 0
    for file_name in data_file_list:
        file_path = os.path.join(data_dir_path, file_name)
        # t0 = time.time()
        with open(file_path, 'rb') as rbf:
            triplet_list = pickle.load(rbf)
            for triplet in triplet_list:
                pos_pair = tokenizer(triplet[0].lower(), triplet[1].lower(), padding='max_length', return_tensors='pt', max_length=512, truncation=True)
                neg_pair = tokenizer(triplet[0].lower(), triplet[2].lower(), padding='max_length', return_tensors='pt', max_length=512, truncation=True)
                triplets.append(torch.tensor([[tensor2list(pos_pair['input_ids'])[0], tensor2list(pos_pair['token_type_ids'])[0], tensor2list(pos_pair['attention_mask'])[0]], [tensor2list(neg_pair['input_ids'])[0], tensor2list(neg_pair['token_type_ids'])[0], tensor2list(neg_pair['attention_mask'])[0]]]))
                cnt += 1
                if cnt % 10000 == 0:
                    print(f'load {cnt} data')
                    # print(f'use time: {str(datetime.timedelta(seconds=int(round(time.time() - t0))))}')
        store_path = os.path.join(token_store_path, file_name)
        with open(store_path, 'wb') as wbf:
            pickle.dump(triplets, wbf)
        triplets = []
                

if __name__ == "__main__":
    transformers.logging.set_verbosity_error()
    bert_model_type = 'bert-base-uncased'
    training_data_tokenize(bert_model_type, BERT_Q_QD_DATA_STORE_PATH, BERT_Q_QD_TOKEN_STORE_PATH)
    training_data_tokenize(bert_model_type, BERT_Q_A_DATA_STORE_PATH, BERT_Q_A_TOKEN_STORE_PATH)
