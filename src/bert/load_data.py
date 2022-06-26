import os
import time
import datetime
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import pickle


def read_all_files_from_dir(dir_name):
    file_list = os.listdir(dir_name)
    return file_list

def tensor2list(tensor):
    return tensor.numpy().tolist()

class BertletsDataset(Dataset):
    
    def __init__(self, bert_model_type, data_dir_path):
        super().__init__()
        data_file_list = read_all_files_from_dir(data_dir_path)
        triplets = []
        t0 = time.time()
        cnt = 0
        for file_name in data_file_list:
            file_path = os.path.join(data_dir_path, file_name)
            with open(file_path, 'rb') as rbf:
                triplet_list = pickle.load(rbf)
                for triplet in triplet_list:
                    triplets.append(triplet)
                    cnt += 1
                    if cnt % 100000 == 0:
                        print(f'load {cnt} data')
                        print(f'use time: {str(datetime.timedelta(seconds=int(round(time.time() - t0))))}')
        print('Loading Completed!')
        # file_path = os.path.join(data_dir_path, 'q_Qd_pair_0.pkl')
        # with open(file_path, 'rb') as rbf:
        #     triplet_list = pickle.load(rbf)
        #     cnt = 0
        #     for triplet in triplet_list:
        #         triplets.append(triplet)
        #         # triplets.append(torch.tensor([pos_pair, neg_pair]))
        #         cnt += 1
        #         if cnt % 1000 == 0:
        #             print(f'load {cnt} data')
        #             break
        self.triplets = triplets
    
    def __getitem__(self, index):
        return self.triplets[index]
    
    def __len__(self):
        return len(self.triplets)
