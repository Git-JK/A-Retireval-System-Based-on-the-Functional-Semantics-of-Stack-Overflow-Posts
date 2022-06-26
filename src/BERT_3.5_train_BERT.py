from tensorboardX import SummaryWriter
import os
import time
import random
import datetime
import torch
import transformers
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import AdamW
from bert.model import BertletsModel, TripletLoss
from bert.load_data import BertletsDataset
from utils.config import BERT_Q_QD_TOKEN_STORE_PATH, BERT_Q_A_TOKEN_STORE_PATH, Q_QD_RUN_DATA_PATH, Q_A_RUN_DATA_PATH, BERT_MODEL_SOTRE_PATH, BERT_Q_A_TEST_TOKEN_STORE_PATH, BERT_Q_QD_TEST_TOKEN_STORE_PATH

seed_val = 1
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)
transformers.logging.set_verbosity_error()



def cal_loss(pos_score, neg_score, loss_margin):
    pos = torch.exp(pos_score) / (torch.exp(pos_score) + torch.exp(neg_score))
    neg = torch.exp(neg_score) / (torch.exp(pos_score) + torch.exp(neg_score))
    tmp = loss_margin - (pos - neg)
    return torch.mean(tmp)

def evaluation(testDataloader, model, device):
    model.eval()
    test_total = 0
    test_correct = 0
    for step, batch in enumerate(testDataloader):
        batch = batch.to(device)
        test_total += 1
        pos_score = model.forward_once(batch[:, 0])
        neg_score = model.forward_once(batch[:, 1])
        if pos_score > neg_score:
            test_correct += 1
    return test_correct / test_total
        

def train_q_Qd_Bertlets():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    q_Qd_writer = SummaryWriter(Q_QD_RUN_DATA_PATH)
    bert_model_type = 'bert-base-uncased'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # batch_size参数
    batch_size = 8
    # 准备dataset和dataloader
    q_Qd_BertletsDataset = BertletsDataset(bert_model_type, BERT_Q_QD_TOKEN_STORE_PATH)
    dataloader = DataLoader(q_Qd_BertletsDataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 准备test_dateset和test_dataloader
    q_Qd_BertletsTestDataset = BertletsDataset(bert_model_type, BERT_Q_QD_TEST_TOKEN_STORE_PATH)
    test_dataloader = DataLoader(q_Qd_BertletsTestDataset, batch_size=1, shuffle=True, num_workers=4)

    # 准备model和model params
    model = BertletsModel(bert_model_type)
    model.to(device)
    
    lr = 2e-5
    epochs = 3
    loss_margin = 0.1

    # 准备optimizer和loss_fn
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = TripletLoss(loss_margin)

    # Training

    for epoch in range(epochs):

        model.train()
        print(f'Epoch {epoch + 1} / {epochs}')
        t0 = time.time()

        for step, batch in tqdm(enumerate(dataloader)):
            
            batch = batch.to(device)
            model.zero_grad()
            optimizer.zero_grad()
            pos_score, neg_score = model(batch)
            
            batch_loss = loss_fn(pos_score, neg_score)
            batch_loss.backward()
            
            clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            
            q_Qd_writer.add_scalar('batch_avg_loss', cal_loss(pos_score.clone().detach(), neg_score.clone().detach(), loss_margin), global_step=step)

            if step % 10 == 0:
                elapsed = str(datetime.timedelta(seconds=int(round(time.time() - t0))))
                print(f'  Batch {step}/{len(dataloader)}, Elapsed {elapsed}, loss = {cal_loss(pos_score.clone().detach(), neg_score.clone().detach(), loss_margin)}')
        
        # training time for one epoch

        training_time = str(datetime.timedelta(seconds=int(round(time.time() - t0))))
        print(f'Training epoch took: {training_time}')

        # evaluation
        test_accuracy = evaluation(test_dataloader, model, device)
        print(f'test accuracy: {test_accuracy}')

        # record test accuracy
        with open('/media/dell/disk/jk/Retrieval/run/q_Qd_test/test_accuracy.txt', 'a+') as f:
            f.writelines('test accuracy: ' + str(test_accuracy))

    print('Training Completed!')
    
    model.eval()
    # save model params
    model_save_path  = os.path.join(BERT_MODEL_SOTRE_PATH, 'q_Qd_BertletsModel_params.pkl')
    torch.save(model.state_dict(), model_save_path)


def train_q_a_Bertlets():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    q_a_writer = SummaryWriter(Q_A_RUN_DATA_PATH)
    bert_model_type = 'bert-base-uncased'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # batch_size参数
    batch_size = 8
    # 准备dataset和dataloader
    q_a_BertletsDataset = BertletsDataset(bert_model_type, BERT_Q_A_TOKEN_STORE_PATH)
    dataloader = DataLoader(q_a_BertletsDataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # 准备test_dateset和test_dataloader
    q_a_BertletsTestDataset = BertletsDataset(bert_model_type, BERT_Q_A_TEST_TOKEN_STORE_PATH)
    test_dataloader = DataLoader(q_a_BertletsTestDataset, batch_size=1, shuffle=True, num_workers=4)

    # 准备model和model params
    model = BertletsModel(bert_model_type)
    model.to(device)
    
    lr = 2e-5
    epochs = 3
    loss_margin = 0.1

    # 准备optimizer和loss_fn
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = TripletLoss(loss_margin)

    # Training

    for epoch in range(epochs):

        model.train()
        print(f'Epoch {epoch + 1} / {epochs}')
        t0 = time.time()

        for step, batch in tqdm(enumerate(dataloader)):
            
            batch = batch.to(device)
            model.zero_grad()
            optimizer.zero_grad()
            pos_score, neg_score = model(batch)
            
            batch_loss = loss_fn(pos_score, neg_score)
            batch_loss.backward()
            
            clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            
            q_a_writer.add_scalar('batch_avg_loss', cal_loss(pos_score.clone().detach(), neg_score.clone().detach(), loss_margin), global_step=step)

            if step % 10 == 0:
                elapsed = str(datetime.timedelta(seconds=int(round(time.time() - t0))))
                print(f'  Batch {step}/{len(dataloader)}, Elapsed {elapsed}, loss = {cal_loss(pos_score.clone().detach(), neg_score.clone().detach(), loss_margin)}')
        
        # training time for one epoch

        training_time = str(datetime.timedelta(seconds=int(round(time.time() - t0))))
        print(f'Training epoch took: {training_time}')

        # evaluation
        test_accuracy = evaluation(test_dataloader, model, device)
        print(f'test accuracy: {test_accuracy}')

        # record test accuracy
        with open('/media/dell/disk/jk/Retrieval/run/q_Qd_test/test_accuracy.txt', 'a+') as f:
            f.writelines('test accuracy: ' + str(test_accuracy))

    print('Training Completed!')
    
    model.eval()
    # save model params
    model_save_path  = os.path.join(BERT_MODEL_SOTRE_PATH, 'q_a_BertletsModel_params.pkl')
    torch.save(model.state_dict(), model_save_path)

if __name__ == "__main__":
    # train_q_Qd_Bertlets()
    train_q_a_Bertlets()
