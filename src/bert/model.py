from turtle import forward
from transformers import BertModel
from torch import nn
import torch

class BertletsModel(nn.Module):

    def __init__(self, bert_model_type):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_type)
        self.linear = nn.Linear(self.bert.config.hidden_size, 1)
        nn.init.xavier_uniform_(self.linear.weight)

    def forward_once(self, input):
        # print(input.shape)
        # print(input[:, 0].shape)
        bert_out = self.bert(input_ids=input[:,0], token_type_ids=input[:,1], attention_mask=input[:,2])
        last_hidden = bert_out[0]
        # print(last_hidden.shape)
        score = self.linear(last_hidden[:,0])
        # print(score.shape)
        return score

    def forward(self, input):
        # print(input[:, 0].shape)
        pos_score = self.forward_once(input[:,0])
        neg_score = self.forward_once(input[:,1])
        return pos_score, neg_score

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super().__init__()
        self.margin = margin
    def forward(self, pos_score, neg_score):
        pos = torch.exp(pos_score) / (torch.exp(pos_score) + torch.exp(neg_score))
        neg = torch.exp(neg_score) / (torch.exp(pos_score) + torch.exp(neg_score))
        tmp = self.margin - (pos - neg)
        result = torch.max(torch.cat((tmp, torch.zeros_like(tmp)), 1), 1).values
        # print(torch.cat((tmp, torch.zeros_like(tmp)), 1))
        # print(torch.max(torch.cat((tmp, torch.zeros_like(tmp)), 1), 1).values)
        return torch.mean(result)


def load_bertlets_model(bert_model_type, bertlets_model_path):
    model = BertletsModel(bert_model_type)
    model.load_state_dict(torch.load(bertlets_model_path))
    model.eval()
    return model