import os
import sys

if os.path.exists('D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'):
    base_dir = 'D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'
else:
    base_dir = '/media/dell/disk/jk/Retrieval'
sys.path.append(base_dir)

import pandas as pd
from baseline.summarization.MMR_Analysis import MMR_Analysis
from pathConfig import res_dir
from baseline.utils.data_util import preprocessing_for_query
from baseline.utils.csv_utils import write_list_to_csv

def get_summary(query, top_ss, topk):
    selected_sentence = MMR_Analysis(query, top_ss, topk)
    summary = [x for x in selected_sentence]
    return summary

def load_ss_result(ss_fpath):
    ss_res = []
    df = pd.read_csv(ss_fpath)
    for idx, row in df.iterrows():
        ss_res.append((row[0], eval(row[1])))
    return ss_res

if __name__ == "__main__":
    ss_fpath = os.path.join(res_dir, 'ss_res.csv')
    topk = 5
    res = []
    for query, ss in load_ss_result(ss_fpath):
        query = ' '.join(preprocessing_for_query(query))
        sum = get_summary(query, ss, topk)
        res.append([query, sum])
        print(f'summary\m{sum}')
    
    res_fpath = os.path.join(res_dir, 'summary_res.csv')
    header = ['query', 'summary']
    write_list_to_csv(res, res_fpath, header)