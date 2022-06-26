import os
import sys
import pandas as pd

if os.path.exists('D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'):
    base_dir = 'D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'
else:
    base_dir = '/media/dell/disk/jk/Retrieval'
sys.path.append(base_dir)

from baseline.utils.file_util import write_file

voc_path = base_dir + "/baseline/sentence_selection/Entropy/idf_voc.txt"

def read_voc():
    voc = {}
    with open(voc_path) as file:
       for line in file:
           word_idf = line.split('   ')
           word = word_idf[0]
           idf = float(word_idf[1].strip())
           voc[word] = idf
    return voc

if __name__ == "__main__":
    idf_csv_path = base_dir + "/baseline/question_retrieval/IDF_vocabulary/idf_vocab.csv"
    df_idf = pd.read_csv(idf_csv_path, low_memory=False)
    voc_str = ''
    voc = {}
    for word in df_idf.itertuples():
        voc[word[1]] = word[2]
    for key in voc.keys():
        voc_str += (str(key) + '   ' + str(voc[key]) + '\n')
    write_file(voc_path, voc_str.strip())
    print('Completed!')