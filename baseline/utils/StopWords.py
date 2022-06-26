import os
import sys

if os.path.exists('D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'):
    base_dir = 'D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'
else:
    base_dir = '/media/dell/disk/jk/Retrieval'
sys.path.append(base_dir)

from baseline.pathConfig import get_base_path
from nltk import word_tokenize

path_of_stopwords_EN = base_dir + '/baseline/utils/StopWords_EN.txt'

def read_EN_stopwords():
    sw_set = set()
    with open(path_of_stopwords_EN) as f:
        for line in f:
            sw_set.add(line.strip())
        return sw_set

def remove_stopwords(sent, sw):
    if isinstance(sent, str):
        wlist = word_tokenize(sent)
    elif isinstance(sent, list):
        wlist = sent
    else:
        raise Exception('Wrong type for removing stopwords!')
    sent_words = []
    for w in wlist:
        if w == '':
            continue
        if w not in sw:
            sent_words.append(w)
    return sent_words

if __name__ == "__main__":
    print(path_of_stopwords_EN)