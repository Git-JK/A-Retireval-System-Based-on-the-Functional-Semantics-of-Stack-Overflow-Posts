import os
import sys
import math
import operator

if os.path.exists('D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'):
    base_dir = 'D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'
else:
    base_dir = '/media/dell/disk/jk/Retrieval'
sys.path.append(base_dir)

from baseline.utils.db_util import read_all_questions_from_repo
from baseline.utils.time_utils import get_current_time
from baseline.utils.csv_utils import write_list_to_csv
from nltk import word_tokenize

def build_IDF_vocabulary():
    qlist = read_all_questions_from_repo()
    total_num = len(qlist)
    voc = {}
    count = 0
    for q in qlist:
        title_wlist = word_tokenize(q.title.strip())
        cur_word_set = set()
        for w in title_wlist:
            if w not in cur_word_set:
                cur_word_set.add(w)
                if w not in voc.keys():
                    voc[w] = 1.0
                else:
                    voc[w] = voc[w] + 1.0
        count += 1
        if count % 10000 == 0:
            print(f'processing {count} unit...', get_current_time())
    for key in voc.keys():
        idf = math.log(total_num / (voc[key] + 1.0))
        voc[key] = idf
    sorted_voc = sorted(voc.items(), key=operator.itemgetter(1))
    return sorted_voc

if __name__ == "__main__":
    fpath = base_dir + '/baseline/question_retrieval/IDF_vocabulary/idf_vocab.csv'
    header = ['word', 'idf']
    vocab = build_IDF_vocabulary()
    write_list_to_csv(vocab, fpath, header)
    print('Completed!')