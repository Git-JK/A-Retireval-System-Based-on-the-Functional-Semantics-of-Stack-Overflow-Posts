import os
import sys

if os.path.exists('D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'):
    base_dir = 'D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'
else:
    base_dir = '/media/dell/disk/jk/Retrieval'
sys.path.append(base_dir)

from baseline.utils.Stemming import stemming_for_word_list
from baseline.utils.StopWords import remove_stopwords

def get_entropy_score(query_words, title_words, stopwords, idf_voc):
    # remove stopwords
    title_words = remove_stopwords(title_words, stopwords)
    # remove duplicate word in list
    query_words = list(set(query_words))
    title_words = list(set(title_words))
    # stemming
    query_words = stemming_for_word_list(query_words)
    title_words = stemming_for_word_list(title_words)
    # remove query word
    for word in query_words:
        if word in title_words:
            title_words.remove(word)
    voc = idf_voc
    entropy = 0.0
    for word in title_words:
        if word in voc.keys():
            entropy += float(voc[word])
    return entropy