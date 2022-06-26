import os
import sys

if os.path.exists('D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'):
    base_dir = 'D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'
else:
    base_dir = '/media/dell/disk/jk/Retrieval'
sys.path.append(base_dir)

from baseline.sentence_selection.Entity.entity_util import load_entity_set

def get_entity_score(query_entities, paragraph):
    paragraph = paragraph.split(' ')
    total_num = len(query_entities)
    include_num = 0.0
    if total_num == 0:
        return 1.5
    for query_entity in query_entities:
        if query_entity in paragraph:
            include_num += 1.0
    return 1.0 + (include_num / total_num)

def get_entities_from_word_list(word_list):
    entity_dic = load_entity_set()
    entity_list = []
    for word in word_list:
        if word in entity_dic:
            entity_list.append(word)
    return entity_list