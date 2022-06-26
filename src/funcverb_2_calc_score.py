from cmath import nan
from unittest import result
from funcverbnet.funcverbnet import FuncVerbNet
from funcverbnet.sentence_classifier import FuncSentenceClassifier
from utils.funcverb.funcverb_utils import find_root_of_dependency_tree, find_funcverb_of_dependency_tree, find_core_clause_of_funcverb, find_p_pattern_of_core_clause, replace_part_of_clause, find_word_in_a_tree
from utils.text_process.puretext_extract import preprocess_title, remove_stopwords
from utils.read.read_utils import load_w2v_model, read_EN_stopwords
from utils.config import PRETRAINED_W2V_MODEL_STORE_PATH
import numpy as np
import os
import spacy


def train_FuncSentenceClassifier():
    classifier = FuncSentenceClassifier()
    classifier.train_model()

def find_word(word, sentence):
    print(sentence)
    for w in sentence:
        if word == w.text:
            return w.i
    return -1

def get_p_pattern_of_sentence(sentence, f_category,  nlp, classifier):
    funcverb_candidates = classifier.find_all_verb_name_by_cate_id(f_category)
    sentence = sentence.lower()
    sentence_doc = nlp(sentence)
    sentence_root = find_root_of_dependency_tree(sentence_doc)
    funcverb = find_funcverb_of_dependency_tree(sentence_root, funcverb_candidates)
    funcverb_clause = find_core_clause_of_funcverb(funcverb, nlp)
    clause_root = find_root_of_dependency_tree(funcverb_clause)
    pattern_candidates = classifier.find_all_pattern_name_by_cate_id(f_category)
    p_pattern = find_p_pattern_of_core_clause(funcverb_clause, clause_root, nlp, pattern_candidates)
    return funcverb, funcverb_clause, p_pattern

def get_f_catogery_of_sentence(sentence, classifier):
    f_category = classifier.find_category_by_any_sentence(sentence)
    return f_category


def get_functionality_category_similarity_score(query_f_category, question_f_category):
    return (query_f_category == question_f_category)

def get_semantic_role_similarity_score(query_semantic_part_list, question_semantic_part_list, nlp, w2v_model):
    if len(query_semantic_part_list) == 0 or len(question_semantic_part_list) == 0:
        return 0.0
    sim_list = []
    sim_sum = 0
    i = 0
    last_place = 0
    query_length = len(query_semantic_part_list)
    question_length = len(question_semantic_part_list)
    for i in range(query_length):
        for j in range(last_place, question_length):
            if query_semantic_part_list[i][0] == question_semantic_part_list[j][0]:
                sim_list.append(get_semantic_part_text_similarity(" ".join(query_semantic_part_list[i][1]), " ".join(question_semantic_part_list[j][1]), w2v_model, nlp))
                last_place = j + 1
                break
    if len(sim_list) == 0:
        return 0
    for sim in sim_list:
        sim_sum += sim
    sim_avg = sim_sum / len(sim_list)
    return float(sim_avg)
    # print(query_p_pattern, query_semantic_part_list)
    # print(question_p_pattern, question_semantic_part_list)

    
        
def get_semantic_part_text_similarity(part1, part2, w2v_model, nlp):
    part1_doc = nlp(part1)
    part2_doc = nlp(part2)
    part1 = []
    part2 = []
    for word in part1_doc:
        part1.append(word.lemma_)
    for word in part2_doc:
        part2.append(word.lemma_)
    part1_vector = np.zeros((1, w2v_model.vector_size))
    part2_vector = np.zeros((1, w2v_model.vector_size))
    for word in part1:
        if word in w2v_model.wv.vocab:
            part1_vector += w2v_model.wv[word]
    for word in part2:
        if word in w2v_model.wv.vocab:
            part2_vector += w2v_model.wv[word]
    part1_vector /= len(part1)
    part2_vector /= len(part2)
    result = np.abs(np.dot(part1_vector, part2_vector.T) / (np.linalg.norm(part1_vector, ord=2) * np.linalg.norm(part2_vector, ord=2)))
    # print(result)
    if result[0][0] is nan:
        return 0
    else:
        return float(result)

def get_text_similarity_score(query, question, w2v_model, stopwords, nlp):
    # 对文本进行预处理
    query = preprocess_title(query)
    question = preprocess_title(question)
    # 去除停用词
    query = remove_stopwords(query, stopwords)
    question = remove_stopwords(question, stopwords)
    # 词形还原
    query_doc = nlp(' '.join(query))
    question_doc = nlp(' '.join(question))
    query = []
    question = []
    for word in query_doc:
        query.append(word.lemma_)
    for word in question_doc:
        question.append(word.lemma_)
    if len(query) == 0 or len(question) == 0:
        return 0.0
    # 计算query和question的word vector
    query_vector = np.zeros((1, w2v_model.vector_size))
    question_vector= np.zeros((1, w2v_model.vector_size))
    for word in query:
        if word in w2v_model.wv.vocab:
            query_vector += w2v_model.wv[word]
    for word in question:
        if word in w2v_model.wv.vocab:
            question_vector += w2v_model.wv[word]
    query_vector /= len(query)
    question_vector /= len(question)
    result = np.abs(np.dot(query_vector, question_vector.T) / (np.linalg.norm(query_vector, ord=2) * np.linalg.norm(question_vector, ord=2)))
    if result[0][0] is nan:
        return 0
    else:
        return float(result)

    

if __name__ == "__main__":
    classifier = FuncVerbNet()
    text1 = 'nothing will stand infront of me.'
    text2 = 'nothing will stand infront of me.'
    nlp= spacy.load('en_core_web_lg')
    w2v_model_path = os.path.join(PRETRAINED_W2V_MODEL_STORE_PATH, "w2v_model.model")
    w2v_model = load_w2v_model(w2v_model_path)
    stopwords = read_EN_stopwords()
    print(get_text_similarity_score(text1, text2, w2v_model, stopwords, nlp))
    # query_category = classifier.find_category_by_any_sentence(text1)
    # question_category = classifier.find_category_by_any_sentence(text2)
    # query_verb, query_clause, query_pattern = get_p_pattern_of_sentence(text1, query_category, nlp, classifier)
    # question_verb, question_clause, question_pattern = get_p_pattern_of_sentence(text2, question_category, nlp, classifier)
    # print(get_semantic_role_similarity_score(text1, " ".join([word.text for word in query_clause]), query_pattern, text2, " ".join([word.text for word in question_clause]), question_pattern, nlp, w2v_model))