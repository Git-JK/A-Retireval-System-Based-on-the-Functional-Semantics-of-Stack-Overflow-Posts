from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet

def find_root_of_dependency_tree(tree):
    if tree is None:
        return None
    if tree.__len__() < 1:
        return None
    tok = tree[0]
    while tok.head != tok:
        tok = tok.head
    return tok


def find_first_verb_of_dependency_tree(root, verb_type):
    first_verb = ''
    if root.pos_ == verb_type:
        first_verb = root
        return first_verb
    for child in root.children:
        child_result = find_first_verb_of_dependency_tree(child, verb_type)
        if child_result is not None:
            first_verb = child_result
            return first_verb
    return None


def find_funcverb_of_dependency_tree(root, verb_candidates):
    funcverb = recursive_check_funcverb(root, verb_candidates)
    if funcverb is None:
        funcverb = find_first_verb_of_dependency_tree(root, 'VERB')
    if funcverb is None:
        funcverb = find_first_verb_of_dependency_tree(root, 'AUX')
    return funcverb


def recursive_check_funcverb(root, verb_candidates):
    funcverb = ''
    origin_root = ''
    if root.pos_ in ['VERB', 'AUX']:
        origin_root = WordNetLemmatizer().lemmatize(root.text, wordnet.VERB)
    if origin_root in verb_candidates:
        funcverb = root
        return funcverb
    for child in root.children:
        child_result = recursive_check_funcverb(child, verb_candidates)
        if child_result is not None:
            funcverb = child_result
            return funcverb
    return None


def find_core_clause_of_funcverb(funcverb, nlp):
    if funcverb is None:
        return None
    clause = [word for word in funcverb.subtree]
    core_clause = []
    for i in range(len(clause)):
        if clause[i].text == funcverb.text:
            core_clause = clause[i:]
            break
    return nlp(' '.join([word.text for word in core_clause]))


def has_subj(word_list):
    # 判断是否有主语
    subj_tag_set = ['subj', 'nsubj', 'npsubj', 'csubj', 'xsubj']
    for word in word_list:
        if word.dep_ in subj_tag_set:
            return True
    return False


def has_pred(word_list):
    # 判断是否有谓语
    for word in word_list:
        if word.dep_ == 'ROOT' and word.pos_ == 'VERB':
            return True
    return False


def check_sentence(clause_list):
    # 有主语(subj)和谓语(pred)就一定是句子
    return has_subj(clause_list) and has_pred(clause_list)


def has_clause_leader(word_list):
    for word in word_list:
        if word.dep_ == 'mark':
            return word.i
    return -1

def find_word_in_a_tree(word, tree):
    # 寻找一个词在tree中的位置，找不到就返回-1
    for w in tree:
        if word.text == w.text:
            return w.i
    return -1

def replace_part_of_clause(tree, root, nlp):
    # print((root.i, root, root.dep_, root.pos_, root.tag_), [(child.i, child, child.dep_, child.pos_, child.tag_, [word for word in child.subtree]) for child in root.children])
    syntatic_pattern = []
    # 将funcverb_clause的核心funcverb替换
    syntatic_pattern.append((root.i, 'V'))
    # 处理root的子树,基准位置是child在原句中的index
    for child in root.children:
        sent = nlp(' '.join([subword.text for subword in child.subtree]))
        sent_root = find_root_of_dependency_tree(sent)
        sent_root_i = find_word_in_a_tree(sent_root, tree)
        syntatic_pattern += replace_subtree(sent_root_i, nlp(' '.join([subword.text for subword in child.subtree])), nlp)
    syntatic_pattern_tmp = sorted(syntatic_pattern, key=lambda x:x[0])
    syntatic_pattern = []
    NP_flag = 0
    for i in range(len(syntatic_pattern_tmp)):
        token = syntatic_pattern_tmp[i][1]
        if NP_flag == 0:
            if token == 'NP':
                NP_flag = 1
            syntatic_pattern.append(syntatic_pattern_tmp[i])
        else:
            if token == 'NP':
                continue
            else:
                NP_flag = 0
                syntatic_pattern.append(syntatic_pattern_tmp[i])
    # print(syntatic_pattern)
    final_pattern = []
    i = 0
    while i < len(syntatic_pattern):
        index_1 = syntatic_pattern[i][0]
        token_1 = syntatic_pattern[i][1]
        if token_1 == 'to':
            if i == len(syntatic_pattern) - 1:
                final_pattern.append(syntatic_pattern[i])
                i += 1
                continue
            index_2 = syntatic_pattern[i + 1][0]
            token_2 = syntatic_pattern[i + 1][1]
            if token_2 == 'V' and index_2 == index_1 + 1:
                final_pattern.append((index_1, 'S_INF'))
                i += 2
                continue
        final_pattern.append(syntatic_pattern[i])
        i += 1
    return final_pattern


# start_place是root在最原始句中的位置
def replace_subtree(start_place, tree, nlp):
    syntatic_pattern = []
    root = find_root_of_dependency_tree(tree)
    # print([word for word in tree], root)
    # print(start_place, (root.i, root, root.dep_, root.pos_, root.tag_), [(child.i, child, child.dep_, child.pos_, child.tag_, [word for word in child.subtree]) for child in root.children])
    # root是名词，限定词，量词，代词或专属名词(如果含有谓语，那么root一定是谓语),那么是一个名词词组，需要看看有没有介词/助词，看看要不要因为介词拆分
    if root.pos_ == 'NOUN' or root.pos_ == 'DET' or root.pos_ == 'NUM' or root.pos_ == 'PRON' or root.pos_ == 'PROPN':
        split_sentence_list, prep_list = check_prep(tree, nlp)
        if len(prep_list) == 0:
            syntatic_pattern.append((start_place, 'NP'))
            return syntatic_pattern
        else:
            for prep in prep_list:
                syntatic_pattern.append((start_place - root.i + prep.i, prep.text))
            for (sent_root_pos, sent) in split_sentence_list:
                syntatic_pattern += replace_subtree(start_place - root.i + sent_root_pos, sent, nlp)
            return syntatic_pattern
    # root是介词或助词所属中的to或者介词，那么root不可能是谓语，且需要进一步切分(需要保留介词)
    elif root.pos_ == 'ADP' or (root.pos_ == 'PART'and root.tag_ in ['TO', 'IN']):
        syntatic_pattern.append((start_place, root.text))
    # root是动词或者助动词，可能是谓语
    elif root.pos_ == 'VERB' or root.pos_ == 'AUX':
        if root.tag_ == 'VBG':
            # 如果root是个VERB且是个VBG，那么这个VBG不可能是谓语，直接替换成S_ING,接着分
            syntatic_pattern.append((start_place, 'S_ING'))
        else:
            # VERB不是VBG，那么是谓语
            # 如果是完整句子，那么clause leader词应该会在句首
            if check_sentence(tree):
                mark_pos = has_clause_leader(tree)
                if mark_pos != -1:
                    syntatic_pattern.append((start_place - root.i + mark_pos, tree[mark_pos].text))
                syntatic_pattern.append((start_place, 'S'))
                return syntatic_pattern
            # 既不是句子，也不是VBG，那必然是谓语，那只能接着划分
            syntatic_pattern.append((start_place, 'V'))
        # 下一个绝对位置是当前的绝对位置(root)+孩子在当前子树的相对位置
    for child in root.children:
        sent = nlp(' '.join([subword.text for subword in child.subtree]))
        sent_root = find_root_of_dependency_tree(sent)
        sent_root_i = find_word_in_a_tree(sent_root, tree)
        syntatic_pattern += replace_subtree(start_place - root.i + sent_root_i, nlp(' '.join([subword.text for subword in child.subtree])), nlp)
    return syntatic_pattern


def check_prep(clause_list, nlp):
    # 助词PART中包含介词，和to
    # 判断名词词组要不要因为介词拆分，如果拆分就按介词分段拆分，返回一个list，包含拆分的段（分段的root在句子中的位置和分段内容）
    prep_list = []
    word_stack = []
    start_pos = 0
    split_sentence_list = []
    for word in clause_list:
        if word.dep_ == 'prep' or (word.pos_ == 'PART' and word.tag_ in ['TO', 'IN']):
            prep_list.append(word)
            if len(word_stack) != 0:
                sentence = nlp(' '.join([w.text for w in word_stack]))
                sentence_root = find_root_of_dependency_tree(sentence)
                split_sentence_list.append((start_pos + sentence_root.i, sentence))
            word_stack = []
            start_pos = word.i + 1
        else:
            word_stack.append(word)
    if len(word_stack) != 0:
        sentence = nlp(' '.join([w.text for w in word_stack]))
        sentence_root = find_root_of_dependency_tree(sentence)
        split_sentence_list.append((start_pos + sentence_root.i, sentence))
        del word_stack
    return split_sentence_list, prep_list


def find_p_pattern_of_core_clause(tree, root, nlp, p_pattern_candidates):
    # 计算sp的components
    if tree is None or root is None:
        return None
    sp = [token[1] for token in replace_part_of_clause(tree, root, nlp)]
    sp_components = []
    i = 0
    while i < len(sp):
        word = nlp(sp[i])[0]
        if (word.pos_ == 'ADP' or (word.pos_ == 'PART' and word.tag_ in ['TO', 'IN'])) and i + 1 < len(sp):
            if sp[i + 1] == 'NP':
                sp_components.append(sp[i] + ' ' + sp[i + 1])
                i += 2
                continue
        sp_components.append(sp[i])
        i += 1
    # print([(word, word.pos_) for word in tree])
    # print(sp_components)
    pattern_candidate_dict = {}
    for pattern_candidate in p_pattern_candidates:
        # 计算pattern_candidate的components
        pattern_candidate_list = pattern_candidate.split(' ')
        pattern_candidate_components = []
        i = 0
        while i < len(pattern_candidate_list):
            word = nlp(pattern_candidate_list[i])[0]
            if (word.pos_ == 'ADP' or (word.pos_ == 'PART' and word.tag_ in ['TO', 'IN'])) and i + 1 < len(pattern_candidate_list):
                if pattern_candidate_list[i + 1][0] == '{':
                    pattern_candidate_components.append(pattern_candidate_list[i] + ' ' + pattern_candidate_list[i + 1])
                    i += 2
                    continue
            pattern_candidate_components.append(pattern_candidate_list[i])
            i += 1
        # print(pattern_candidate_components)
        # components对比
        i = 0
        not_match_flag = 0
        match_count = 0
        # 如果pattern_candidate_components长度大于sp_components，那么pattern_candidate_components中一定会有无法匹配的component，排除
        if len(pattern_candidate_components) > len(sp_components):
            pattern_candidate_dict[pattern_candidate] = [0, 1]
            continue
        while i < len(pattern_candidate_components):
            if not compare_component(sp_components[i], pattern_candidate_components[i]):
                not_match_flag = 1
            else:
                match_count += 1
            i += 1
        pattern_candidate_dict[pattern_candidate] = [match_count, not_match_flag]
        # print(sp_components, pattern_candidate_components, not_match_flag)

    # 选择最优的pattern_candidate作为p_pattern
    match_pattern_list = []
    not_match_pattern_list = []
    for pattern in pattern_candidate_dict.keys():
        if pattern_candidate_dict[pattern][1] == 0:
            match_pattern_list.append([pattern, pattern_candidate_dict[pattern][0]])
        else:
            not_match_pattern_list.append([pattern, pattern_candidate_dict[pattern][0]])
    match_pattern_list = sorted(match_pattern_list, key= lambda x:x[1], reverse=True)
    not_match_pattern_list = sorted(not_match_pattern_list, key= lambda x:x[1], reverse=True)
    # print(match_pattern_list)
    # print(not_match_pattern_list)
    if len(match_pattern_list) != 0:
        return match_pattern_list[0][0]
    else:
        return None


# p_pattern里有S，S_ING, S_INF,直接匹配即可
def compare_component(component1, component2):
    # 只可能是长度为1的component和长度为2的component
    component1_list = component1.split(' ')
    component2_list = component2.split(' ')
    # 长度不一致时，必然匹配失败
    if len(component1_list) != len(component2_list):
        return False
    if len(component1_list) == 1:
        # 长度为1且一样，那么匹配
        if component1 == component2:
            return True
        else:
            # 长度为1且一个为NP，一个为语义角色，那么匹配
            if component1 == 'NP' and component2[0] == "{":
                return True
            else:
                return False
    elif len(component1_list) == 2:
        if component1_list[0] in component2_list[0].split('/'):
            # print("\n", component1_list, component2_list, component2_list[0].split('/'), "\n")
            return True
        else:
            return False
    else:
        return False

