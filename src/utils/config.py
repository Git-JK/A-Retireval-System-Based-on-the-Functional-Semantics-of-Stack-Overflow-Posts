import os

if os.path.exists('D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'):
    base_dir = 'D:/Github/A-retrieval-system-based-on-Stack-Overflow-Posts-functional-semantics'
else:
    base_dir = '/media/dell/disk/jk/Retrieval'

# MYSQL parameters
Mysql_addr = "162.105.16.191"
Mysql_user = "root"
Mysql_password = "root"
Mysql_dbname_sotorrent = "sotorrent"

SO_POSTS_STORE_PATH = f'{base_dir}/data/so_posts/'
TAG_STORE_PATH = f'{base_dir}/src/dictionary/'
TRAINSET_STORE_PATH = f'{base_dir}/data/trainset_posts/'
SEARCHSET_STORE_PATH = f'{base_dir}/data/searchset_posts/'
TESTSET_STORE_PATH = f'{base_dir}/data/testset_posts/'
W2V_CORPUS_STORE_PATH = f'{base_dir}/src/word2vec/corpus/'
PRETRAINED_W2V_MODEL_STORE_PATH = f'{base_dir}/src/word2vec/model/'
STOPWORDS_PATH = f'{base_dir}/src/dictionary/'
BERT_Q_QD_DATA_STORE_PATH = f'{base_dir}/data/bert-q-Qd_data/'
BERT_Q_A_DATA_STORE_PATH = f'{base_dir}/data/bert-q-a_data/'
Q_QD_RUN_DATA_PATH = f'{base_dir}/run/q_Qd/'
Q_A_RUN_DATA_PATH = f'{base_dir}/run/q_a/'
BERT_MODEL_SOTRE_PATH = f'{base_dir}/src/bert/model/'
BERT_Q_QD_TOKEN_STORE_PATH = f'{base_dir}/data/bert-q-Qd_token/'
BERT_Q_A_TOKEN_STORE_PATH = f'{base_dir}/data/bert-q-a_token/'
BERT_Q_QD_TEST_DATA_STORE_PATH = f'{base_dir}/data/bert-q-Qd_test_data/'
BERT_Q_A_TEST_DATA_STORE_PATH = f'{base_dir}/data/bert-q-a_test_data/'
BERT_Q_QD_TEST_TOKEN_STORE_PATH = f'{base_dir}/data/bert-q-Qd_test_token/'
BERT_Q_A_TEST_TOKEN_STORE_PATH = f'{base_dir}/data/bert-q-a_test_token/'
TEST_RESULTS_STORE_PATH = f'{base_dir}/data/test_results/'