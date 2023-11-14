# ==============================================================================
# CSE 842
# Constants used in assignment
#
# Authors: Yue Deng, Josh Erno, Christopher Nosowsky
#
# ==============================================================================

# FEATURE CONSTANTS
BOW = 0
NGRAMS = 1
TFIDF = 2
DOC2VEC = 3

# DATASET CONSTANTS
NEWS_20 = 0
NEWS_AG = 1
BOTH = 2

# MODEL CONSTANTS
KERAS_MODEL = "Keras FCNN Model"
NAIVE_BAYES_MODEL = "Naive Bayes Model"
RIPPER_MODEL = "RIPPER Model"
BERT_MODEL = "BERT Model"

# FILEPATH CONSTANTS
NEWS_20_DIR = "data/20newsgroup"
AG_NEWS_DIR = "data/ag_news"
NEWS_20_PATH_TRAIN = NEWS_20_DIR + "/20_news_train.csv"
NEWS_20_PATH_TEST = NEWS_20_DIR + "/20_news_test.csv"
AG_NEWS_PATH_TRAIN = AG_NEWS_DIR + "/train.csv"
AG_NEWS_PATH_TEST = AG_NEWS_DIR + "/test.csv"

# MISC CONSTANTS
NLTK_CORPUS = ["punkt", "stopwords", "wordnet"]
DEFAULT_TEST_SIZE = 'default'
TOP_VOCAB_WORDS = 15000
