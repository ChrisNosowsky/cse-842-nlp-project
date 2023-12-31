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
DOC2VEC = 3
WORD2VEC = 4

# DATASET CONSTANTS
NEWS_20 = 0
NEWS_AG = 1
BOTH = 2

# MODEL CONSTANTS
KERAS_MODEL = "Keras FCNN Model"
NAIVE_BAYES_MODEL = "Naive Bayes Model"
BERT_MODEL = "BERT Model"
LOG_REG_MODEL = "Logistic Regression Model"

# FILEPATH CONSTANTS
NEWS_20_DIR = "data/20newsgroup"
AG_NEWS_DIR = "data/ag_news"
PREPROCESS_DIR = "data/preprocess"
NEWS_20_PATH_TRAIN = NEWS_20_DIR + "/20_news_train.csv"
NEWS_20_PATH_TEST = NEWS_20_DIR + "/20_news_test.csv"
AG_NEWS_PATH_TRAIN = AG_NEWS_DIR + "/train.csv"
AG_NEWS_PATH_TEST = AG_NEWS_DIR + "/test.csv"

# MISC CONSTANTS
NLTK_CORPUS = ["punkt", "stopwords", "wordnet"]
DEFAULT_TEST_SIZE = 'default'
NGRAMS_SIZE = 2
TOP_VOCAB_WORDS = 15000
MIN_CHARS_TO_REMOVE_FROM_TOKENS = 1     # e.g. 1 = single words + blanks removed from preprocessed dataset
AG_NEWS_CLASS_MAPPING = {
    1: 'World',
    2: 'Sports',
    3: 'Business',
    4: 'Sci/Tech'
}
