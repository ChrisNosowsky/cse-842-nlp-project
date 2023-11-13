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
NEWS_20_PATH_TRAIN = "data/20newsgroup/20_news_train.csv"
NEWS_20_PATH_TEST = "data/20newsgroup/20_news_test.csv"
AG_NEWS_PATH_TRAIN = ""
AG_NEWS_PATH_TEST = ""

# MISC CONSTANTS
CLASS_TO_LABEL_MAPPING = {}
NLTK_CORPUS = ["punkt", "stopwords"]
TOP_VOCAB_WORDS = 15000
