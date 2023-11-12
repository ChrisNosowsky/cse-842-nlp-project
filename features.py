# ==============================================================================
# CSE 842
# Project: Article Text Categorization
#
# Authors: Yue Deng, Josh Erno, Christopher Nosowsky
#
# ==============================================================================
import constants
from enum import Enum


class Features(Enum):
    BOW = constants.BOW
    NGRAMS = constants.NGRAMS
    TFIDF = constants.TFIDF
    DOC2VEC = constants.DOC2VEC
