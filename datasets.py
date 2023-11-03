# ==============================================================================
# CSE 842
# Project: Article Text Categorization
#
# Authors: Yue Deng, Josh Erno, Christopher Nosowsky
#
# ==============================================================================
import constants
from enum import Enum


class Datasets(Enum):
    NEWS_20 = constants.NEWS_20
    NEWS_AG = constants.NEWS_AG
    BOTH = constants.BOTH

