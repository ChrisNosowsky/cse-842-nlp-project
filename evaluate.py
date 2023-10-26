# ==============================================================================
# CSE 842
# Project: Article Text Categorization
#
# Authors: Yue Deng, Josh Erno, Christopher Nosowsky
#
# ==============================================================================



class Evaluate:
    """
    Evaluate predictions
    """

    def __init__(self):
        """
        Constructor
        """
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0

    def evaluate(self, predictions, actual):
        pass