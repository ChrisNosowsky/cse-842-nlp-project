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

    def __init__(self, model, x_test, y_test):
        """
        Constructor
        """
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.accuracy = 0.0
        self.precision = 0.0
        self.recall = 0.0
        self.f1 = 0.0

    def predict(self):
        return self.model.predict(self.x_test)

    def evaluate(self, predictions):
        pass