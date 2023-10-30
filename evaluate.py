# ==============================================================================
# CSE 842
# Project: Article Text Categorization
#
# Authors: Yue Deng, Josh Erno, Christopher Nosowsky
#
# ==============================================================================
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


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
        # Compute the number of correct predictions
        correct_predictions = np.sum(predictions.argmax(axis=1) == self.y_test)

        # Calculate accuracy
        accuracy = correct_predictions / len(self.y_test)

        # Calculate precision, recall, and F1 score
        precision = precision_score(self.y_test, predictions.argmax(axis=1), average='weighted')
        recall = recall_score(self.y_test, predictions.argmax(axis=1), average='weighted')
        f1 = f1_score(self.y_test, predictions.argmax(axis=1), average='weighted')

        print("Test accuracy of model: " + str(accuracy))
        print("Test precision of model: " + str(precision))
        print("Test recall of model: " + str(recall))
        print("Test f1 score of model: " + str(f1))

    def plot_results(self):
        # TODO: TBD Later stage of project
        pass
