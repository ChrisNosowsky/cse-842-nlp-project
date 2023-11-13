# ==============================================================================
# CSE 842
# Project: Article Text Categorization
#
# Authors: Yue Deng, Josh Erno, Christopher Nosowsky
#
# ==============================================================================
import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from transformers import BertTokenizer
from constants import *


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

    def predict(self, model_name):
        if model_name == KERAS_MODEL:
            return np.argmax(self.model.predict(self.x_test), axis=1)
        elif model_name == BERT_MODEL:
            model_name = 'bert-base-uncased'
            tokenizer = BertTokenizer.from_pretrained(model_name)
            model = self.model

            inputs = tokenizer(self.x_test, padding=True, truncation=True, return_tensors="pt", max_length= 128)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                predicted_labels = torch.argmax(logits, dim=1)
            predicted_labels = predicted_labels.tolist()
            print(predicted_labels)

            # todo (Yue, 11/10/2023, done): The labels of true are 1, 2, 3, for example, while those of predicted here may be 3, 1, 2 correspondingly. That matters?
            #  YES! That matters except for accuracy. So, we need to do training.
            # textInput = []
            # for this_x_test in self.x_test:
            #     thisInput = self.model.tokenizer(this_x_test, return_tensors="pt", padding=True, truncation=True,  max_length = 128)
            #     thisTextInput = self.model.tokenizer.decode(thisInput['input_ids'][0])
            #     textInput.append(thisTextInput)
            # output = self.model(textInput)
            #extractedLabels = [int(thisLabel[6:]) for thisLabel in predicted_labels]

            return predicted_labels
        else:
            return self.model.predict(self.x_test)

    def evaluate(self, predictions):
        self.accuracy = accuracy_score(self.y_test, predictions)
        self.precision = precision_score(self.y_test, predictions, average=None)
        self.recall = recall_score(self.y_test, predictions, average=None)
        self.f1 = f1_score(self.y_test, predictions, average=None)

    def plot_results(self):
        # TODO: TBD Later stage of project
        pass

