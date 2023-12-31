# ==============================================================================
# CSE 842
# Project: Article Text Categorization
#
# Authors: Yue Deng, Josh Erno, Christopher Nosowsky
#
# ==============================================================================
import torch
import numpy as np
from constants import *
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from torch.utils.data import DataLoader, TensorDataset


class Evaluate:
    """
    Evaluate predictions
    """

    def __init__(self, model, x_test, y_test, device=None):
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
        self.device = device

    def predict(self, model_name):
        """
        Predictions function for predicting the labels, given test data
        :param model_name: the model trained
        :return: the predicted results
        """
        if model_name == KERAS_MODEL:
            return np.argmax(self.model.predict(self.x_test), axis=1)
        elif model_name == BERT_MODEL:
            predictions, true_labels = self.bert_evaluate_predict()
            return predictions, true_labels
        else:
            return self.model.predict(self.x_test)

    def bert_evaluate_predict(self):
        """
        Evaluates the performance of a BERT-based model on the provided test data
        and returns predictions and true labels.

        This function tokenizes the input test data using the BERT model's tokenizer,
        constructs dataloaders for prediction,
        and then performs inference on the test set.

        The model is put in evaluation mode, and predictions are generated for each batch of test data.

        :return: tuple: A tuple containing two lists - predictions and true_labels.
        - predictions (list): A list of numpy arrays,
        where each array contains the predicted labels for a batch of test data.
        - true_labels (list): A list of numpy arrays,
        where each array contains the true labels for a batch of test data.
        """
        tokenized_test = self.model.tokenizer(self.x_test, padding=True, truncation=True, return_tensors="pt",
                                              max_length=128)

        labels = torch.tensor(self.y_test, dtype=torch.int64).to(self.device)
        input_ids = tokenized_test["input_ids"].to(self.device)
        attention_mask = tokenized_test["attention_mask"].to(self.device)

        batch_size = 32

        prediction_data = TensorDataset(input_ids, attention_mask, labels)
        prediction_dataloader = DataLoader(prediction_data, batch_size=batch_size, shuffle=True)

        # Prediction on test set

        print('Predicting labels for {:,} test sentences...'.format(len(input_ids)))

        # Put model in evaluation mode
        self.model.eval()

        # Tracking variables
        predictions, true_labels = [], []

        # Predict
        for batch in prediction_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch

            # Telling the model not to compute or store gradients, saving memory and
            # speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.model(b_input_ids, token_type_ids=None,
                                     attention_mask=b_input_mask)

            # Move logits and labels to CPU
            logits = outputs.logits
            label_ids = b_labels.to('cpu').numpy()
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            # Store predictions and true labels
            predictions.append(preds)
            true_labels.append(label_ids)
        return predictions, true_labels

    def evaluate(self, predictions):
        """
        Basic evaluation with the predictions
        :param predictions: Np array of the predicted output
        """
        self.accuracy = accuracy_score(self.y_test, predictions)
        self.precision = precision_score(self.y_test, predictions, average=None)
        self.recall = recall_score(self.y_test, predictions, average=None)
        self.f1 = f1_score(self.y_test, predictions, average=None)

    def evaluate_classification_report(self, model_name, label_encoder, predictions, true_labels=None):
        """
        Basic evaluation with the predictions
        :param model_name: The trained model
        :param label_encoder: The label encoder object
        :param predictions: The model predictions
        :param true_labels: The true labels (for BERT only)
        :return: The classification report
        """
        if model_name == BERT_MODEL and true_labels is not None:
            flat_predictions = np.concatenate(predictions, axis=0)
            flat_true_labels = np.concatenate(true_labels, axis=0)

            print("The models predictions...")
            print(label_encoder.inverse_transform(flat_predictions))
            print("\nThe test predictions...")
            print(label_encoder.inverse_transform(flat_true_labels))
            print("\n")

            return classification_report(label_encoder.inverse_transform(flat_true_labels),
                                         label_encoder.inverse_transform(flat_predictions))
        else:
            return classification_report(label_encoder.inverse_transform(self.y_test),
                                         label_encoder.inverse_transform(predictions))
