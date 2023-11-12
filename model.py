# ==============================================================================
# CSE 842
# Project: Article Text Categorization
#
# Authors: Yue Deng, Josh Erno, Christopher Nosowsky
#
# ==============================================================================
import os
import wittgenstein as lw
from sklearn.naive_bayes import MultinomialNB
from tensorflow import keras
from data_reader import *
import tensorflow as tf
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

################################################
# MODELING TO DO LIST AS PROJECT DEVELOPS
# TODO: Tweak model parameters -- All team
# TODO: Add Ray Tune or Hyperopt tuning parameters -- Yue
# TODO: LM? (BERT pretrained models, maybe two? -- One for Yue TODO, One for Chris TODO
################################################


class KerasFCNNModel:
    def __init__(self, x_train, y_train, dataset=Datasets.BOTH):
        self.x_train = x_train
        self.y_train = y_train
        self.history = None
        self.dataset = dataset

    def learn(self):
        in_shape = self.x_train.shape[1]
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(in_shape, activation=tf.nn.relu, input_shape=(in_shape,)))
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        num_classes = 0
        if self.dataset == Datasets.NEWS_20:
            num_classes = 20
        elif self.dataset == Datasets.NEWS_AG:
            num_classes = 4
        else:
            num_classes = 24
        model.add(tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax))
        # learning_rate=0.001
        opt = keras.optimizers.Adam()
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.history = model.fit(self.x_train, self.y_train,
                                 epochs=7, batch_size=64)
        model.summary()
        train_acc = self.history.history['accuracy'][-1]
        train_loss = self.history.history['loss'][-1]
        print('Keras FCNN Training Accuracy: ' + str(train_acc))
        print('Keras FCNN Training Loss: ' + str(train_loss))
        return model

    def plot_train_accuracy(self):
        # TODO: TBD Later stage of project
        pass

    def plot_loss_accuracy(self):
        # TODO: TBD Later stage of project
        pass

class NaiveBayesModel:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def learn(self):
        print('Training Naive Bayes model...')
        model = MultinomialNB()
        model.fit(self.x_train, self.y_train)
        print('Naive Bayes model trained')

        return model

class RIPPERModel:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def learn(self):
        print('Training Ripple model...')
        model = lw.RIPPER()
        model.fit(self.x_train, self.y_train, class_feat='Poisonous/Edible', pos_class='p')
        print('Ripple model trained')

        return model

class BERTModel:
    def __init__(self, dataset, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        if dataset == NEWS_20:
            self.numClasses = 20
        elif dataset == NEWS_AG:
            self.numClasses = 4
        else:
            self.numClasses = 20 + 4
    def usePretrainedBert(self):
        print('Training BERT model...')
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=self.numClasses)

        inputs = tokenizer(self.x_train.tolist(), padding=True, truncation=True, return_tensors="pt", max_length= 128)
        labels = torch.tensor(self.y_train, dtype=torch.int64)

        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        batch_size = 3
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = AdamW(model.parameters(), lr=1e-5)

        num_epochs = 10
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            average_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")

        model.save_pretrained("bert_trained")
        print('BERT model trained')

        return model