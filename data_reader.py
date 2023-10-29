# ==============================================================================
# CSE 842
# Project: Article Text Categorization
#
# Authors: Yue Deng, Josh Erno, Christopher Nosowsky
#
# ==============================================================================
import os
import string
import numpy as np
import pandas as pd
import nltk
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from constants import *
from features import *

################################################
# PREPROCESSING TO DO LIST AS PROJECT DEVELOPS
# TODO: Remove numerics?
# TODO: Add stemmers?
################################################


class DataReader:

    def __init__(self, feature=BOW):
        self.vocab = None
        self.data_train = None
        self.data_test = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.classes = None
        self.class_mapping = None
        self.feature = feature
        self.nltk_download_check()

    @staticmethod
    def nltk_download_check():
        installed_corpora = os.listdir(nltk.data.find("corpora"))

        print("Checking if NLTK corpora requirements are installed..")
        for corpus in NLTK_CORPUS:
            if corpus not in installed_corpora:
                print("The '" + corpus + "' corpus is not downloaded. Downloading...")
                nltk.download(corpus)
                print("Downloaded the '" + corpus + "' corpus")
            else:
                print("The '" + corpus + "' corpus is downloaded.")

    def convert_file_dir_to_csv(self):
        # code used to combine data from folders and save
        test_data = []
        train_data = []
        # iterate through test and train folders
        for i, data_folder in enumerate(os.listdir('data/20newsgroup')):
            # iterate through each label folder
            for j, label_folder in enumerate(os.listdir(os.path.join('data/20newsgroup', data_folder))):
                # iterate through each file
                for k, file in enumerate(os.listdir(os.path.join('data/20newsgroup', data_folder, label_folder))):
                    # read in file
                    f = os.path.join('data/20newsgroup', data_folder, label_folder, file)
                    f = open(f, 'r')
                    f = f.read()
                    # append to train/test list
                    test_data.append([f, [label_folder]]) if i == 0 else train_data.append([f, label_folder])
                    x = 10
                print('done', label_folder)
        test_df = pd.DataFrame(test_data)
        train_df = pd.DataFrame(train_data)
        test_df.to_csv('data/20newsgroup/20_news_test.csv')
        train_df.to_csv('data/20newsgroup/20_news_train.csv')
        np.savetxt('data/20newsgroup/20_news_pre_processed_test.csv', self.data_test)
        np.savetxt('data/20newsgroup/20_news_pre_processed_train.csv', self.data_train)

        test_df = pd.DataFrame(self.data_test)
        train_df = pd.DataFrame(self.data_train)
        test_df.to_csv('data/20newsgroup/20_news_pre_processed_test.csv')
        train_df.to_csv('data/20newsgroup/20_news_pre_processed_train.csv')

        self.data_test = np.array(pd.read_csv('data/20newsgroup/20_news_pre_processed_test.csv'))[:, 1:]
        self.data_train = np.array(pd.read_csv('data/20newsgroup/20_news_pre_processed_train.csv'))[:, 1:]

    def open_dataset(self, debug=False):
        if debug:
            top_rows = 1000
        else:
            top_rows = None

        self.data_test = np.array(pd.read_csv('data/20newsgroup/20_news_test.csv', nrows=top_rows))[:, 1:]
        self.data_train = np.array(pd.read_csv('data/20newsgroup/20_news_train.csv', nrows=top_rows))[:, 1:]

        self.data_test[:, 0] = self.preprocess(self.data_test)
        self.data_train[:, 0] = self.preprocess(self.data_train)
        self.classes = set(list(self.data_test[:, 1]))

        np.random.shuffle(self.data_test)
        np.random.shuffle(self.data_train)

        label_encoder = LabelEncoder()

        self.x_train = self.data_train[:, 0]
        y_train_words = self.data_train[:, 1]
        self.y_train = label_encoder.fit_transform(y_train_words)

        self.x_test = self.data_test[:, 0]
        y_test_words = self.data_test[:, 1]
        self.y_test = label_encoder.fit_transform(y_test_words)

        self.class_mapping = dict(zip(self.y_train, y_train_words))

    def preprocess(self, data):
        processed_data = []
        for i in range(data.shape[0]):
            # break text into list of words
            tokens = nltk.word_tokenize(data[i][0])
            # make text lowercase
            lowercased_tokens = [token.lower() for token in tokens]
            # remove punctuation
            filtered_tokens = [token for token in lowercased_tokens if token not in string.punctuation]
            # remove stopwords
            stopwords = nltk.corpus.stopwords.words("english")
            filtered_tokens = [token for token in filtered_tokens if token not in stopwords]
            text = ' '.join(filtered_tokens)
            processed_data.append(text)
        processed_data = np.array(processed_data)
        return processed_data

    def build_vocab(self):
        text = []
        for i, words in enumerate(self.data_train[:, 0]):
            text += self.data_train[i][0].split(' ') + self.data_train[i][0].split(' ')
        text = set(text)
        vectorizer = CountVectorizer()
        vectorizer.fit_transform(text)
        self.vocab = vectorizer.get_feature_names_out()

    def build_feature_set(self):
        if self.feature == Features.BOW:
            print("Creating BoW Feature")
            self.x_train, self.x_test = self.generate_bow_feature()
        if self.feature == Features.NGRAMS:
            print("Creating NGRAMS Feature")
            # TODO TBD Later stage of project
        if self.feature == Features.TFIDF:
            print("Creating TFIDF Feature")
            # TODO TBD Later stage of project

    def generate_bow_feature(self):
        bow_vectorizer = CountVectorizer(vocabulary=self.vocab)
        x_train = bow_vectorizer.fit_transform(self.x_train)
        x_test = bow_vectorizer.fit_transform(self.x_test)
        return x_train.toarray(), x_test.toarray()


