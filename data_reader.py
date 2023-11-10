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
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from constants import *
from features import *
from datasets import *
from collections import Counter

################################################
# PREPROCESSING TO DO LIST AS PROJECT DEVELOPS
# TODO: Remove numerics? -- Josh
# TODO: Add stemmers? -- Chris
# TODO: Further preprocessing -- Josh
# TODO: Word2Vec -- Josh or Chris
################################################


class DataReader:

    def __init__(self, feature=BOW, top_vocab_words=False):
        self.vocab = None
        self.data_train = None
        self.data_test = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.classes = None
        self.class_mapping = None
        self.top_vocab_words = top_vocab_words
        self.feature = feature
        self.nltk_download_check()

        self.original_x_test = None

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

    def open_dataset(self, dataset=BOTH, debug=False):
        # dataset:
        #   NEWS_20   - 20newsgroup
        #   NEWS_AG   - ag news
        #   BOTH - combine both datasets

        if debug:
            top_rows = 1000
        else:
            top_rows = None

        # 20newsgroup
        if dataset == NEWS_20 or dataset == BOTH:
            data_test_20 = np.array(pd.read_csv('data/20newsgroup/20_news_test.csv', nrows=top_rows))[:, 1:]
            data_train_20 = np.array(pd.read_csv('data/20newsgroup/20_news_train.csv', nrows=top_rows))[:, 1:]
            print("pre-processing 20newsgroup dataset")
            data_test_20[:, 0] = self.preprocess(data_test_20)
            data_train_20[:, 0] = self.preprocess(data_train_20)
            print("done pre-processing")
            if dataset == NEWS_20:
                self.data_test = data_test_20
                self.data_train = data_train_20

        # ag news
        if dataset == NEWS_AG or dataset == BOTH:
            data_test_ag = pd.read_csv('data/ag_news/test.csv')
            data_train_ag = pd.read_csv('data/ag_news/train.csv')
            data_test_ag.drop(['title'], axis=1)
            data_train_ag.drop(['title'], axis=1)
            data_test_ag = np.array(data_test_ag.reindex(columns=['text', 'class']))
            data_train_ag = np.array(data_train_ag.reindex(columns=['text', 'class']))
            random_indices = np.random.choice(data_train_ag.shape[0], size=12000, replace=False)
            data_train_ag = data_train_ag[random_indices]
            for i in range(data_test_ag.shape[0]): data_test_ag[i][1] = str(data_test_ag[i][1])
            for i in range(data_train_ag.shape[0]): data_train_ag[i][1] = str(data_train_ag[i][1])
            print('pre-processing ag news dataset')
            data_test_ag[:, 0] = self.preprocess(data_test_ag)
            data_train_ag[:, 0] = self.preprocess(data_train_ag)
            print('done pre-processing')
            if dataset == NEWS_AG:
                self.data_test = data_test_ag
                self.data_train = data_train_ag

        # combine datasets
        if dataset == BOTH:
            self.data_test = np.concatenate((data_test_20, data_test_ag), axis=0)
            self.data_train = np.concatenate((data_train_20, data_train_ag), axis=0)

        self.classes = set(list(self.data_test[:, 1]))

        np.random.shuffle(self.data_test)
        np.random.shuffle(self.data_train)

        label_encoder = LabelEncoder()

        self.x_train = self.data_train[:, 0]
        y_train_words = self.data_train[:, 1]
        self.y_train = label_encoder.fit_transform(y_train_words)

        self.x_test = self.data_test[:, 0]
        self.original_x_test = self.data_test[:, 0]
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
        for row, words in enumerate(self.data_train[:, 0]):
            text += self.data_train[row][0].split(' ') + self.data_train[row][0].split(' ')
        if self.top_vocab_words:    # Limit vocab
            word_counts = Counter(text)
            most_common_words = [word for word, _ in word_counts.most_common(TOP_VOCAB_WORDS)]
            self.vocab = np.array(most_common_words)
        else:                       # No limit vocab
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
            self.x_train, self.x_test = self.generate_ngrams_feature()
        if self.feature == Features.TFIDF:
            print("Creating TFIDF Feature")
            self.x_train, self.x_test = self.generate_tfidf_feature()

    def generate_bow_feature(self):
        bow_vectorizer = CountVectorizer(vocabulary=self.vocab)
        x_train = bow_vectorizer.fit_transform(self.x_train)
        x_test = bow_vectorizer.fit_transform(self.x_test)
        return x_train.toarray(), x_test.toarray()

    def generate_ngrams_feature(self, n=2):
        ngrams_vectorizer = CountVectorizer(vocabulary=self.vocab, ngram_range=(n,n))
        x_train = ngrams_vectorizer.fit_transform(self.x_train)
        x_test = ngrams_vectorizer.fit_transform(self.x_test)
        return x_train.toarray(), x_test.toarray()

    def generate_tfidf_feature(self, max_feat=10000):
        tfidf_vectorizer = TfidfVectorizer(max_features=max_feat)
        x_train = tfidf_vectorizer.fit_transform(self.x_train)
        x_test = tfidf_vectorizer.fit_transform(self.x_test)
        return x_train.toarray(), x_test.toarray()