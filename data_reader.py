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

# nltk.download('punkt')
# nltk.download('stopwords')

class DataReader:

    def __init__(self):
        self.vocab = None
        self.data_train = None
        self.data_test = None
        self.classes = None

    def open_dataset(self):
        # # code used to combine data from folders and save
        # test_data = []
        # train_data = []
        # # iterate through test and train folders
        # for i, data_folder in enumerate(os.listdir('data/20newsgroup')):
        #     # iterate through each label folder
        #     for j, label_folder in enumerate(os.listdir(os.path.join('data/20newsgroup', data_folder))):
        #         # iterate through each file
        #         for k, file in enumerate(os.listdir(os.path.join('data/20newsgroup', data_folder, label_folder))):
        #             # read in file
        #             f = os.path.join('data/20newsgroup', data_folder, label_folder, file)
        #             f = open(f, 'r')
        #             f = f.read()
        #             # append to train/test list
        #             test_data.append([f, [label_folder]]) if i == 0 else train_data.append([f, label_folder])
        #             x = 10
        #         print('done', label_folder)
        # test_df = pd.DataFrame(test_data)
        # train_df = pd.DataFrame(train_data)
        # test_df.to_csv('data/20newsgroup/20_news_test.csv')
        # train_df.to_csv('data/20newsgroup/20_news_train.csv')

        self.data_test = np.array(pd.read_csv('data/20newsgroup/20_news_test.csv'))[:, 1:]
        self.data_train = np.array(pd.read_csv('data/20newsgroup/20_news_train.csv'))[:, 1:]

        self.data_test[:, 0] = self.preprocess(self.data_test)
        self.data_train[:, 0] = self.preprocess(self.data_train)
        # np.savetxt('data/20newsgroup/20_news_pre_processed_test.csv', self.data_test)
        # np.savetxt('data/20newsgroup/20_news_pre_processed_train.csv', self.data_train)

        # test_df = pd.DataFrame(self.data_test)
        # train_df = pd.DataFrame(self.data_train)
        # test_df.to_csv('data/20newsgroup/20_news_pre_processed_test.csv')
        # train_df.to_csv('data/20newsgroup/20_news_pre_processed_train.csv')

        # self.data_test = np.array(pd.read_csv('data/20newsgroup/20_news_pre_processed_test.csv'))[:, 1:]
        # self.data_train = np.array(pd.read_csv('data/20newsgroup/20_news_pre_processed_train.csv'))[:, 1:]
        self.classes = set(list(self.data_test[:,1]))

        np.random.shuffle(self.data_test)
        np.random.shuffle(self.data_train)



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


    def bow_feature_generation(self):
        pass
