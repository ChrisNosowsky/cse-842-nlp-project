# ==============================================================================
# CSE 842
# Project: Article Text Categorization
#
# Authors: Yue Deng, Josh Erno, Christopher Nosowsky
#
# ==============================================================================
import os
import re
import string
import numpy as np
import pandas as pd
import nltk
import multiprocessing
from random import shuffle
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
from nltk import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from constants import *
from features import *
from collections import Counter
from gensim.models import Word2Vec

################################################
# PREPROCESSING TO DO LIST AS PROJECT DEVELOPS
# TODO: Save the DM and DBOW embeddings to speed up runtime after first save
# TODO: Figure how to load GoogleNews pretrained embeddings for Doc2Vec feature
################################################


class DataReader:

    def __init__(self, feature=BOW, top_vocab_words=False, stem=False, lemma=False, test_size=DEFAULT_TEST_SIZE):
        self.vocab = None
        self.data_train = None
        self.data_test = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        self.classes = None
        self.class_mapping = None
        self.original_x_test = None
        self.original_x_train = None
        self.label_encoder = LabelEncoder()
        self.top_vocab_words = top_vocab_words
        self.stem = stem
        self.lemma = lemma
        self.feature = feature
        self.cores = multiprocessing.cpu_count()
        self.nltk_download_check()
        self.test_size = test_size

    @staticmethod
    def nltk_download_check():
        installed_corpora = os.listdir(nltk.data.find("corpora"))
        installed_tokenizers = os.listdir(nltk.data.find("tokenizers"))
        installed_nltk = installed_corpora + installed_tokenizers
        print(installed_nltk)
        print("Checking if NLTK corpora requirements are installed..")
        for corpus in NLTK_CORPUS:
            corpus_zip = corpus + ".zip"
            if corpus not in installed_nltk and corpus_zip not in installed_nltk:
                print("The '" + corpus + "' corpus is not downloaded. Downloading...")
                nltk.download(corpus)
                print("Downloaded the '" + corpus + "' corpus")
            else:
                print("The '" + corpus + "' corpus is downloaded.")

    @staticmethod
    def lemmatization(text):
        """
        reduces word to base form
        :param text: String of the document content
        :return: String of lemmatized text
        """
        wn_lemmatizer = WordNetLemmatizer()
        text = ' '.join([wn_lemmatizer.lemmatize(word) for word in text.split()])
        return text

    @staticmethod
    def stem_text(text):
        """
        Stems the text (eaten becomes eat for example)
        :param text: String of the document content
        :return: String of the stemmed text
        """
        ps = PorterStemmer()
        text = ' '.join([ps.stem(word) for word in text.split()])
        return text

    @staticmethod
    def remove_non_letters(text):
        """
        Removes all non-letters using RegEx
        :param text: String of the document content
        :return: String of alphabetic text only
        """
        return re.sub("[^a-zA-Z]", " ", text)

    @staticmethod
    def convert_file_dir_to_csv():
        # code used to combine data from folders and save
        test_data = []
        train_data = []
        # iterate through test and train folders
        for i, data_folder in enumerate(os.listdir('data/20newsgroup')):
            if 'csv' not in data_folder:
                # iterate through each label folder
                for j, label_folder in enumerate(os.listdir(os.path.join('data/20newsgroup', data_folder))):
                    # iterate through each file
                    for k, file in enumerate(os.listdir(os.path.join('data/20newsgroup', data_folder, label_folder))):
                        # read in file
                        f = os.path.join('data/20newsgroup', data_folder, label_folder, file)
                        f = open(f, 'r')
                        f = f.read()
                        # append to train/test list
                        test_data.append([f, label_folder]) if i == 0 else train_data.append([f, label_folder])
                        x = 10
                    print('done', label_folder)
        test_df = pd.DataFrame(test_data)
        train_df = pd.DataFrame(train_data)
        test_df.to_csv(NEWS_20_PATH_TEST)
        train_df.to_csv(NEWS_20_PATH_TRAIN)

    def open_dataset(self, dataset=BOTH, debug=False):
        """
        Opens the dataset of choosing.
        Options:
            NEWS_20   - 20newsgroup
            NEWS_AG   - ag news
            BOTH - combine both datasets
        :param dataset: Enum value of the dataset(s) of choosing
        :param debug: Boolean value of whether to run in debug mode
        """
        data_test_20 = None
        data_train_20 = None
        data_test_ag = None
        data_train_ag = None

        if debug:
            top_rows = 1000
        else:
            top_rows = None

        # 20newsgroup
        if dataset == NEWS_20 or dataset == BOTH:
            data_train_20 = np.array(pd.read_csv(NEWS_20_PATH_TRAIN, nrows=top_rows))[:, 1:]
            data_test_20 = np.array(pd.read_csv(NEWS_20_PATH_TEST, nrows=top_rows))[:, 1:]

            print("pre-processing 20newsgroup dataset")
            data_test_20[:, 0] = self.preprocess(data_test_20)
            data_train_20[:, 0] = self.preprocess(data_train_20)
            print("done pre-processing")

            if dataset == NEWS_20:
                if self.test_size != DEFAULT_TEST_SIZE:
                    data_combine_20 = np.concatenate((data_train_20, data_test_20), axis=0)
                    X_train, X_test, y_train, y_test = train_test_split(data_combine_20[:, 0], data_combine_20[:, 1],
                                                                        test_size=self.test_size, random_state=42)
                    data_train_20 = np.concatenate((X_train.reshape(-1, 1), y_train.reshape(-1, 1)), axis=1)
                    data_test_20 = np.concatenate((X_test.reshape(-1, 1), y_test.reshape(-1, 1)), axis=1)
                self.data_test = data_test_20
                self.data_train = data_train_20

        # ag news
        if dataset == NEWS_AG or dataset == BOTH:
            data_train_ag = pd.read_csv(AG_NEWS_PATH_TRAIN)
            data_test_ag = pd.read_csv(AG_NEWS_PATH_TEST)

            data_train_ag.drop(['title'], axis=1)
            data_test_ag.drop(['title'], axis=1)

            data_train_ag['topic'] = data_train_ag['topic'].map(AG_NEWS_CLASS_MAPPING)
            data_test_ag['topic'] = data_test_ag['topic'].map(AG_NEWS_CLASS_MAPPING)

            data_train_ag = np.array(data_train_ag.reindex(columns=['text', 'topic']))
            data_test_ag = np.array(data_test_ag.reindex(columns=['text', 'topic']))

            if top_rows is not None:
                data_test_ag = data_test_ag[:top_rows]
                data_train_ag = data_train_ag[:top_rows]
            else:
                random_indices = np.random.choice(data_train_ag.shape[0], size=12000, replace=False)
                data_train_ag = data_train_ag[random_indices]
            for i in range(data_test_ag.shape[0]): data_test_ag[i][1] = str(data_test_ag[i][1])
            for i in range(data_train_ag.shape[0]): data_train_ag[i][1] = str(data_train_ag[i][1])
            print('pre-processing ag news dataset')
            data_test_ag[:, 0] = self.preprocess(data_test_ag)
            data_train_ag[:, 0] = self.preprocess(data_train_ag)
            print('done pre-processing')
            if dataset == NEWS_AG:
                if self.test_size != DEFAULT_TEST_SIZE:
                    data_combine_ag = np.concatenate((data_train_ag, data_test_ag), axis=0)
                    X_train, X_test, y_train, y_test = train_test_split(data_combine_ag[:, 0], data_combine_ag[:, 1],
                                                                        test_size=self.test_size, random_state=42)
                    data_train_ag = np.concatenate((X_train.reshape(-1, 1), y_train.reshape(-1, 1)), axis=1)
                    data_test_ag = np.concatenate((X_test.reshape(-1, 1), y_test.reshape(-1, 1)), axis=1)
                self.data_test = data_test_ag
                self.data_train = data_train_ag

        # combine datasets
        if dataset == BOTH:
            data_test_both = np.concatenate((data_test_20, data_test_ag), axis=0)
            data_train_both = np.concatenate((data_train_20, data_train_ag), axis=0)
            if self.test_size != DEFAULT_TEST_SIZE:
                data_combine_both = np.concatenate((data_train_both, data_test_both), axis=0)
                X_train, X_test, y_train, y_test = train_test_split(data_combine_both[:, 0], data_combine_both[:, 1],
                                                                    test_size=self.test_size, random_state=42)
                data_train_both = np.concatenate((X_train.reshape(-1, 1), y_train.reshape(-1, 1)), axis=1)
                data_test_both = np.concatenate((X_test.reshape(-1, 1), y_test.reshape(-1, 1)), axis=1)

            self.data_test = data_test_both
            self.data_train = data_train_both

        self.classes = set(list(self.data_test[:, 1]))

        # shuffle data
        np.random.shuffle(self.data_test)
        np.random.shuffle(self.data_train)

        self.x_train = self.data_train[:, 0]                        # Select column 0 (tokenized articles) in matrix
        self.original_x_train = self.data_train[:, 0]
        y_train_words = self.data_train[:, 1]                       # Select column 1 (target labels) in matrix
        self.y_train = self.label_encoder.fit_transform(y_train_words)   # Convert labels to encoded numeric format

        self.x_test = self.data_test[:, 0]
        self.original_x_test = self.data_test[:, 0]
        y_test_words = self.data_test[:, 1]
        self.y_test = self.label_encoder.fit_transform(y_test_words)

        self.class_mapping = dict(zip(self.y_train, y_train_words))

    def preprocess(self, data):
        processed_data = []
        for i in range(data.shape[0]):
            # break text into list of words
            tokens = nltk.word_tokenize(data[i][0])

            stopwords = nltk.corpus.stopwords.words("english")
            filtered_tokens = []
            for token in tokens:
                # make text lowercase
                token = token.lower()
                # remove punctuation
                if token in string.punctuation:
                    continue
                token = self.remove_non_letters(token)
                # remove stopwords
                if token in stopwords:
                    continue
                # lemmatization (optional)
                if self.lemma:
                    token = self.lemmatization(token)
                # remove stems (optional)
                if self.stem:
                    token = self.stem_text(token)
                # remove whitespace and drop empty elements
                token = token.strip()
                filtered_tokens.append(token)

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
        if self.feature == Features.DOC2VEC:
            print("Creating Doc2Vec Feature")
            all_newsgroup_documents = []
            train_docs = self.convert_newsgroup_to_tagged_docs(self.x_train, 'train')
            test_docs = self.convert_newsgroup_to_tagged_docs(self.x_test, 'test')
            all_newsgroup_documents.extend(train_docs)
            all_newsgroup_documents.extend(test_docs)
            doc_list = all_newsgroup_documents[:]
            print('%d docs: %d train, %d test' % (len(doc_list), len(train_docs), len(test_docs)))
            print("Train Label Size: " + str(len(self.y_train)))
            self.x_train, self.x_test = self.generate_doc2vec_feature(all_newsgroup_documents,
                                                                      doc_list, train_docs, test_docs)
        if self.feature == Features.WORD2VEC:
            print("Creating Word2Vec Feature")
            self.x_train, self.x_test = self.generate_word2vec_features()

    @staticmethod
    def convert_newsgroup_to_tagged_docs(docs, split):
        # global doc_count
        tagged_documents = []

        for i, v in enumerate(docs):
            label = '%s_%s' % (split, i)
            tagged_documents.append(TaggedDocument(v, [label]))

        return tagged_documents

    @staticmethod
    def extract_vectors(model, docs):
        vectors_list = []
        for doc_no in range(len(docs)):
            doc_label = docs[doc_no].tags[0]
            doc_vector = model.dv[doc_label]
            vectors_list.append(doc_vector)
        return vectors_list

    @staticmethod
    def get_infer_vectors(model, docs):
        vecs = []
        for doc in docs:
            vecs.append(model.infer_vector(doc.words))
        return vecs

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

    def generate_doc2vec_feature(self, all_newsgroup_documents, doc_list, train_docs, test_docs, window_size=5):
        dbow_model = Doc2Vec(dm=0, dm_concat=1, sample=1e-5, window=5, negative=5, hs=0, min_count=2,
                             workers=self.cores)
        dm_model = Doc2Vec(dm=1, dm_mean=1, sample=1e-5, window=10, negative=5, hs=0, min_count=2,
                           workers=self.cores)

        # bow_model.load(self.load_pretrained_word_embeddings())
        dbow_model.build_vocab(
            all_newsgroup_documents)

        # dm_model.load(self.load_pretrained_word_embeddings())
        dm_model.build_vocab(
            all_newsgroup_documents)

        dbow_dmm_model = ConcatenatedDoc2Vec([dbow_model, dm_model])
        alpha, min_alpha, passes = (0.025, 0.001, 100)
        shuffle(doc_list)
        dbow_model.alpha, dbow_model.min_alpha = alpha, alpha
        dbow_model.train(doc_list, total_examples=len(doc_list), epochs=passes)
        dm_model.alpha, dm_model.min_alpha = alpha, alpha
        dm_model.train(doc_list, total_examples=len(doc_list), epochs=passes)

        dbow_dmm_model.alpha, dbow_dmm_model.min_alpha = alpha, alpha
        dbow_dmm_model.train(doc_list, total_examples=len(doc_list), epochs=passes)

        train_vectors = self.extract_vectors(dbow_dmm_model, train_docs)
        test_vectors = self.extract_vectors(dbow_dmm_model, test_docs)

        return np.array(train_vectors), np.array(test_vectors)

    # def load_pretrained_word_embeddings(self):
    #     f_in = gzip.open('GoogleNews-vectors-negative300.bin.gz', 'rb')
    #     f_out = open('GoogleNews-vectors-negative300.bin', 'wb')
    #     f_out.writelines(f_in)
    #
    #     model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin')
    #     model.save('GoogleNews-vectors-negative300.bin')
    #     return model

    def generate_word2vec_features(self):
        tokenized_senteces = [sentence.split() for sentence in self.x_train]
        word2vec_model = Word2Vec(sentences=tokenized_senteces, vector_size=300, window=30, min_count=30, workers=4)

        def sentence_to_vector(sentence):
            words = sentence.split()
            vectors = [word2vec_model.wv[word] for word in words if word in word2vec_model.wv]
            if not vectors:
                return np.zeros(word2vec_model.vector_size)
            return np.mean(vectors, axis=0)

        x_train_w2v = np.array([sentence_to_vector(sentence) for sentence in self.x_train])
        x_test_w2v = np.array([sentence_to_vector(sentence) for sentence in self.x_test])
        return x_train_w2v, x_test_w2v
