# ==============================================================================
# CSE 842
# Project: Article Text Categorization
#
# Authors: Yue Deng, Josh Erno, Christopher Nosowsky
#
# ==============================================================================
from sklearn.preprocessing import MinMaxScaler

from model import *
from evaluate import *
import tensorflow as tf

# ==== SETUP PARAMS HERE ====
DEBUG_MODE = False                      # DEBUG Mode limits dataset sizes for debug purposes
DATASET = BOTH                          # BOTH, NEWS_AG, or NEWS_20
FEATURE = Features.NGRAMS                 # BOW, NGRAMS, TFIDF, WORD2VEC or DOC2VEC
MODELS_TO_TRAIN = [KERAS_MODEL]         # Models to train
# === PREPROCESSING SPECIFIC FLAGS === #
TOP_VOCAB = True                        # Limit VOCAB size to top 15000 vocab only
STEM = True                             # Stem words
LEMMA = False                           # Lemmatize words
USE_GRID_SEARCH = False                  # Use GridSearchCV
# === MISC FLAGS === #
SAVE_TRAIN_TEST_TO_FILES = True      # Save preprocessed data or no?
# LOAD_TRAIN_TEST_TO_FILES = False      # Recommend this being True after first preprocessing
TEST_SIZE = DEFAULT_TEST_SIZE           # Either DEFAULT_TEST_SIZE or value between (0,1)
# ===========================
# TODO: Fix memoryerror on grid search -- NB
# TODO: Fix low accuracy on Doc2Vec + TFIDF features
# TODO: Fix Bi, Tri Grams

if __name__ == '__main__':
    # Check GPU is available
    print(tf.config.list_physical_devices('GPU'))
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    # Step 1: Data Reader
    dr = DataReader(FEATURE, dataset=DATASET, top_vocab_words=TOP_VOCAB)
    print("Begin preprocessing")
    dr.open_dataset(debug=DEBUG_MODE)
    dr.build_vocab()
    # if LOAD_TRAIN_TEST_TO_FILES:
    #     if dr.check_if_saved():             # If you want to load, then check if saved data exists
    #         dr.load_features_labels_vocab()
    #     else:
    #         print("All or some train/test data not saved. Please build features first. Exiting.")
    #         exit()
    # else:
    dr.build_feature_set()

    if SAVE_TRAIN_TEST_TO_FILES:
        print("Electing to save/overwrite data")
        dr.save_features_labels_vocab()

    # Step 2: Model(s)
    models = []
    scaler = MinMaxScaler()
    for model in MODELS_TO_TRAIN:
        if model == KERAS_MODEL:
            KerasModel = KerasFCNNModel(dr.x_train, dr.y_train, dataset=DATASET, use_grid_search=USE_GRID_SEARCH)
            kerasFcnnModel = KerasModel.learn()
            models.append(kerasFcnnModel)
            KerasModel.plot_train_accuracy()
            KerasModel.plot_training_loss()

        elif model == NAIVE_BAYES_MODEL:
            if FEATURE == Features.DOC2VEC or FEATURE == Features.WORD2VEC:
                print("Normalize x_train to avoid negative values error in NB")
                x_train = scaler.fit_transform(dr.x_train)
                NBModel = NaiveBayesModel(x_train, dr.y_train, dataset=DATASET, use_grid_search=USE_GRID_SEARCH)
            else:
                NBModel = NaiveBayesModel(dr.x_train, dr.y_train, dataset=DATASET, use_grid_search=USE_GRID_SEARCH)
            naiveBayesModel = NBModel.learn()
            models.append(naiveBayesModel)

        elif model == BERT_MODEL:
            BertModel = BERTModel(DATASET, dr.original_x_train, dr.y_train)
            bertModel = BertModel.use_pretrained_bert()
            models.append(bertModel)

    # Step 3: Evaluate
    for modelIndex, thisModel in enumerate(models):
        if MODELS_TO_TRAIN[modelIndex] == BERT_MODEL:
            evaluate = Evaluate(thisModel, dr.original_x_test.tolist(), dr.y_test)
        else:
            if MODELS_TO_TRAIN[modelIndex] == NAIVE_BAYES_MODEL \
                    and (FEATURE == Features.DOC2VEC or FEATURE == Features.WORD2VEC):
                print("Normalize x_test to avoid negative values error in NB")
                x_test = scaler.transform(dr.x_test)
                evaluate = Evaluate(thisModel, x_test, dr.y_test)
            else:
                evaluate = Evaluate(thisModel, dr.x_test, dr.y_test)

        preds = evaluate.predict(MODELS_TO_TRAIN[modelIndex])
        evaluate.evaluate(preds)
        print(evaluate.evaluate_classification_report(dr.label_encoder, preds))
        print('The accuracy of ' + MODELS_TO_TRAIN[modelIndex] + ' was: ', evaluate.accuracy)
        print('The precision of ' + MODELS_TO_TRAIN[modelIndex] + ' was: ', evaluate.precision)
        print('The recall of ' + MODELS_TO_TRAIN[modelIndex] + ' was: ', evaluate.recall)
        print('The f1 score of ' + MODELS_TO_TRAIN[modelIndex] + ' was: ', evaluate.f1)
