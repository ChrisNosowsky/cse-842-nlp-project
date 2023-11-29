# ==============================================================================
# CSE 842
# Project: Article Text Categorization
#
# Authors: Yue Deng, Josh Erno, Christopher Nosowsky
#
# ==============================================================================
from model import *
from evaluate import *
from sklearn.utils import check_random_state
import tensorflow as tf

# ==== SETUP PARAMS HERE ====
DEBUG_MODE = False                          # DEBUG Mode limits dataset sizes for debug purposes
DATASET = NEWS_20                           # BOTH, NEWS_AG, or NEWS_20
FEATURE = Features.NGRAMS                      # BOW, NGRAMS, WORD2VEC or DOC2VEC
MODELS_TO_TRAIN = [NAIVE_BAYES_MODEL]           # KERAS_MODEL, NAIVE_BAYES_MODEL, or BERT_MODEL, LOG_REG_MODEL as options
# === PREPROCESSING SPECIFIC FLAGS === #
TOP_VOCAB = True                            # Limit VOCAB size to top 15000 vocab only
STEM = True                                 # Stem words
LEMMA = False                               # Lemma words
USE_GRID_SEARCH = False                     # Use GridSearchCV
# === MISC FLAGS === #
SAVE_TRAIN_TEST_TO_FILES = False            # Save preprocessed data or no?
TEST_SIZE = DEFAULT_TEST_SIZE               # Either DEFAULT_TEST_SIZE or value between (0,1)
SEED_NUM = 42
# ===========================


def set_project_seed():
    np.random.seed(SEED_NUM)
    tf.random.set_seed(SEED_NUM)
    check_random_state(SEED_NUM)
    random.seed(SEED_NUM)


if __name__ == '__main__':
    # Set project seed
    set_project_seed()
    # Check GPU is available
    print(tf.config.list_physical_devices('GPU'))
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    # # Step 1: Data Reader
    dr = DataReader(FEATURE, dataset=DATASET, top_vocab_words=TOP_VOCAB)
    print("Begin preprocessing")
    dr.open_dataset(debug=DEBUG_MODE)
    dr.build_vocab()
    dr.build_feature_set()

    if SAVE_TRAIN_TEST_TO_FILES and not DEBUG_MODE:
        print("Electing to save/overwrite data")
        dr.save_features_labels_vocab()

    # Step 2: Model(s)
    models = []
    for model in MODELS_TO_TRAIN:
        if model == KERAS_MODEL:
            KerasModel = KerasFCNNModel(dr.x_train, dr.y_train,
                                        dataset=DATASET,
                                        use_grid_search=USE_GRID_SEARCH,
                                        features=FEATURE)
            kerasFcnnModel = KerasModel.learn()
            models.append(kerasFcnnModel)
            # KerasModel.plot_train_accuracy()
            # KerasModel.plot_training_loss()
        elif model == NAIVE_BAYES_MODEL:
            if FEATURE == Features.DOC2VEC or FEATURE == Features.WORD2VEC:
                print("Cannot train Naive Bayes with Word2Vec or Doc2Vec. Please choose another feature.")
                exit()
            else:
                NBModel = NaiveBayesModel(dr.x_train, dr.y_train, dataset=DATASET, use_grid_search=USE_GRID_SEARCH)
            naiveBayesModel = NBModel.learn()
            models.append(naiveBayesModel)
        elif model == LOG_REG_MODEL:
            LRModel = LogisticRegressionModel(dr.x_train, dr.y_train, dataset=DATASET)
            logRegModel = LRModel.learn()
            models.append(logRegModel)
        elif model == BERT_MODEL:
            BertModel = BERTModel(DATASET, dr.original_x_train, dr.y_train)
            BertModel.set_seed()
            BertModel.get_num_classes()
            bertModel = BertModel.use_pretrained_bert()
            models.append(bertModel)
            # df_stats = BertModel.table_training_stats()
            # BertModel.plot_training_validation_loss(df_stats)

    # Step 3: Evaluate
    for modelIndex, thisModel in enumerate(models):
        true_labels = None
        if MODELS_TO_TRAIN[modelIndex] == BERT_MODEL:
            evaluate = Evaluate(thisModel, dr.original_x_test.tolist(), dr.y_test, thisModel.device)
            predictions, true_labels = evaluate.predict(MODELS_TO_TRAIN[modelIndex])
            print(evaluate.evaluate_classification_report(MODELS_TO_TRAIN[modelIndex], dr.label_encoder,
                                                          predictions, true_labels))
        else:
            evaluate = Evaluate(thisModel, dr.x_test, dr.y_test)
            predictions = evaluate.predict(MODELS_TO_TRAIN[modelIndex])
            evaluate.evaluate(predictions)
            print(evaluate.evaluate_classification_report(MODELS_TO_TRAIN[modelIndex], dr.label_encoder, predictions))
            print('The accuracy of ' + MODELS_TO_TRAIN[modelIndex] + ' was: ', evaluate.accuracy)
            print('The precision of ' + MODELS_TO_TRAIN[modelIndex] + ' was: ', evaluate.precision)
            print('The recall of ' + MODELS_TO_TRAIN[modelIndex] + ' was: ', evaluate.recall)
            print('The f1 score of ' + MODELS_TO_TRAIN[modelIndex] + ' was: ', evaluate.f1)
