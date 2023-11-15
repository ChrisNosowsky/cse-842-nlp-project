# ==============================================================================
# CSE 842
# Project: Article Text Categorization
#
# Authors: Yue Deng, Josh Erno, Christopher Nosowsky
#
# ==============================================================================
from model import *
from evaluate import *
import tensorflow as tf

# ==== SETUP PARAMS HERE ====
DEBUG_MODE = False                      # DEBUG Mode limits dataset sizes for debug purposes
DATASET = BOTH                          # BOTH, NEWS_AG, or 20_NEWS
FEATURE = Features.BOW               # BOW, NGRAMS, TFIDF, DOC2VEC, or WORD2VEC
MODELS_TO_TRAIN = [KERAS_MODEL]         # Models to train
TOP_VOCAB = True                        # Limit VOCAB size to top 15000 vocab only
STEM = True                             # Stem words
LEMMA = False                           # Lemmatize words
USE_GRID_SEARCH = True                  # Use GridSearchCV
TEST_SIZE = DEFAULT_TEST_SIZE           # Either DEFAULT_TEST_SIZE or value between (0,1)
# ===========================


if __name__ == '__main__':
    # Check GPU is available
    print(tf.config.list_physical_devices('GPU'))
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    # Step 1: Data Reader
    dr = DataReader(FEATURE, top_vocab_words=TOP_VOCAB)
    dr.open_dataset(dataset=DATASET, debug=DEBUG_MODE)
    dr.build_vocab()
    dr.build_feature_set()

    # Step 2: Model(s)
    models = []
    for model in MODELS_TO_TRAIN:
        if model == KERAS_MODEL:
            KerasModel = KerasFCNNModel(dr.x_train, dr.y_train, dataset=DATASET, use_grid_search=USE_GRID_SEARCH)
            kerasFcnnModel = KerasModel.learn()
            models.append(kerasFcnnModel)
            KerasModel.plot_train_accuracy()
            KerasModel.plot_training_loss()

        elif model == NAIVE_BAYES_MODEL:
            NBModel = NaiveBayesModel(dr.x_train, dr.y_train, use_grid_search=USE_GRID_SEARCH)
            naiveBayesModel = NBModel.learn()
            models.append(naiveBayesModel)

        elif model == RIPPER_MODEL:
            RippleModel = RIPPERModel(dr.x_train, dr.y_train)
            rippleModel = RippleModel.learn()
            models.append(rippleModel)

        elif model == BERT_MODEL:
            BertModel = BERTModel(DATASET, dr.original_x_train, dr.y_test)
            bertModel = BertModel.use_pretrained_bert()
            models.append(bertModel)

    # Step 3: Evaluate
    for modelIndex, thisModel in enumerate(models):
        if MODELS_TO_TRAIN[modelIndex] == BERT_MODEL:
            evaluate = Evaluate(thisModel, dr.original_x_test.tolist(), dr.y_test)
        else:
            evaluate = Evaluate(thisModel, dr.x_test, dr.y_test)

        preds = evaluate.predict(MODELS_TO_TRAIN[modelIndex])
        evaluate.evaluate(preds)
        print(evaluate.evaluate_classification_report(dr.label_encoder, preds))
        print('The accuracy of ' + MODELS_TO_TRAIN[modelIndex] + ' was: ', evaluate.accuracy)
        print('The precision of ' + MODELS_TO_TRAIN[modelIndex] + ' was: ', evaluate.precision)
        print('The recall of ' + MODELS_TO_TRAIN[modelIndex] + ' was: ', evaluate.recall)
        print('The f1 score of ' + MODELS_TO_TRAIN[modelIndex] + ' was: ', evaluate.f1)
