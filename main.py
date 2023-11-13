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

#  TODO CHRIS: TEST NEW FEATURE (DOC2VEC)
#  TODO CHRIS: TEST GRID SEARCH ON NAIVE BAYES + KERAS
#  TODO CHRIS: TEST STEMMER
#  TODO CHRIS: TEST KERAS GRAPHS (TRAIN ACC + LOSS METHODS)

# ==== SETUP PARAMS HERE ====
DEBUG_MODE = True
DATASET = NEWS_20
FEATURE = Features.BOW
MODELS_TO_TRAIN = [KERAS_MODEL]
TOP_VOCAB = True
STEM = False
LEMMA = False
USE_GRID_SEARCH = False
TEST_SIZE = DEFAULT_TEST_SIZE # either DEFAULT_TEST_SIZE or value between (0,1)
# ===========================


if __name__ == '__main__':
    # Check GPU is available
    print(tf.config.list_physical_devices('GPU'))

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
        print('The accuracy of ' + MODELS_TO_TRAIN[modelIndex] + ' was: ', evaluate.accuracy)
        print('The precision of ' + MODELS_TO_TRAIN[modelIndex] + ' was: ', evaluate.precision)
        print('The recall of ' + MODELS_TO_TRAIN[modelIndex] + ' was: ', evaluate.recall)
        print('The f1 score of ' + MODELS_TO_TRAIN[modelIndex] + ' was: ', evaluate.f1)
