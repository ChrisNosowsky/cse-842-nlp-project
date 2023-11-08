# ==============================================================================
# CSE 842
# Project: Article Text Categorization
#
# Authors: Yue Deng, Josh Erno, Christopher Nosowsky
#
# ==============================================================================
from data_reader import *
from model import *
from evaluate import *

if __name__ == '__main__':
    # set the dataset to either: NEWS_20 = 20newsgroup, NEWS_AG = ag news, BOTH = combine both datasets
    dataset = NEWS_20
    # Step 1: Data Reader
    dr = DataReader(Features.TFIDF, top_vocab_words=True)
    dr.open_dataset(dataset=dataset, debug=False)
    dr.build_vocab()
    dr.build_feature_set()

    # Step 2: Model(s)
    modelNames = ['kerasFcnnModel']

    # KerasModel
    KerasModel = KerasFCNNModel(dr.x_train, dr.y_train)
    kerasFcnnModel = KerasModel.learn()

    # NaiveBayesModel
    # NaiveBayesModel = NaiveBayesModel(dr.x_train, dr.y_train)
    # naiveBayesModel = NaiveBayesModel.learn()

    # RippleModel
    # RippleModel = RIPPERModel(dr.x_train, dr.y_train)
    # rippleModel = RippleModel.learn()

    # Step 3: Evaluate
    modelIndex = 0
    for model in [kerasFcnnModel]:
        evaluate = Evaluate(model, dr.x_test, dr.y_test)
        if modelNames[modelIndex] == 'kerasFcnnModel':
            preds = np.argmax(evaluate.predict(), axis=1)
        else:
            preds = evaluate.predict()

        evaluate.evaluate(preds)
        print('The accuracy of ' + modelNames[modelIndex] + 'was: ', evaluate.accuracy)
        print('The precision of ' + modelNames[modelIndex] + 'was: ', evaluate.precision)
        print('The recall of ' + modelNames[modelIndex] + 'was: ', evaluate.recall)
        print('The f1 score of ' + modelNames[modelIndex] + 'was: ', evaluate.f1)
        modelIndex += 1
