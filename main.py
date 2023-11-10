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


# TODO: Tensorflow-GPU?

if __name__ == '__main__':
    # set the dataset to either: NEWS_20 = 20newsgroup, NEWS_AG = ag news, BOTH = combine both datasets
    dataset = NEWS_20
    # Step 1: Data Reader
    dr = DataReader(Features.BOW, top_vocab_words=True)
    dr.open_dataset(dataset=dataset, debug=True)
    dr.build_vocab()
    dr.build_feature_set()

    # Step 2: Model(s)
    modelNames = ['bertModel']
    models = []
    for thisModelName in modelNames:
        if thisModelName == 'kerasFcnnModel':
            KerasModel = KerasFCNNModel(dr.x_train, dr.y_train)
            kerasFcnnModel = KerasModel.learn()
            models.append(kerasFcnnModel)

        elif thisModelName == 'naiveBayesModel':
            NaiveBayesModel = NaiveBayesModel(dr.x_train, dr.y_train)
            naiveBayesModel = NaiveBayesModel.learn()
            models.append(naiveBayesModel)

        elif thisModelName == 'rippleModel':
            RippleModel = RIPPERModel(dr.x_train, dr.y_train)
            rippleModel = RippleModel.learn()
            models.append(rippleModel)

        elif thisModelName == 'bertModel':
            BertModel = BERTModel(dataset)
            bertModel = BertModel.usePretrainedBert()
            models.append(bertModel)

    # Step 3: Evaluate
    modelIndex = 0
    for thisModel in models:
        if modelNames[modelIndex] == 'bertModel':
            evaluate = Evaluate(thisModel, dr.original_x_test.tolist(), dr.y_test)
        else:
            evaluate = Evaluate(thisModel, dr.x_test, dr.y_test)

        preds = evaluate.predict(modelNames[modelIndex])

        evaluate.evaluate(preds)
        print('The accuracy of ' + modelNames[modelIndex] + 'was: ', evaluate.accuracy)
        print('The precision of ' + modelNames[modelIndex] + 'was: ', evaluate.precision)
        print('The recall of ' + modelNames[modelIndex] + 'was: ', evaluate.recall)
        print('The f1 score of ' + modelNames[modelIndex] + 'was: ', evaluate.f1)

        modelIndex += 1
