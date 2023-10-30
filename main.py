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
    # Step 1: Data Reader
    dr = DataReader(Features.BOW)
    dr.open_dataset(debug=True)
    dr.build_vocab()
    dr.build_feature_set()

    # Step 2: Model(s)
    KerasModel = KerasFCNNModel(dr.x_train, dr.y_train)
    model = KerasModel.learn()

    # Step 3: Evaluate
    evaluate = Evaluate(model, dr.x_test, dr.y_test)
    preds = evaluate.predict()
    evaluate.evaluate(preds)
    # TODO: Do writeup
