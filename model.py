# ==============================================================================
# CSE 842
# Project: Article Text Categorization
#
# Authors: Yue Deng, Josh Erno, Christopher Nosowsky
#
# ==============================================================================
import os
import tensorflow as tf
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

################################################
# MODELING TO DO LIST AS PROJECT DEVELOPS
# TODO: Tweak model parameters
# TODO: Use XGBoost?
################################################


class KerasFCNNModel:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.history = None

    def learn(self):
        in_shape = self.x_train.shape[1]
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(in_shape, activation=tf.nn.relu, input_shape=(in_shape,)))
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
        model.add(tf.keras.layers.Dense(20, activation=tf.nn.softmax))
        # learning_rate=0.001
        opt = keras.optimizers.Adam()
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.history = model.fit(self.x_train, self.y_train,
                                 epochs=7, batch_size=64)
        model.summary()
        train_acc = self.history.history['accuracy'][-1]
        train_loss = self.history.history['loss'][-1]
        print('Keras FCNN Training Accuracy: ' + str(train_acc))
        print('Keras FCNN Training Loss: ' + str(train_loss))
        return model

    def plot_train_accuracy(self):
        # TODO: TBD Later stage of project
        pass

    def plot_loss_accuracy(self):
        # TODO: TBD Later stage of project
        pass


class NaiveBayesModel:
    # TODO: TBD Later stage of project
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def learn(self):
        pass


class RIPPERModel:
    # TODO: TBD Later stage of project. Will use n-grams
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def learn(self):
        pass
