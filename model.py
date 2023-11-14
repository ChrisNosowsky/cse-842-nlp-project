# ==============================================================================
# CSE 842
# Project: Article Text Categorization
#
# Authors: Yue Deng, Josh Erno, Christopher Nosowsky
#
# ==============================================================================
import torch
import tensorflow as tf
import wittgenstein as lw
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from tensorflow import keras
from data_reader import *
from datasets import *
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset, random_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

################################################
# MODELING TO DO LIST AS PROJECT DEVELOPS
# TODO: Tweak model parameters -- All team
# TODO: Add Ray Tune or Hyperopt tuning parameters -- Yue
# TODO: LM? (BERT pretrained models, maybe two? -- One for Yue TODO, One for Chris TODO
# TODO: Try again at adding GridSearch with Keras? (optional)
# Chris Notes:
# Below is originally for Doc2Vec feature.
#         # clf = LinearSVC(C=0.0025)
#         # clf.fit(train_vectors, self.y_train)
################################################


class AbstractModel:
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def fine_tune_model(self, model, params, k_fold_splits=5):
        grid_search = GridSearchCV(estimator=model, param_grid=params, cv=k_fold_splits, n_jobs=-1, verbose=2)
        grid_search.fit(self.x_train, self.y_train)
        print("Best Parameters: ", grid_search.best_params_)
        print("Best Accuracy: {:.2f}%".format(grid_search.best_score_ * 100))


class KerasFCNNModel(AbstractModel):
    def __init__(self, x_train, y_train, dataset=Datasets.BOTH, use_grid_search=False):
        super().__init__(x_train, y_train)
        self.history = None
        self.dataset = dataset
        self.use_grid_search = use_grid_search

    def learn(self):
        in_shape = self.x_train.shape[1]
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(in_shape, activation=tf.nn.relu, input_shape=(in_shape,)))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.1)))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.1)))
        if self.dataset == NEWS_20:
            num_classes = 20
        elif self.dataset == NEWS_AG:
            num_classes = 4
        else:
            num_classes = 24
        model.add(tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax))
        opt = keras.optimizers.Adam()
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # if self.use_grid_search:
        #     params = {
        #         'optimizer__lr': [0.0001, 0.001, 0.05, 0.1],
        #         'model__dropout': [0, 0.5],
        #         'epochs': [5, 10, 20],
        #         'batch_size': [16, 32, 64]
        #     }
        #     model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        #     model_wrapped = KerasClassifier(model=model, loss="sparse_categorical_crossentropy", optimizer=opt,
        #                                     metrics=['accuracy'], verbose=0)
        #     self.fine_tune_model(model_wrapped, params)
        # else:
        #     model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        self.history = model.fit(self.x_train, self.y_train,
                                 epochs=7, batch_size=64)
        model.summary()
        train_acc = self.history.history['accuracy'][-1]
        train_loss = self.history.history['loss'][-1]
        print('Keras FCNN Training Accuracy: ' + str(train_acc))
        print('Keras FCNN Training Loss: ' + str(train_loss))
        return model

    def plot_train_accuracy(self):
        # Extract training accuracy values and epochs from the History object
        train_acc = self.history.history['accuracy']
        epochs = np.arange(1, len(train_acc) + 1)

        # Plot training accuracy
        plt.plot(epochs, train_acc, label='Training Accuracy', marker='o')
        plt.title('Training Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_training_loss(self):
        # Extract training loss values and epochs from the History object
        train_loss = self.history.history['loss']
        epochs = np.arange(1, len(train_loss) + 1)

        # Plot training loss
        plt.plot(epochs, train_loss, label='Training Loss', marker='o')
        plt.title('Training Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()


class NaiveBayesModel(AbstractModel):
    def __init__(self, x_train, y_train, use_grid_search=False):
        super().__init__(x_train, y_train)
        self.use_grid_search = use_grid_search

    def learn(self):
        print('Training Naive Bayes model...')
        model = MultinomialNB()

        if self.use_grid_search:
            params = {'alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000]}
            # best in debug: 0.1
            self.fine_tune_model(model, params)

        model.fit(self.x_train, self.y_train)
        print('Naive Bayes model trained')

        return model


class RIPPERModel(AbstractModel):
    def __init__(self, x_train, y_train):
        super().__init__(x_train, y_train)

    def learn(self):
        print('Training Ripple model...')
        model = lw.RIPPER()
        model.fit(self.x_train, self.y_train, class_feat='Poisonous/Edible', pos_class='p')
        print('Ripple model trained')

        return model


class BERTModel(AbstractModel):
    def __init__(self, dataset, x_train, y_train):
        super().__init__(x_train, y_train)
        if dataset == NEWS_20:
            self.numClasses = 20
        elif dataset == NEWS_AG:
            self.numClasses = 4
        else:
            self.numClasses = 20 + 4

    def use_pretrained_bert(self):
        print('Training BERT model...')
        model_name = 'bert-base-uncased'
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=self.numClasses)

        inputs = tokenizer(self.x_train.tolist(), padding=True, truncation=True, return_tensors="pt", max_length=128)
        labels = torch.tensor(self.y_train, dtype=torch.int64)

        dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)

        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        batch_size = 2
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = AdamW(model.parameters(), lr=1e-5)

        num_epochs = 10
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            average_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")

        print('BERT model trained')

        return model
