# ==============================================================================
# CSE 842
# Project: Article Text Categorization
#
# Authors: Yue Deng, Josh Erno, Christopher Nosowsky
#
# ==============================================================================
import torch
import random
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
from data_reader import *
from datasets import *
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, random_split

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class AbstractModel:
    def __init__(self, x_train, y_train, dataset):
        self.x_train = x_train
        self.y_train = y_train
        self.dataset = dataset
        if self.dataset == NEWS_20:
            self.num_classes = 20
        elif self.dataset == NEWS_AG:
            self.num_classes = 4
        else:
            self.num_classes = 24

    def fine_tune_model(self, model, params, k_fold_splits=5):
        grid_search = GridSearchCV(estimator=model, param_grid=params, cv=k_fold_splits, n_jobs=-1, verbose=2)
        grid_search.fit(self.x_train, self.y_train)
        print("Best Parameters: ", grid_search.best_params_)
        print("Best Accuracy: {:.2f}%".format(grid_search.best_score_ * 100))


class KerasFCNNModel(AbstractModel):
    def __init__(self, x_train, y_train, dataset=Datasets.BOTH, use_grid_search=False, features=None):
        super().__init__(x_train, y_train, dataset)
        self.history = None
        self.dataset = dataset
        self.use_grid_search = use_grid_search
        self.features = features

    def learn(self):
        in_shape = self.x_train.shape[1]
        model = tf.keras.Sequential()

        # Fully connected layers, relu to introduce non-linearity
        # Dropout added for regularization to prevent over fitting/reliance on specific nodes/learn more robust feats.
        # L2 penalty added to prevent overfitting and penalize large weights
        # Output layer applies softmax to convert raw output to probabilities,
        # each node representing probability of the input belonging to a particular class
        if self.features == Features.DOC2VEC:
            model.add(tf.keras.layers.Dense(in_shape, activation=tf.nn.relu, input_shape=(in_shape,)))
            model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
            model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
            model.add(tf.keras.layers.Dense(self.num_classes, activation=tf.nn.softmax))
        else:
            model.add(tf.keras.layers.Dense(in_shape, activation=tf.nn.relu, input_shape=(in_shape,)))
            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.1)))
            model.add(tf.keras.layers.Dropout(0.5))
            model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.1)))
            model.add(tf.keras.layers.Dense(self.num_classes, activation=tf.nn.softmax))

        # Adam adapts the learning rate of each parameter individually based on history of gradients
        opt = keras.optimizers.Adam()
        # Sparse categorical crossentropy allows us to send raw labels in instead of one hot encoded labels for binary
        # Also measures the difference between predicted prob. distribution and true probability distribution
        model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        if self.features == Features.DOC2VEC:
            self.history = model.fit(self.x_train, self.y_train,
                                     epochs=25, batch_size=32)
        else:
            self.history = model.fit(self.x_train, self.y_train,
                                     epochs=7, batch_size=32)

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
    def __init__(self, x_train, y_train, dataset=Datasets.BOTH, use_grid_search=False):
        super().__init__(x_train, y_train, dataset)
        self.use_grid_search = use_grid_search

    def learn(self):
        print('Training Naive Bayes model...')
        # Alpha = laplace smoothing parameter = Ensure no feature has probability of zero.
        model = MultinomialNB(alpha=0.1)

        if self.use_grid_search:
            params = {'alpha': [0.00001, 0.0001, 0.001, 0.1, 1, 10, 100, 1000]}
            # best in debug: 0.1
            self.fine_tune_model(model, params)

        model.fit(self.x_train, self.y_train)
        print('Naive Bayes model trained')

        return model


class LogisticRegressionModel(AbstractModel):
    def __init__(self, x_train, y_train, dataset):
        super().__init__(x_train, y_train, dataset)

    def learn(self):
        print('Training Logistic Regression model...')
        # C = Regularization strength. Smaller = more regularization/prevent over fitting
        model = LogisticRegression(C=5, multi_class='multinomial', solver='saga', max_iter=1000)
        model.fit(self.x_train, self.y_train)
        print('Logistic Regression model trained')
        return model


class BERTModel(AbstractModel):
    def __init__(self, dataset, x_train, y_train):
        super().__init__(x_train, y_train, dataset)
        self.num_classes = 0
        self.batch_size = 32
        self.num_epochs = 4
        self.train_split = 0.9
        self.model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.training_stats = []
        self.device = None

    def get_num_classes(self):
        if self.dataset == NEWS_20:
            self.num_classes = 20
        elif self.dataset == NEWS_AG:
            self.num_classes = 4
        else:
            self.num_classes = 24

    def check_cuda_available(self):
        if torch.cuda.is_available():
            # Tell PyTorch to use the GPU.
            self.device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        else:
            print('No GPU available, using the CPU instead.')
            self.device = torch.device("cpu")
        return self.device

    @staticmethod
    def set_seed():
        seed_val = 42
        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

    @staticmethod
    def flat_accuracy(preds, labels):
        """
        Calculate the accuracy of our predictions vs labels
        """
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def use_pretrained_bert(self):
        """
            Fine-tune a pre-trained BERT model for sequence classification on the provided training data.
        Note:
        - The function loads a pre-trained BERT model for sequence classification
        with a specified number of output classes.
        - The model is moved to the GPU if available.
        - The training data is tokenized using the provided tokenizer,
        ensuring padding, truncation, and a maximum sequence length of 128.
        - The input tokens are converted into PyTorch tensors.
        - The dataset is created using input IDs, attention mask, and labels,
        and then split into training and validation sets.
        - Data loaders are created for both training and validation sets, allowing batch processing during training.

        :return: The fine-tuned BERT model
        """
        device = self.check_cuda_available()
        print('Training BERT model...')

        # Load pre-trained BERT model for sequence classification
        model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=self.num_classes)

        # Move the model to GPU if available
        model.cuda()
        model.to(device)

        # Tokenize the training data using the specified tokenizer
        # Padding=true ensures all sequences have same length, adding padding to shorter sequences
        # Truncation means limit the length of sequences to specified max length
        # Return_tensors=pt means that PyTorch tensors to be returned
        # Max_length=128 means all inputs will only have tokenized sequence of size 128 at max
        inputs = self.tokenizer(self.x_train.tolist(),
                                padding=True, truncation=True,
                                return_tensors="pt",
                                max_length=128)

        # Convert labels to PyTorch tensor and move to GPU
        labels = torch.tensor(self.y_train, dtype=torch.int64).to(device)

        # Move input tokens and attention mask to GPU
        # Input ID's are numbers mapped to each word in tokenized sequence
        # Attention mask is either 0 or 1, 0 meaning ignore token, 1 meaning attend to that token.
        # O I assume is MASK token
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        print("Input IDs size:", input_ids.size())
        print("Attention Mask size:", attention_mask.size())
        print("Labels size:", labels.size())

        # Create a PyTorch TensorDataset from input IDs, attention mask, and labels
        dataset = TensorDataset(input_ids, attention_mask, labels)

        train_size = int(self.train_split * len(dataset))
        val_size = len(dataset) - train_size

        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        print('{:>5,} training samples'.format(train_size))
        print('{:>5,} validation samples'.format(val_size))

        data_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        validation_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        # Setup optimizer and learning rate scheduler
        # Add eps?
        optimizer = AdamW(
            model.parameters(),
            lr=2e-5
        )

        # Total number of training steps is [number of batches] x [number of epochs]
        total_steps = len(data_loader) * self.num_epochs

        # Create the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0,  # Default value in run_glue.py
                                                    num_training_steps=total_steps)

        print("Length of the data_loader steps: " + str(len(data_loader)))

        for epoch in range(self.num_epochs):
            model.train()
            # Reset the total loss each step to re-calc avg for each epoch
            total_loss = 0
            for batch in data_loader:
                optimizer.zero_grad()
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                model.zero_grad()
                outputs = model(input_ids=input_ids,
                                token_type_ids=None,
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss
                # Perform a backward pass to calculate the gradients
                loss.backward()
                # Clip the norm of the gradients to 1.0.
                # Avoid exploding gradient problem
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Network gets told to update the parameters
                optimizer.step()
                # Update the learning rate
                scheduler.step()
                total_loss += loss.item()
            # Calculate the average loss over all the batches
            average_loss = total_loss / len(data_loader)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Training Loss: {average_loss:.4f}")

            print("")
            print("Running Validation...")

            # Put the model in evaluation mode
            model.eval()

            total_eval_accuracy = 0
            total_eval_loss = 0

            # Evaluate data for one epoch
            for batch in validation_dataloader:
                input_ids = batch[0].to(device)
                input_mask = batch[1].to(device)
                labels = batch[2].to(device)

                # Tell pytorch not to bother with constructing the compute graph during
                # the forward pass, since this is only needed for backprop (training).
                with torch.no_grad():
                    # Forward pass, calculate logit predictions.
                    outputs = model(input_ids,
                                    token_type_ids=None,
                                    attention_mask=input_mask,
                                    labels=labels)

                # Accumulate the validation loss
                loss = outputs.loss
                logits = outputs.logits

                total_eval_loss += loss.item()

                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to('cpu').numpy()

                # Calculate the accuracy for this batch of test sentences, and
                # accumulate it over all batches
                total_eval_accuracy += self.flat_accuracy(logits, label_ids)

            # Report the final accuracy for this validation run
            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            print("  Accuracy: {0:.2f}".format(avg_val_accuracy))

            # Calculate the average loss over all the batches
            avg_val_loss = total_eval_loss / len(validation_dataloader)
            print("  Validation Loss: {0:.2f}".format(avg_val_loss))

            self.training_stats.append(
                {
                    'Epoch': epoch + 1,
                    'Training Loss': average_loss,
                    'Validation Loss': avg_val_loss,
                    'Validation Accuracy': avg_val_accuracy,
                }
            )

        print('BERT model trained')

        return model

    def table_training_stats(self):
        # Display floats with two decimal places
        pd.set_option('display.float_format', '{:.2f}'.format)

        # Create a DataFrame from our training statistics
        df_stats = pd.DataFrame(data=self.training_stats)

        # Use the 'epoch' as the row index.
        df_stats = df_stats.set_index('Epoch')

        # Display the table
        print(df_stats)

        return df_stats

    @staticmethod
    def plot_training_validation_loss(df_stats):
        # Use plot styling from seaborn.
        sns.set(style='darkgrid')

        # Increase the plot size and font size.
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (12, 6)

        # Plot the learning curve.
        plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
        plt.plot(df_stats['Validation Loss'], 'g-o', label="Validation")

        # Label the plot.
        plt.title("Training & Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.xticks([1, 2, 3, 4])

        plt.show()
