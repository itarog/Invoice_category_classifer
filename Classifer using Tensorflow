import pandas as pd
import numpy as np

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('stopwords')

import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.models import Sequential
from keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout

from enum import Enum

class WordNormalization(Enum):
    CHAR_ONLY_SPACE_REGEXP = r'[^a-zA-Z\s]'
    HTML_MARKUPS_SPACE_REGEXP = r'(<.*?>)'
    NON_ASCII_AND_DIGITS_SPACE_REGEXP = r'(\\W|\\d)'
    MIN_WORD_LENGTH = 3
    MAX_WORD_LENGTH = 16
    STOP_WORDS =  list(nltk.corpus.stopwords.words('english')) + list(nltk.corpus.stopwords.words('german'))

# Plot metrics function
mpl.rcParams['figure.figsize'] = (20, 16)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_metrics(history):
    metrics =  ['loss', 'AUC', 'Precision', 'Recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])
        plt.legend()

# Load data and inital wrangling
csv_path = '/content/dataset.csv'

full_df = pd.read_csv(csv_path)
full_df = full_df.dropna(subset=['ocr_text'])

target_column = full_df.category
variable_column = full_df.ocr_text

# Preprocess text
def scrub_text(text):
    text = re.sub(WordNormalization.CHAR_ONLY_SPACE_REGEXP.value,' ', text)
    text = re.sub(WordNormalization.HTML_MARKUPS_SPACE_REGEXP.value,' ',text)
    text = re.sub(WordNormalization.NON_ASCII_AND_DIGITS_SPACE_REGEXP.value,' ',text)
    text = text.lower()
    text = text.strip()
    return text

def stem_text(text):
    porter_stemmer = PorterStemmer()
    text = porter_stemmer.stem(text)
    return text

def preprocess_text(text):
    text = scrub_text(text)
    text = stem_text(text)
    wpt = nltk.WordPunctTokenizer()
    tokens = wpt.tokenize(text)
    filtered_tokens = [token for token in tokens if (token not in WordNormalization.STOP_WORDS.value) and (len(token)>WordNormalization.MIN_WORD_LENGTH.value) and (len(token)<WordNormalization.MAX_WORD_LENGTH.value) and (token.isalpha())]
    return ' '.join(filtered_tokens)

variable_column = variable_column.apply(preprocess_text)

# Split train test data
X_train, X_test, y_train, y_test = train_test_split(variable_column, target_column,
                                                    test_size=0.2,
                                                    random_state=40)

# X data tokenizer
tokenizer_obj = Tokenizer()
total_items = variable_column

tokenizer_obj.fit_on_texts(total_items)

max_length = max([len(string.split()) for string in total_items])

vocab_size = (len(tokenizer_obj.word_index) + 1) // 2

X_train_tokens = tokenizer_obj.texts_to_sequences(X_train)
X_test_tokens = tokenizer_obj.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_tokens, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_tokens, maxlen=max_length, padding='post')

# y data encoder
label_encoder = OneHotEncoder()
label_encoder.fit(np.array(target_column).reshape(-1,1))
all_labels = label_encoder.transform(np.array(target_column).reshape(-1,1))
train_labels = label_encoder.transform(np.array(y_train).reshape(-1,1))
test_labels = label_encoder.transform(np.array(y_test).reshape(-1,1))

num_labels = len(set(target_column))

# NN metrics
METRICS = [
           tf.keras.metrics.Precision(name='Precision'),
           tf.keras.metrics.Recall(name='Recall'),
           tf.keras.metrics.AUC(name='AUC'),
           ]

# NN model (unopt)
EMBEDDING_DIM = 300
DENSE_LAYER_ACTIVATION = 'relu'
FIRST_DENSE_LAYER_QUANT = 10*num_labels
FIRST_DROPOUT_LAYER_RATE = 0.5
SECOND_DENSE_LAYER_QUANT = num_labels
OUTPUT_LAYER_ACTIVATION = 'softmax'
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'categorical_crossentropy'


unopt_model = Sequential(
    [
     Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length),
     GlobalAveragePooling1D(),
     Dense(FIRST_DENSE_LAYER_QUANT, activation=DENSE_LAYER_ACTIVATION),
     Dropout(FIRST_DROPOUT_LAYER_RATE),
     Dense(SECOND_DENSE_LAYER_QUANT, activation=DENSE_LAYER_ACTIVATION),
     Dense(num_labels, activation=OUTPUT_LAYER_ACTIVATION),
    ]
)
# unopt_model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))
# unopt_model.add(GlobalAveragePooling1D())
# unopt_model.add(Dense(FIRST_DENSE_LAYER_QUANT, activation=DENSE_LAYER_ACTIVATION))
# unopt_model.add(Dropout(FIRST_DROPOUT_LAYER_RATE))
# unopt_model.add(Dense(SECOND_DENSE_LAYER_QUANT, activation=DENSE_LAYER_ACTIVATION))
# unopt_model.add(Dense(num_labels, activation=OUTPUT_LAYER_ACTIVATION))

unopt_model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)

unopt_model.summary()

# Training NN (unopt)
num_epochs = 500
batch_size = 64
unopt_history = unopt_model.fit(X_train_pad, train_labels,
                                batch_size = batch_size,
                                epochs=num_epochs,
                                verbose=2,
                                class_weight='balanced',
                                validation_split=0.2)

# Test results (unopt)
loss, precision, recall, auc = unopt_model.evaluate(X_test_pad, test_labels)
print(f'test results are:\n loss: {loss}\n precision: {precision}\n recall: {recall}\n AUC: {auc}')

# Metrics plot (unopt)
plot_metrics(unopt_history)

# NN model (opt)
EMBEDDING_DIM = 300
DENSE_LAYER_ACTIVATION = 'relu'
FIRST_DENSE_LAYER_QUANT = 10*num_labels
FIRST_DROPOUT_LAYER_RATE = 0.5
SECOND_DENSE_LAYER_QUANT = num_labels
OUTPUT_LAYER_ACTIVATION = 'softmax'
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'categorical_crossentropy'

opt_model = Sequential()
opt_model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))
opt_model.add(GlobalAveragePooling1D())
opt_model.add(Dense(FIRST_DENSE_LAYER_QUANT, activation=DENSE_LAYER_ACTIVATION))
opt_model.add(Dropout(FIRST_DROPOUT_LAYER_RATE))
opt_model.add(Dense(SECOND_DENSE_LAYER_QUANT, activation=DENSE_LAYER_ACTIVATION))
opt_model.add(Dense(num_labels, activation=OUTPUT_LAYER_ACTIVATION))

opt_model.compile(loss=LOSS_FUNCTION, optimizer=OPTIMIZER, metrics=METRICS)

opt_model.summary()

# training NN (opt)
num_epochs = 200
batch_size = 64
opt_history = opt_model.fit(X_train_pad, train_labels,
                            batch_size = batch_size,
                            epochs=num_epochs,
                            verbose=2,
                            class_weight='balanced',
                            validation_split=0.2)

# Test results (opt)
loss, precision, recall, auc = opt_model.evaluate(X_test_pad, test_labels)
print(f'test results are:\n loss: {loss}\n precision: {precision}\n recall: {recall}\n AUC: {auc}')

# Metrics plot (opt)
plot_metrics(opt_history)
