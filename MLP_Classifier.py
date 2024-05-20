# Basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import re

# ML libraries
from sklearn.metrics import (auc, classification_report,
                             f1_score, precision_recall_curve,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# NLP libraries
# Install spacy
# !pip install spacy
# Download the English language model
# !python -m spacy download en
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

# NLTK
# import nltk
# nltk.download('wordnet')
# nltk.download('stopwords')
# from nltk.tokenize import word_tokenize
# from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

# Progress bar
from tqdm import tqdm

#%% Constants
TARGET_NAMES = ['negative','positive']

# !pip install --upgrade tensorflow
# !nvidia-smi

#%% Read data
csv_file_path = "IMDB Dataset.csv"
# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Display the DataFrame
print(df.head())
df.shape
# df['review'][3]
sns.countplot(x="sentiment", data=df)

#%% Volume control
oldDF = df
# df = df.iloc[:2000,:]

#%% Preprocessing

TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    '''Removes HTML tags: replaces anything between opening and closing <> with empty space'''

    return TAG_RE.sub('', text)


def preprocess_text(sen):
    '''Cleans text data up, leaving only 2 or more char long non-stepwords composed of A-Z & a-z only
    in lowercase'''

    sentence = sen.lower()
    # Remove html tags
    sentence = remove_tags(sentence)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)  # When we remove apostrophe from the word "Mark's", the apostrophe is replaced by an empty space. Hence, we are left with single character "s" that we are removing here.
    # Remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)  # Next, we remove all the single characters and replace it by a space which creates multiple spaces in our text. Finally, we remove the multiple spaces from our text as well.
    # Remove Stopwords
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
    sentence = pattern.sub('', sentence)

    return sentence

# Calling preprocessing_text function on df
X = []
sentences = list(df['review'])
for sen in tqdm(sentences, desc='Preprocessing: '):
    X.append(preprocess_text(sen))

#%% Label conversion
# Converting sentiment labels to 0 & 1
y = df['sentiment']
y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))

#%% Train Test Split
# Split the data into training (70%) and temporary data (30%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# Split the temporary data into testing (50%) and validation (50%)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print('Train samples: {}'.format(len(X_train)))
print('Val samples: {}'.format(len(X_val)))
print('Test samples: {}'.format(len(X_test)))

#%% Tokenization
nlp = spacy.load('en_core_web_sm', disable=["tagger", "parser","ner"])
nlp.add_pipe('sentencizer')

def tokenize_samples(samples):

    tokenized_samples = []
    for i in tqdm(range(len(samples)), desc='Tokenize: '):
        doc = nlp(samples[i])  # Tokenize the sample into sentences
        tokens = []
        for sent in doc.sents:
            for tok in sent:  # Iterate through the words of the sentence
                if '\n' in tok.text or "\t" in tok.text or "--" in tok.text or "*" in tok.text or tok.text.lower() in STOP_WORDS:
                    continue
                if tok.text.strip():
                    tokens.append(tok.text.replace('"',"'").strip())
        tokenized_samples.append(tokens)

    return tokenized_samples

X_train_tokenized = tokenize_samples(X_train)
X_val_tokenized = tokenize_samples(X_val)
X_test_tokenized = tokenize_samples(X_test)

#%% Tokenization example
for item in X_train_tokenized[:2]:
     print(item, '\n')

#%% Vectorize

vectorizer = TfidfVectorizer(ngram_range=(1,2),
                             max_features = 5000,
                             sublinear_tf=True)

X_train_tfidf = vectorizer.fit_transform([" ".join(x) for x in X_train_tokenized])
X_val_tfidf   = vectorizer.transform([" ".join(x) for x in X_val_tokenized])
X_test_tfidf  = vectorizer.transform([" ".join(x) for x in X_test_tokenized])

#%% SVD
# Reduce dimensionality using svd 5000 --> 500
svd = TruncatedSVD(n_components=500, random_state=4321)
X_train_svd = svd.fit_transform(X_train_tfidf)
X_val_svd = svd.transform(X_val_tfidf)
X_test_svd = svd.transform(X_test_tfidf)

#%% Normalization -> bad
# from sklearn.preprocessing import StandardScaler

# # Assuming X is your data
# scaler = StandardScaler()
# X_train_svd = scaler.fit_transform(X_train_svd)
# X_val_svd = scaler.transform(X_val_svd)
# X_test_svd = scaler.transform(X_test_svd)

#%% Baseline: Majority

def my_auc_report(y_true,y_pred, title = 'UNK')->None:
    """
    Inputs the y_true and y_pred,
    Prints the PR-AUC for each class
    """
    print(f"=== PR-AUC for {title} ===")
    precision, recall, _ = precision_recall_curve(y_true, y_pred[:,0],pos_label=0)
    area = auc(recall, precision)
    print(f"PR-AUC for class 0: {area*100:.2f}%")
    precision, recall, _ = precision_recall_curve(y_true, y_pred[:,1])
    area = auc(recall, precision)
    print(f"PR-AUC for class 1: {area*100:.2f}%\n")

# Train
# The dummy classifier always predicts the 'most frequent' class
baseline = DummyClassifier(strategy='most_frequent')
start_time = time.time()
baseline.fit(X_train_svd, y_train)
print("Training took: {} seconds \n".format(time.time() - start_time))

# Evaluate
model = baseline
# Classification Reports
predictions = model.predict(X_train_svd)
print(classification_report(y_train, predictions, zero_division=0, target_names=TARGET_NAMES))
predictions = model.predict(X_val_svd)
print(classification_report(y_val, predictions, zero_division=0, target_names=TARGET_NAMES))
predictions = model.predict(X_test_svd)
print(classification_report(y_test, predictions, zero_division=0, target_names=TARGET_NAMES))

# Prepare the predictions for PR-AUC
predictions_train = model.predict_proba(X_train_svd)
predictions_val   = model.predict_proba(X_val_svd)
predictions_test  = model.predict_proba(X_test_svd)
# PR-AUC report
my_auc_report(y_train, y_pred=predictions_train, title='Train')
my_auc_report(y_val, y_pred=predictions_val, title='Validation')
my_auc_report(y_test, y_pred=predictions_test, title='Test')

#%% Baseline: Logistic
clf = LogisticRegression()
clf.fit(X_train_svd, y_train)
logit_val_accuracy = clf.score(X_val_svd, y_val)

# Evaluate
model = clf
# Classification Reports
predictions = model.predict(X_train_svd)
print(classification_report(y_train, predictions, target_names=TARGET_NAMES))
predictions = model.predict(X_val_svd)
print(classification_report(y_val, predictions, target_names=TARGET_NAMES))
predictions = model.predict(X_test_svd)
print(classification_report(y_test, predictions, target_names=TARGET_NAMES))

# Prepare the predictions for PR-AUC
predictions_train = model.predict_proba(X_train_svd)
predictions_val = model.predict_proba(X_val_svd)
predictions_test = model.predict_proba(X_test_svd)
# PR-AUC report
my_auc_report(y_train, y_pred=predictions_train, title='Train')
my_auc_report(y_val, y_pred=predictions_val, title='Validation')
my_auc_report(y_test, y_pred=predictions_test, title='Test')

#%% Metrics
class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = (self.model.predict(self.validation_data[0]) > 0.5).astype("int32")
        val_targ = self.validation_data[1]

        _val_f1 = f1_score(val_targ, val_predict, zero_division=1)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return

#%% MLP
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# # SEQ OLD
# model = Sequential()
# model.add(Dense(512, input_dim=X_train_svd.shape[1], activation='relu')) # Input_dim: Ta features tou SVD
# model.add(Dropout(0.5))
# model.add(Dense(256,  activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1, activation='sigmoid')) # To output layer prepei na exei tosous nevrones oses kai oi katigories mas # Softmax: logo Multiclass


# Sequential API --> After TUNER
model = Sequential()
model.add(Dense(384, input_dim=X_train_svd.shape[1]))
# model.add(BatchNormalization())  # Add batch normalization -> Bad
model.add(Activation('relu'))  # Moved activation to after batch normalization
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

print(model.summary()) # Tipwnei keimeno pou perigrafei pws settarisame to network

# Configures the model for training.
# Binary_crossentropy: Computes the crossentropy loss between the label and prediction.
# Metrics -> .compile | Callbacks -> .fit
model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"]
    )

if not os.path.exists('./checkpoints'):
  os.makedirs('./checkpoints')
# Callback to save the Keras model or model weights at some frequency.
checkpoint = ModelCheckpoint(
    'checkpoints/weights.hdf5',
    monitor='val_loss', # Save the model that scores the best @ validation accuracy
    mode='min', # Since accuracy -> maximization
    verbose=2,
    save_best_only=True,
    save_weights_only=True,
    )

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=10)

start_training_time = time.time()

history = model.fit(
    X_train_svd,
    y_train,
    validation_data=(X_val_svd, y_val),
    batch_size=128,
    epochs=100,
    shuffle=True, # Shuffle the train data on every epoch
    callbacks=[Metrics(valid_data=(X_val_svd, y_val)), checkpoint, early_stopping]
    )
end_training_time = time.time()

print(f'\nTraining time: {time.strftime("%H:%M:%S", time.gmtime(end_training_time - start_training_time))} \n')

#%% Curves
# history: Returned by the fit. Keeps the metrics for each epoch
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.axhline(y=logit_val_accuracy, color='red', linestyle='--', label='Logistic')
plt.legend(['train', 'dev', 'Logistic'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'dev'], loc='upper right')
plt.show()

#%% Evaluation
predictions = (model.predict(X_train_svd) >0.5)
print(classification_report(y_train, predictions, target_names=TARGET_NAMES))
predictions = (model.predict(X_val_svd) >0.5)
print(classification_report(y_val, predictions, target_names=TARGET_NAMES))
predictions = (model.predict(X_test_svd) >0.5)
print(classification_report(y_test, predictions, target_names=TARGET_NAMES))

#%% PR-AUC
def custom_pred_proba(X)->np.ndarray[tuple[float, float]]:
    ''' Calculates the classic .predict_proba() for the X input feature matix,
        because it is needed for the precision_recall_curve.
        Returns an array: [P(x=0), P(x=1)] '''
    pred = model.predict(X)
    return np.hstack((1-pred, pred))


# Prepare predictions for PR-AUC
predictions_train = custom_pred_proba(X_train_svd)
predictions_val = custom_pred_proba(X_val_svd)
predictions_test = custom_pred_proba(X_test_svd)
# PR-AUC report
my_auc_report(y_train, y_pred=predictions_train, title='Train')
my_auc_report(y_val, y_pred=predictions_val, title='Validation')
my_auc_report(y_test, y_pred=predictions_test, title='Test')


#%% Tuner
# # !pip install -U keras-tuner
# import keras_tuner as kt
# from tensorflow.keras.callbacks import EarlyStopping

# def build_model(hp):
#     model = Sequential()

#     layer_index = 0
#     for i in range(hp.Int(name='num_layers',min_value=1,max_value=3)):
#         if layer_index == 0:
#             model.add(Dense(hp.Int(name='hidden_units_'+str(i),min_value=128,max_value=512,step=64),
#                             activation=hp.Choice(name='activation_layer'+str(i),values=['relu','tanh']),
#                             input_dim=X_train_svd.shape[1]
#                            ))
#             model.add(Dropout(hp.Choice(name='dropout_layer_'+str(i),values=[0.1,0.2,0.3,0.4,0.5])))
#         else:
#             model.add(Dense(hp.Int(name='hidden_units_'+str(i),min_value=128,max_value=512,step=64),
#                             activation=hp.Choice(name='activation_layer'+str(i),values=['relu','tanh'])))
#             model.add(Dropout(hp.Choice(name='dropout_layer_'+str(i),values=[0.1,0.2,0.3,0.4,0.5])))

#         layer_index += 1

#     # Add last layer that produces the logits
#     model.add(Dense(1, activation='sigmoid'))

#     # Tune the learning rate for the optimizer
#     # Choose an optimal value from 0.01, 0.001, or 0.0001
#     hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 1e-4])
#     model.compile(loss='binary_crossentropy',
#                   optimizer=Adam(learning_rate=hp_learning_rate),
#                   metrics=['accuracy'])

#     return model

# tuner = kt.RandomSearch(build_model,
#                         objective='val_accuracy',
#                         max_trials=20,
#                         directory='KT_dir',
#                         project_name='KT_tuning')

# early_stopping = EarlyStopping(monitor='val_loss',
#                                patience=10)

# # tuner.search_space_summary()

# tuner.search(X_train_svd, y_train,
#              validation_data=(X_val_svd, y_val),
#              epochs=50,
#              batch_size = 128,
#              callbacks=[early_stopping, checkpoint])

# # Get best hyper-parameters setup
# tuner.get_best_hyperparameters()[0].values

# best_model = tuner.get_best_models(num_models=1)[0]
# best_model.summary()

# #%% Evaluate best
# predictions = (best_model.predict(X_val_svd) >0.5)
# print(classification_report(y_val, predictions, target_names=['negative','positive']))
