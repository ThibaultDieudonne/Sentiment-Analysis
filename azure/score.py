import warnings
warnings.simplefilter(action='ignore')
import json
import numpy as np
import os
import pickle
import joblib
import tensorflow as tf
import pandas as pd
import unidecode
import nltk

MAX_LEN = 25

PATH = "deploy"

def init():
    global model
    global preprocesser
    preprocesser = Preprocessing()
    model = tf.keras.models.load_model(f"{PATH}/glove_lstm_nn.h5")

def run(text_list):
    text_list = text_list.split("X")
    text_list = [*map(preprocesser.clean_up, text_list)]
    ds = preprocesser.encode(text_list)
    y_pred = model.predict(ds)
    y_pred = [yp if len(txt) else 'NaN' for yp, txt in zip(y_pred.tolist(), text_list)]
    return y_pred

class Preprocessing:
    def __init__(self):
        with open(f"{PATH}/vocabulary.txt", 'r') as f:
            self.vocab = f.read()
        self.vocab = self.vocab.split()
        self.vocab = set(self.vocab)
        with open(f"{PATH}/sw.txt", 'r') as f:
            self.stop_words = f.read()
        self.stop_words = self.stop_words.split()
        self.stop_words = set(self.stop_words)
        self.reg_exp_tokenizer = nltk.RegexpTokenizer(r'\w+')
        self.tokenizer = joblib.load(f"{PATH}/tokenizer.joblib")

    def clean_up(self, text):
        if not isinstance(text, str):
            return ""
        text = text.lower()
        for exclude in ['&quot;', '&amp;']:
            text = text.replace(exclude, ' ')
        text = text.replace('-', '')
        text = text.split()
        for exclude in ['@', '/', 'www']:
            text = [w for w in text if not exclude in w]
        text = ' '.join(text)
        text = unidecode.unidecode(text)
        tokens = self.reg_exp_tokenizer.tokenize(text)
        tokens = [w for w in tokens if not w in self.stop_words and w in self.vocab]
        if len(tokens) > MAX_LEN:
            tokens = tokens[:MAX_LEN]
        return ' '.join(tokens) if len(tokens) > 1 else ''

    def encode(self, data_list):
        encoded = self.tokenizer.texts_to_sequences(data_list)
        data_list = np.asarray(tf.keras.preprocessing.sequence.pad_sequences(encoded, maxlen=MAX_LEN, padding='post'))
        y_test = np.zeros(len(data_list))
        ds = tf.data.Dataset.from_tensor_slices((data_list, tf.cast(y_test, tf.int32)))
        ds = ds.batch(16)
        return ds

if __name__=="__main__":
    PATH = "."

