import json
import numpy as np
import os
import pickle
import joblib
import tensorflow as tf
import pandas as pd
import unidecode
import nltk
from io import StringIO

MAX_LEN = 86

def init():
    global model
    global preprocesser
    preprocesser = Preprocessing()
    model = tf.keras.models.load_model("deploy/glove_cnn.h5")

def run(raw_data):
    raw_data = StringIO(raw_data)
    df = pd.read_csv(raw_data, encoding='latin-1')
    df["text"] = df["text"].apply(preprocesser.clean_up)
    ds = preprocesser.encode(df["text"].tolist())
    y_pred = model.predict(ds)
    return y_pred.tolist()

class Preprocessing:
    def __init__(self):
        with open("deploy/vanilla_vocab_25000.txt", 'r') as f:
            self.vocab = f.read()
        self.vocab = self.vocab.split()
        self.vocab = set(self.vocab)
        self.reg_exp_tokenizer = nltk.RegexpTokenizer(r'\w+')
        self.tokenizer = joblib.load(f"deploy/tokenizer.joblib")

    def clean_up(self, text):
        if isinstance(text, float):
            return ""
        text = unidecode.unidecode(''.join(c for c in text if not c.isdigit()).replace('\n', '').lower())
        tokens = self.reg_exp_tokenizer.tokenize(text)
        tokens = [w for w in tokens if w in self.vocab and len(w) > 1]
        return " ".join(tokens)

    def encode(self, data_list):
        encoded = self.tokenizer.texts_to_sequences(data_list)
        data_list = np.asarray(tf.keras.preprocessing.sequence.pad_sequences(encoded, maxlen=MAX_LEN, padding='post'))
        y_test = np.zeros(len(data_list))
        ds = tf.data.Dataset.from_tensor_slices((data_list, tf.cast(y_test, tf.int32)))
        ds = ds.batch(10)
        return ds
