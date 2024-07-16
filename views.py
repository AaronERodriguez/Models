from flask import Blueprint, render_template, request

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_length = 5018
trunc_type='post'

with open('label_encoder.pickle', 'rb') as label:
    label_encoder = pickle.load(label)
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = tf.keras.models.load_model('./tmp/cyber.keras')

views = Blueprint(__name__, "views")

@views.route("/")
def home():
    return render_template("index.html", hide='hide-container')

@views.route('/predict', methods=['POST', 'GET'])
def makePrediction():
        padded_text = pad_sequences(tokenizer.texts_to_sequences([request.form['text']]), maxlen=max_length, truncating=trunc_type)
        prediction = model.predict(padded_text, verbose=0)
        label = ''
        for item in prediction:
            label = label_encoder.inverse_transform([np.argmax(item)])[0]
            listOfPercentages = []
            index=0
            for ele in item:
                listOfPercentages.append(str(label_encoder.inverse_transform([index])[0])+' — '+str(round(ele*100,4))+' %')
                index+=1
        return render_template('index.html', hide='predict-container', pred=label, age=listOfPercentages[0], ethnicity=listOfPercentages[1], gender=listOfPercentages[2], not_cyberbullying=listOfPercentages[3], other_cyberbullying=listOfPercentages[4], religion=listOfPercentages[5])