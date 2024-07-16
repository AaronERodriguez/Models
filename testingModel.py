import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = None

max_length = 5018
trunc_type='post'

with open('label_encoder.pickle', 'rb') as label:
    label_encoder = pickle.load(label)
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = tf.keras.models.load_model('./tmp/cyber.keras')

def makePrediction(txt):
        padded_text = pad_sequences(tokenizer.texts_to_sequences([txt]), maxlen=max_length, truncating=trunc_type)
        prediction = model.predict(padded_text, verbose=0)
        for item in prediction:
            print(f'The following text is: {label_encoder.inverse_transform([np.argmax(item)])[0]}')
            index=0
            for ele in item:
                print(label_encoder.inverse_transform([index])+' — '+str(round(ele*100,4))+' %')
                index+=1

userInput = input("Which text would you like to classify as cyberbullying: ")         

makePrediction(userInput)
