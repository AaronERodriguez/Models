import tensorflow as tf
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import json
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('cyberbullying_tweets.csv')
df = df.sample(frac=1).reset_index(drop=True)

maxLen = 0 
for item in df['tweet_text']:
  if maxLen<len(item):
       maxLen=len(item)
print(maxLen)

# integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(df['cyberbullying_type'])
print(integer_encoded.shape)
#binary encode
onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
print(onehot_encoded)
# invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[2, :])])
print(inverted)

train_dataset, test_dataset = df[:38154],df[38154:]

vocab_size = 10000
embedding_dim = 16
max_length = maxLen
trunc_type='post'
oov_tok = '<OOV>'
training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_dataset['tweet_text'])
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(train_dataset['tweet_text'])
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

print(padded.shape)

train_integer_encoded = label_encoder.fit_transform(train_dataset['cyberbullying_type'])
train_integer_encoded = train_integer_encoded.reshape(len(train_integer_encoded), 1)
train_onehot_encoder = OneHotEncoder(sparse_output=False)
train_onehot_encoded = train_onehot_encoder.fit_transform(train_integer_encoded)


testing_sequences = tokenizer.texts_to_sequences(test_dataset['tweet_text'])
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)

test_integer_encoded = label_encoder.fit_transform(test_dataset['cyberbullying_type'])
test_integer_encoded = test_integer_encoded.reshape(len(test_integer_encoded), 1)
test_onehot_encoder = OneHotEncoder(sparse_output=False)
test_onehot_encoded = test_onehot_encoder.fit_transform(test_integer_encoded)

vocab = {}
for word, index in word_index.items():
  if index <= vocab_size:
    vocab[word] = index

with open('tokenizer.pickle', 'wb') as handle:
  pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

model = tf.keras.Sequential()

model.add(tf.keras.layers.Embedding(vocab_size, 13, input_shape=(max_length,)))
model.add(tf.keras.layers.Conv1D(128, 5, activation='relu'))
model.add(tf.keras.layers.GlobalAveragePooling1D())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(6, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

NUM_EPOCHS = 50
BATCH_SIZE = 64

history = model.fit(padded, train_onehot_encoded, validation_data=(testing_padded, test_onehot_encoded), epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=2)


text = 'Can you please help me?'
padded_text = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=max_length, truncating=trunc_type)

prediction = model.predict(padded_text, verbose=0)
for item in prediction:
  index=0
  for ele in item:
    print(label_encoder.inverse_transform([index])+' — '+str(round(ele*100,4))+' %')
    index+=1

model.save('cyber.keras')