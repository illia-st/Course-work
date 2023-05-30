import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


max_seq_len = 50
model = tf.keras.models.load_model('model.h5')
index_to_label = None
tokenizer = None

with open('index_to_label.pkl', 'rb') as f:
    index_to_label = pickle.load(f)
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def predict(text):
    seq = tokenizer.texts_to_sequences([text])
    bytes_seq = pad_sequences(seq, truncating='post', padding='post', maxlen=max_seq_len)

    p = model.predict(np.expand_dims(bytes_seq[0], axis=0))[0]
    return index_to_label[np.argmax(p).astype('uint8')]



st.write("""
    Coursework\n
    Made by Illia Stetsenko
""")

st.sidebar.header('Enter the sentence to predict')

def user_input():
    return st.sidebar.text_input("Enter the sentence to predict")

input = user_input()

st.write(input)
predicition = predict(input)
st.subheader("Prediction")
if len(input) == 0:
    st.write('Enter the sentence to predict')
else:
    st.write(predicition)
