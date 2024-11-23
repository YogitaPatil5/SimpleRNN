# Import Libraries and load the model

import numpy as np
import tensorflow as tf 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

# load IMDB dataset word index
word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}

# Load the model
model = load_model('imdb_model_simplernn.h5')

# helper function
def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i-3, '?') for i in encoded_review])  

# Function to preprocess user input
def preprocess_text(text):
    
    # Convert text to lowercase    
    words = text.lower().split()
    
    # Remove stop words
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    
    # Pad the review
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500, padding='post', truncating='post')
    return padded_review

import streamlit as st 

st.title('IMDB Movie Review Sentiment Analysis')

st.write('Enter a movie review below to classify it as positive or negative:')

# User input
user_input = st.text_input('Movie Review')

if st.button('Classify'):

    # Preprocess user input
    preprocessed_input = preprocess_text(user_input)

    # Make prediction
    prediction = model.predict(preprocessed_input)

    # Display prediction result
    
    st.write('Prediction:', 'Positive' if prediction[0][0] > 0.5 else 'Negative')
    st.write(f'Prediction Score: {prediction[0][0]}')

else:
    st.write('Please enter a movie review to classify.')