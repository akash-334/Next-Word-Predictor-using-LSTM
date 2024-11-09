import numpy as np
import pickle
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model and tokenizer
model = load_model("lstm_model.h5")
print("Model loaded successfully!")  # Debug line

with open("tokenizer.pkl", "rb") as file:
    tokenizer = pickle.load(file)
print("Tokenizer loaded successfully!")  # Debug line

def predict_next_words(text, num_words=5):
    print("Prediction function called!")  # Debug line
    for _ in range(num_words):
        tok = tokenizer.texts_to_sequences([text])[0]
        pad = pad_sequences([tok], maxlen=93, padding='pre')
        pred = model.predict(pad)
        pos = np.argmax(pred, axis=-1)[0]

        for word, index in tokenizer.word_index.items():
            if index == pos:
                text += " " + word
                break
    return text

st.title("Text Prediction using LSTM")
text = st.text_input("Enter text to continue prediction:")

if st.button("Predict Next Words"):
    result = predict_next_words(text)
    st.write("Predicted Text:", result)

if st.button("About"):
    st.text("Text Predictor")
    st.text("Built with Streamlit and TensorFlow LSTM model")

if __name__ == '__main__':
    pass
