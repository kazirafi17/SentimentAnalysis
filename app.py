import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizer from .pkl files
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Sentiment labels
sentiment_labels = {0: "Positive Sentiment", 1: "Negative Sentiment", 2: "Neutral Sentiment"}

# Function to preprocess and predict sentiment
def predict_sentiment(text):
    # Preprocess the text
    text_seq = tokenizer.texts_to_sequences([text])
    text_padded = pad_sequences(text_seq, padding='post', maxlen=30)  # Use maxlen based on your model's training
    # Predict sentiment
    predictions = model.predict(text_padded)
    predicted_class_index = predictions.argmax(axis=-1)
    # Return the predicted sentiment label
    return sentiment_labels[predicted_class_index[0]]

# Streamlit app UI
st.title('Deep RNN Sentiment Analysis App')
st.write('Enter text to analyze its sentiment.')

# Text input
user_input = st.text_area('Enter your text here:')

# Button to make prediction
if st.button('Analyze Sentiment'):
    if user_input.strip() != "":
        sentiment = predict_sentiment(user_input)
        st.write(f"Predicted Sentiment: **{sentiment}**")
    else:
        st.write("Please enter some text.")

# Optional: Display some example texts
st.sidebar.title('Example Texts')
if st.sidebar.button('Load Example 1'):
    st.sidebar.write('The price of products are everage.')
if st.sidebar.button('Load Example 2'):
    st.sidebar.write('I was disappointed with the service, not coming back.')

st.markdown("""
    <style>
        /* Apply background color to the whole page */
        .reportview-container {
            background: #87CEEB;  /* Sky Blue Background */
            padding-top: 0px;
        }

        .sidebar .sidebar-content {
            background: #87CEEB;  /* Sky Blue Background */
        }

        /* Main content area */
        .main {
            background-color: #87CEEB;  /* Sky Blue Background */
            padding: 0;
            margin: 0;
        }

        /* Header adjustments */
        header, .stApp {
            background-color: #87CEEB;
            color: black;
            padding: 10px 0;
        }

        /* Footer styling */
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            padding: 10px;
            background-color: #87CEEB;  /* Sky Blue Background */
            color: #000;
        }
        .footer a {
            color: #000;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
    </style>
    <div class="footer">
        &copy; <a href="https://www.linkedin.com/in/abdul-mukit-1bbb72218" target="_blank">Abdul Mukit</a>
    </div>
""", unsafe_allow_html=True)
