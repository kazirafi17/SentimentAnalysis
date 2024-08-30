import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model and tokenizer from .pkl files
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

try:
    with open('tokenizer.pkl', 'rb') as handle:
        tokenizer = pickle.load(handle)
except Exception as e:
    st.error(f"Error loading tokenizer: {e}")
    tokenizer = None

# Sentiment labels
sentiment_labels = {0: "Positive Sentiment", 1: "Negative Sentiment", 2: "Neutral Sentiment"}

# Function to preprocess and predict sentiment
def predict_sentiment(text):
    if tokenizer is None or model is None:
        st.error("Model or tokenizer is not loaded correctly.")
        return None
    # Preprocess the text
    text_seq = tokenizer.texts_to_sequences([text])
    text_padded = pad_sequences(text_seq, padding='post', maxlen=30)  # Use maxlen based on your model's training
    # Predict sentiment
    predictions = model.predict(text_padded)
    predicted_class_index = predictions.argmax(axis=-1)
    # Return the predicted sentiment label
    return sentiment_labels.get(predicted_class_index[0], "Unknown Sentiment")

# Initialize session state for user_input
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

# Streamlit app UI
st.title('Deep RNN Sentiment Analysis App')
st.write('Enter text to analyze its sentiment.')

# Sidebar for example texts
st.sidebar.title('Example Texts')
example_1 = "The movie was outstanding, a must-watch!"
example_2 = "I was disappointed with the service, not coming back."
example_3 = "Absolutely loved the new restaurant! Great food and atmosphere."
example_4 = "I feel like the product did not meet my expectations. Very dissatisfied."
example_5 = "I need to update my address on the website.."

if st.sidebar.button('Load Example 1'):
    st.session_state.user_input = example_1

if st.sidebar.button('Load Example 2'):
    st.session_state.user_input = example_2

if st.sidebar.button('Load Example 3'):
    st.session_state.user_input = example_3

if st.sidebar.button('Load Example 4'):
    st.session_state.user_input = example_4

if st.sidebar.button('Load Example 5'):
    st.session_state.user_input = example_5

# Text input
user_input = st.text_area('Enter your text here:', value=st.session_state.user_input, height=200)

# Button to make prediction
if st.button('Analyze Sentiment'):
    if user_input.strip() != "":
        sentiment = predict_sentiment(user_input)
        if sentiment:
            st.write(f"Predicted Sentiment: **{sentiment}**")
        else:
            st.write("Error in prediction.")
    else:
        st.write("Please enter some text.")

# Custom CSS
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

        /* Button styling */
        .stButton>button {
            background-color: #2F4F4F;  /* Dark Gray Background */
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-size: 1rem;
            cursor: pointer;
        }
        .stButton>button:hover {
            background-color: #45a049;  /* Light Green Background on Hover */
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
