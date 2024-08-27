# Sentiment Analysis Using Deep RNN

This repository contains a deep Recurrent Neural Network (RNN) model for sentiment analysis, designed to classify text into three sentiment categories: Positive, Negative, and Neutral. The model was built using TensorFlow and Keras, and the app is deployed using Streamlit.

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project aims to analyze text data to determine the sentiment behind it. The model leverages a deep RNN with GRU layers to capture the sequential nature of text data. The trained model is integrated into a web application using Streamlit, allowing users to input text and receive sentiment predictions.

## Model Architecture

- **Embedding Layer**: Converts words into dense vectors of fixed size.
- **GRU Layers**: Two Gated Recurrent Unit (GRU) layers are used to capture temporal dependencies in the text data.
- **Dropout**: Added to reduce overfitting by randomly dropping units.
- **Dense Layer**: Final classification layer with a softmax activation function to output the probability of each sentiment class.

## Dataset

The model was trained on a dataset consisting of text samples labeled with three sentiment classes:
- **Positive Sentiment**
- **Negative Sentiment**
- **Neutral Sentiment**

The data was preprocessed, tokenized, and padded to ensure uniform input size for the model.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/SentimentAnalysis.git
   cd SentimentAnalysis
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # For Windows
   source venv/bin/activate  # For macOS/Linux
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Usage

- **Input Text**: You can input any text in the provided text box within the Streamlit app.
- **Output Sentiment**: The app will display the predicted sentiment label (Positive, Negative, or Neutral) based on the input text.

## Results

The model achieves good accuracy on the validation dataset and performs well in real-time sentiment prediction through the Streamlit app.

- **Best Epoch**: The model performs optimally at 8 epochs.
- **Validation Accuracy**: Validation accuracy is tracked and displayed during training.

## Contributing

If you'd like to contribute to this project, please fork the repository and use a feature branch. Pull requests are welcome.
