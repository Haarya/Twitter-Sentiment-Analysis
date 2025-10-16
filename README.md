Twitter Sentiment Analysis in Google Colab
This repository contains the code to train a Logistic Regression model for Twitter sentiment analysis. The entire project is designed to be run in a Google Colab environment.
The model is trained on the Sentiment140 dataset to classify tweets as either Positive or Negative.
How to Run This Project in Google ColabFollow these steps carefully to run the code in a single Colab session.
Step 1: 
Open a New Colab NotebookGo to colab.research.google.com.
Click on "New notebook".

Step 2: 
Upload Your Kaggle API KeyTo download the dataset, you need your Kaggle API token.Go to your Kaggle account, click on your profile picture, and go to Account.
Scroll down to the API section and click "Create New API Token". 
This will download a kaggle.json file.In your Colab notebook, click on the folder icon on the left sidebar to open the file explorer.Click the "Upload to session storage" icon (file with an upward arrow) and select the kaggle.json file you just downloaded.

Step 3: 
Train the ModelNow, copy the entire contents of the pythonproject.py script and paste it into the first cell of your Colab notebook.
Run the cell by clicking the play button or pressing Shift + Enter.
This script will:Install the Kaggle library.Move your kaggle.json file to the correct directory.Download and extract the Sentiment140 dataset.Preprocess the text data (stemming, stopword removal).
Train a Logistic Regression model.Evaluate the model and print its accuracy.Save the trained model as trained_model.sav and the vectorizer as vectorizer.sav in your Colab session.This process will take several minutes to complete.

Step 4: 
Make Predictions on New TweetsOnce the training cell has finished running, the model is ready for predictions.Click the "+ Code" button at the top of your notebook to create a new cell.Copy the code below and paste it into this new cell. This code loads your saved model and sets up an interactive prediction loop.# This code should be in a new cell, run AFTER the training is complete.

import pickle
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# --- Load the Saved Model and Vectorizer ---
vectorizer = pickle.load(open('vectorizer.sav', 'rb'))
loaded_model = pickle.load(open('trained_model.sav', 'rb'))

port_stem = PorterStemmer()

# --- Re-create the Preprocessing Function ---
def preprocess_tweet(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

# --- Create the Prediction Function ---
def predict_sentiment(tweet_text):
    if not tweet_text.strip():
        return "Cannot analyze empty text."
    processed_tweet = preprocess_tweet(tweet_text)
    tweet_vector = vectorizer.transform([processed_tweet])
    prediction = loaded_model.predict(tweet_vector)
    return 'Positive' if prediction[0] == 1 else 'Negative'

# --- Interactive Loop ---
print("--- Twitter Sentiment Analyzer ---")
print("Enter a sentence to analyze, or type 'quit' to exit.")

while True:
    user_input = input("\nEnter your text: ")
    if user_input.lower() == 'quit':
        break
    sentiment = predict_sentiment(user_input)
    print(f"--> Predicted Sentiment: {sentiment}")
Run this new cell. You will now be prompted to enter any sentence to see its predicted sentiment. Type quit to stop.File Descriptionspythonproject.py: The main script for the complete training pipeline in Google Colab.prediction_model.py: An example script showing how to load the model and make predictions (the logic from this is used in Step 4 above).
