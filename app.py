import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

nltk.download('stopwords')
nltk.download('punkt')

# Load model and vectorizer
model = pickle.load(open(r"E:\Sentiment Analysis\logistic_model.pkl", 'rb'))
vectorizer = pickle.load(open(r"E:\Sentiment Analysis\vectorizer.pkl", 'rb'))


# Text preprocessing
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()


def data_processing(text):
    text = text.lower()
    text = re.sub(r"https\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    filtered_text = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(filtered_text)


def get_sentiment(polarity):
    if polarity < 0:
        return "Negative"
    elif polarity == 0:
        return "Neutral"
    else:
        return "Positive"


# Streamlit UI
st.title("💬 Sentiment Analysis of Tweets")
st.write("This app analyzes the sentiment of tweets using a machine learning model.")

tweet = st.text_area("Enter a tweet:")

if st.button("Analyze"):
    if tweet:
        processed_text = data_processing(tweet)
        polarity = TextBlob(tweet).sentiment.polarity
        sentiment = get_sentiment(polarity)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]

        st.subheader("ML Model Prediction")
        st.write(f"Predicted Sentiment: **{prediction}**")

        st.subheader("TextBlob Polarity Check")
        st.write(f"TextBlob Sentiment: **{sentiment}** | Polarity Score: **{polarity:.2f}**")
    else:
        st.warning("Please enter a tweet for analysis.")
