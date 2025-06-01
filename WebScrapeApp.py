import os
os.environ["STREAMLIT_WATCHER_PAUSE_ON_NO_HANDLE"] = "1"

import streamlit as st
import re
import torch
import numpy as np
import time
import tensorflow as tf
import pickle
from keras.preprocessing.sequence import pad_sequences
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


# TEST TWEET URLS:
# not malicious:   https://x.com/ppyowna/status/1916866370949755368
# malicious:    https://x.com/DonLagoTV/status/1910518593340543479

#NTS: this web scrape file will only work with the base cnda environment. for training/ipynb files, ASl-Translator env works better 
# NTS: things to visualize in the data:
# - how much of the training data came from each CSV
# - word count of the tweets from each csv that show if malicious tweets tend to be shorter 
# - outliers? how much of the data wasn't in english or how much was actual nonsense 
# April 29 NTS: Bert_Model2 is the most recently trained bert model with the closest accuracy that reflects Rane's bert model metrics
# I should delete my Bert_Model folder and rename Bert_Model2 to Bert_Model 

# Evaluation Results: {'eval_loss': 0.23020809888839722, 'eval_accuracy': 0.9084373817631479, 'eval_precision_not_malicious': 0.9324296985636253, 'eval_recall_not_malicious': 0.879076864390616, 
#                      'eval_f1_not_malicious': 0.9049676025917927, 'eval_precision_malicious': 0.8873689820572037, 'eval_recall_malicious': 0.9373240758115969, 'eval_f1_malicious': 0.9116627121737544,
#                      'eval_runtime': 8.7387, 'eval_samples_per_second': 1209.797, 'eval_steps_per_second': 37.878, 'epoch': 2.0}





# ===================== BACKGROUND STYLING =====================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f9fcd2;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===================== MODEL LOADING: LSTM =====================
@st.cache_resource
def load_lstm_model():
    model = tf.keras.models.load_model('LSTM_Model/lstm_model.h5')
    #load tokenizer
    with open('LSTM_Model/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model, tokenizer
#model, tokenizer = load_lstm_model()

# ===================== MODEL LOADING: Bert =====================
@st.cache_resource
def load_Bert_model():
    model = DistilBertForSequenceClassification.from_pretrained("Bert_Model_reworkedData")
    tokenizer = DistilBertTokenizer.from_pretrained("Bert_Model_reworkedData")
    model.eval()
    return model, tokenizer
model, tokenizer = load_Bert_model()

# ===================== TEXT CLEANING =====================
def clean_text(text):
    text = text.casefold()
    text = re.sub(r'(rt)?\s?@\w+:?', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    return text

def clean_text_lstm(text):
    text = re.sub(r'(rt)?\s?@\w+:?', '', text)
    text = re.sub(r'(RT)?\s?@\w+:?', '', text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^A-Za-z\'\s]', '', text)
    return text.lower().strip()

# ===================== CLASSIFICATION =====================
def classify_text(text):
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs).item()
    label = "🚨 Malicious" if pred == 1 else "✅ Not Malicious"
    confidence = round(probs[0][pred].item() * 100, 2)
    return f"{label} ({confidence}% confidence)", cleaned


def classify_text_lstm(text):
    cleaned = clean_text_lstm(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=50, padding='post')  # adjust if your LSTM uses a different maxlen
    prob = model.predict(padded)[0][0]
    
    label = "🚨 Malicious" if prob > 0.7 else "✅ Not Malicious"
    confidence = round(prob * 100, 2) if prob > 0.5 else round((1 - prob) * 100, 2)

    return f"{label} ({confidence}% confidence)", cleaned


# ===================== TWEET SCRAPER =====================
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_tweet_text(tweet_url, max_retries=2):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
    #  this doesn't work for me for some reason vvv
    #driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),options=options)

    try:
        driver.get(tweet_url)
        time.sleep(3)  # wait for tweet to load
        tweet_text = driver.find_element("xpath", '//div[@data-testid="tweetText"]').text
    except Exception as e:
        tweet_text = None
    driver.quit()
    return tweet_text

# ===================== STREAMLIT UI =====================
st.title("🧠 TweetSweep AI")
tweet_url = st.text_input("Paste a public tweet URL:")

if tweet_url:
    st.info("Fetching tweet...")
    tweet_text = get_tweet_text(tweet_url)

    if tweet_text:
        st.success("Tweet found!")
        label, cleaned_text = classify_text(tweet_text)
        st.markdown(f"### Result: {label}")
        st.markdown(f"**Tweet Content:**\n> {cleaned_text}")
    else:
        st.error("❌ Could not fetch tweet. Make sure the URL is public and correct.")




