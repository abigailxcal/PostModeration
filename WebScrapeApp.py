import os
os.environ["STREAMLIT_WATCHER_PAUSE_ON_NO_HANDLE"] = "1"

import streamlit as st
import re
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from selenium import webdriver
from selenium.webdriver.chrome.service import Service 
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

# TEST TWEET URLS:
# ******* not malicious ********:
# https://x.com/ppyowna/status/1916866370949755368
# https://x.com/i_zzzzzz/status/1917742577450680750

# ******** malicious **********:    
# https://x.com/DonLagoTV/status/1910518593340543479
# https://x.com/Chloe_w34/status/1911345323479351453
# https://x.com/RubiconCrosser1/status/1916312867126604046     very bad lmao

#NTS: this web scrape file will only work with the base cnda environment. for training/ipynb files, ASl-Translator env works better 


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

# ===================== MODEL LOADING =====================
@st.cache_resource
def load_model():
    model = DistilBertForSequenceClassification.from_pretrained("Bert_Model")
    tokenizer = DistilBertTokenizer.from_pretrained("Bert_Model")
    model.eval()
    return model, tokenizer

model, tokenizer = load_model()

# ===================== TEXT CLEANING =====================
def clean_text(text):
    text = text.casefold()
    text = re.sub(r'(rt)?\s?@\w+:?', ' ', text)
    text = re.sub(r'http\S+', ' ', text)
    return text

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

# ===================== TWEET SCRAPER =====================
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
def get_tweet_text(tweet_url, max_retries=2):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")
    options.add_argument('--disable-blink-features=AutomationControlled')
    # options.add_argument("--disable-popup-blocking")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
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
st.title("TweetSweep AI")
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
        st.error("Could not fetch tweet. Make sure the URL is public and correct.")




