import os
os.environ["STREAMLIT_WATCHER_PAUSE_ON_NO_HANDLE"] = "1"

import streamlit as st
import re
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import time

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
# def get_tweet_text(tweet_url):
#     options = Options()
#     options.add_argument("--headless")
#     options.add_argument("--disable-gpu")
#     options.add_argument("--window-size=1920x1080")
#     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

#     try:
#         driver.get(tweet_url)
#         time.sleep(3)  # wait for tweet to load
#         tweet_text = driver.find_element("xpath", '//div[@data-testid="tweetText"]').text
#     except Exception as e:
#         tweet_text = None
#     driver.quit()
#     return tweet_text
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_tweet_text(tweet_url, max_retries=2):
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920x1080")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    
    tweet_text = None
    attempt = 0
    
    while attempt < max_retries:
        driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()))
        try:
            driver.get(tweet_url)

            # ⏳ Wait up to 10 seconds for the div
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.XPATH, '//div[@data-testid="tweetText"]'))
            )

            # ✅ Grab the tweet text
            tweet_div = driver.find_element(By.XPATH, '//div[@data-testid="tweetText"]')
            tweet_text = tweet_div.text.strip()

            if tweet_text:
                print(f"✅ Tweet scraped successfully on attempt {attempt+1}")
                driver.quit()
                return tweet_text

        except Exception as e:
            print(f"❌ Error scraping tweet (attempt {attempt+1}):", e)
            attempt += 1
            driver.quit()
            time.sleep(2)  # wait a little before retrying

    print("❌ Failed to scrape tweet after retries.")
    return None

# def get_tweet_text(tweet_url):
#     from selenium.webdriver.common.by import By
#     from selenium.webdriver.support.ui import WebDriverWait
#     from selenium.webdriver.support import expected_conditions as EC

#     options = Options()
#     options.add_argument("--headless")
#     options.add_argument("--disable-gpu")
#     options.add_argument("--window-size=1920x1080")
#     options.add_argument("--no-sandbox")
#     options.add_argument("--disable-dev-shm-usage")
#     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

#     tweet_text = None
#     try:
#         driver.get(tweet_url)

#         # Wait up to 10 seconds for the tweetText div to appear
#         WebDriverWait(driver, 10).until(
#             EC.presence_of_element_located((By.XPATH, '//div[@data-testid="tweetText"]'))
#         )

#         # Now grab that div
#         tweet_div = driver.find_element(By.XPATH, '//div[@data-testid="tweetText"]')
#         tweet_text = tweet_div.text.strip()

#     except Exception as e:
#         print("❌ Error fetching tweet:", e)
#     finally:
#         driver.quit()
    
#     return tweet_text

# def get_tweet_text(tweet_url):
#     options = Options()
#     options.add_argument("--headless")
#     options.add_argument("--disable-gpu")
#     options.add_argument("--window-size=1920x1080")
#     options.add_argument("--no-sandbox")
#     options.add_argument("--disable-dev-shm-usage")
#     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

#     tweet_text = None
#     try:
#         driver.get(tweet_url)
#         time.sleep(5)  # Longer wait time — Twitter is slow
#         tweet_text_element = driver.find_element("xpath", '//article//div[@lang]')
#         tweet_text = tweet_text_element.text
#     except Exception as e:
#         print("Error fetching tweet:", e)
#     finally:
#         driver.quit()
    
#     return tweet_text


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




