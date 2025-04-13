## Project Overview
This project applies Natural Language Processing (NLP) and supervised machine learning to classify social media posts (tweets) as malicious or not. The goal is to demonstrate how automated content moderation systems can detect harmful speech at scale while minimizing false positives. This project explores an ML pipeline capable of detecting such content using publicly available datasets.

## Methodology
### Dataset
- (insert sources of dataset used to train supervised ML models)
### Data Preprocessing
- Removal of usernames, URLs, and special characters
- Lowercasing text
- Tokenization (`nltk` or `spaCy`)
- Stopword removal
- Lemmatization
- TF-IDF vectorization for feature extraction
### Model Training
Baseline models implemented (we'll compare the performance of different models):
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier
Advanced models (optional extension):
- Fine-tuned BERT using Huggingface Transformers
- LSTM-based neural network
### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
Special attention was given to class imbalance handling and ethical considerations surrounding false positives and false negatives.

## Ethical Considerations
Automated moderation is prone to:
- Bias in training data
- Misclassification of slang, dialects, or minority language
- Over-removal of content reducing free expression
These challenges highlight the importance of combining machine learning with human moderation review systems.

## Deployment
- Flask or FastAPI?
- Application Idea: user sends link of tweet/post or enters the text of the tweet/post and our app tells the user if it's malicious or not
