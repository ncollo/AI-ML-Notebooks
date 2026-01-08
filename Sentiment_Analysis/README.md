# Sentiment Analysis with NLTK

This notebook demonstrates how to perform sentiment analysis using the Natural Language Toolkit (NLTK) in Python. It covers the basics of text preprocessing, applying NLTK's VADER sentiment analyzer, and evaluating the results.

## Table of Contents
1.  [Introduction](#introduction)
2.  [Sentiment Analysis Use Cases](#sentiment-analysis-use-cases)
3.  [Ways to Perform Sentiment Analysis in Python](#ways-to-perform-sentiment-analysis-in-python)
4.  [Step 1 - Import Libraries and Load Dataset](#step-1---import-libraries-and-load-dataset)
5.  [Step 2 - Preprocess Text](#step-2---preprocess-text)
6.  [Step 3 - NLTK Sentiment Analyzer](#step-3---nltk-sentiment-analyzer)
7.  [Evaluation](#evaluation)
8.  [Conclusion](#conclusion)

## Introduction
Sentiment analysis, a subset of Natural Language Processing (NLP), aims to classify texts into sentiments such as positive, negative, or neutral. This notebook focuses on deciphering the underlying mood, emotion, or sentiment of a text, also known as Opinion Mining.

## Sentiment Analysis Use Cases
Sentiment analysis provides valuable insights for data-driven decisions. Key use cases include:
*   **Social Media Monitoring for Brand Management**: Gauging public opinion about a brand.
*   **Product/Service Analysis**: Understanding market reception of products or services through customer reviews.
*   **Stock Price Prediction**: Analyzing news headlines to predict stock movements.

## Ways to Perform Sentiment Analysis in Python
Python offers various methods for sentiment analysis, including:
*   Using Text Blob
*   Using VADER
*   Using Bag of Words Vectorization-based Models
*   Using LSTM-based Models
*   Using Transformer-based Models

This notebook specifically implements the VADER approach from NLTK.

## Step 1 - Import Libraries and Load Dataset

This step involves importing necessary libraries like `pandas` for data manipulation and various modules from `nltk` for text processing. The NLTK corpus, including stopwords and the VADER lexicon, is downloaded. The dataset used is an Amazon review dataset loaded directly from a URL into a pandas DataFrame.

## Step 2 - Preprocess Text

Text preprocessing is crucial for sentiment analysis. A function `preprocess_text` is defined to:
1.  **Tokenize** the text into individual words.
2.  **Remove Stop Words** to eliminate common words that do not carry significant sentiment.
3.  **Lemmatize** tokens to reduce words to their base form.

This function is then applied to the `reviewText` column of the DataFrame.

## Step 3 - NLTK Sentiment Analyzer

The `SentimentIntensityAnalyzer` from NLTK's VADER module is used to determine the sentiment of the preprocessed text. A `get_sentiment` function is created that:
1.  Obtains polarity scores (positive, negative, neutral) for a given text.
2.  Classifies the sentiment as `1` (positive) if the positive score is greater than `0`, and `0` (negative/neutral) otherwise.

This function is applied to the `reviewText` column, creating a new `sentiment` column in the DataFrame.

## Evaluation

The performance of the sentiment analysis model is evaluated using a confusion matrix and a classification report from `sklearn.metrics`. This allows for a comparison between the actual sentiment (`Positive` column) and the predicted sentiment (`sentiment` column).

## Conclusion

NLTK is a powerful and flexible library for sentiment analysis and other NLP tasks. This notebook demonstrated its use for preprocessing text data, applying VADER's sentiment analyzer, and evaluating the model. By mastering these techniques, one can gain valuable insights from text data for various applications.
