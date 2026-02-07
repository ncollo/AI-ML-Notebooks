# Spam vs. Ham Classifier

This project implements a machine learning model to classify SMS messages as either "spam" or "ham" (legitimate). The goal is to build an effective text classifier using scikit-learn.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Dependencies](#dependencies)
4. [Procedure](#procedure)
5. [Results and Model Evaluation](#results-and-model-evaluation)
6. [Usage](#usage)

## 1. Project Overview

This repository contains code for an SMS spam detection system. It leverages natural language processing (NLP) techniques to convert text messages into numerical features and then applies a Naive Bayes classifier to distinguish between spam and legitimate messages. The project demonstrates a typical machine learning pipeline from data loading and preprocessing to model training and evaluation.

## 2. Dataset

The dataset used is the "SMS Spam Collection v1" from the UCI Machine Learning Repository.

**Source:** [https://archive.ics.uci.edu/ml/datasets/sms+spam+collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

**Description:** This dataset comprises 5,572 English SMS messages, tagged as either "ham" (legitimate) or "spam".

## 3. Dependencies

To run this project, you will need the following Python libraries:

*   `pandas`
*   `scikit-learn`
*   `requests`
*   `zipfile`
*   `io`

You can install them using pip:

```bash
pip install pandas scikit-learn requests
```

## 4. Procedure

The project follows these steps:

1.  **Install/Import Libraries:** Necessary libraries like pandas and scikit-learn components are imported.
2.  **Load Dataset:** The SMS Spam Collection dataset is downloaded and loaded into a pandas DataFrame.
3.  **Preprocessing:**
    *   Labels (`ham`/`spam`) are encoded into numerical values (`0`/`1`).
4.  **Train-Test Split:** The dataset is split into training and testing sets to evaluate the model's performance on unseen data.
5.  **Text Vectorization (TF-IDF):** The text messages are converted into numerical feature vectors using `TfidfVectorizer`. This process assigns weights to words based on their frequency in a document relative to the entire corpus, filtering out common English stop words.
6.  **Model Training:** A `Multinomial Naive Bayes` classifier is trained on the TF-IDF transformed training data.
7.  **Model Evaluation:** The trained model's performance is evaluated using metrics such as accuracy and a detailed classification report (precision, recall, F1-score).

## 5. Results and Model Evaluation

The Multinomial Naive Bayes classifier achieved the following performance on the test set:

```
Accuracy: 0.97847533632287

Classification Report:
               precision    recall  f1-score   support

           0       0.98      1.00      0.99       966
           1       1.00      0.84      0.91       149

    accuracy                           0.98      1115
   macro avg       0.99      0.92      0.95      1115
weighted avg       0.98      0.98      0.98      1115
```

**Analysis of Results:**

*   **Accuracy (0.978):** The model correctly classified approximately 97.8% of the messages.
*   **Ham Class (Label 0):**
    *   **Precision (0.98):** When the model predicts a message is 'ham', it is correct 98% of the time.
    *   **Recall (1.00):** The model correctly identified 100% of all actual 'ham' messages.
    *   **F1-score (0.99):** A very high F1-score indicates excellent performance for the 'ham' class.
*   **Spam Class (Label 1):**
    *   **Precision (1.00):** When the model predicts a message is 'spam', it is correct 100% of the time. This is excellent, meaning no legitimate messages were incorrectly flagged as spam (no false positives).
    *   **Recall (0.84):** The model identified 84% of all actual 'spam' messages. This means 16% of actual spam messages were missed (false negatives).
    *   **F1-score (0.91):** A good F1-score, though slightly lower than 'ham' due to the recall for spam.

**Conclusion:**
The Multinomial Naive Bayes model performs exceptionally well in classifying 'ham' messages and has perfect precision for 'spam' messages, meaning it avoids flagging legitimate messages as spam. While its recall for spam is 84% (it misses some spam), its high overall accuracy and precision make it a strong candidate for an initial spam filter.

## 6. Usage

To run this project, execute the cells in the provided Jupyter/Colab notebook sequentially. The notebook demonstrates each step of the pipeline from data loading to model evaluation.