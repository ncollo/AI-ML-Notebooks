```markdown
# Handling Imbalanced Datasets: Spam vs. Ham Classification

This notebook demonstrates various techniques to handle imbalanced datasets, specifically in the context of Spam vs. Ham classification using the classic SMS Spam Collection dataset.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Preprocessing and Feature Engineering](#preprocessing-and-feature-engineering)
- [Baseline Model](#baseline-model)
- [Imbalance Handling Techniques](#imbalance-handling-techniques)
  - [1. Random Undersampling](#1-random-undersampling)
  - [2. Random Oversampling](#2-random-oversampling)
  - [3. SMOTE (Synthetic Minority Over-sampling Technique)](#3-smote-synthetic-minority-over-sampling-technique)
  - [4. Class Weighting](#4-class-weighting)
- [Evaluation](#evaluation)
- [Summary](#summary)

## Introduction
Class imbalance is a common problem in machine learning where the number of observations belonging to one class is significantly lower than those belonging to other classes. This notebook explores and compares several methods to address this issue in a spam detection task.

## Dataset
- **Name**: SMS Spam Collection Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Description**: Contains SMS messages labeled as either 'ham' (legitimate) or 'spam'. The dataset is inherently imbalanced, with a much larger number of 'ham' messages compared to 'spam' messages.

## Preprocessing and Feature Engineering
1.  **Loading Data**: The dataset is loaded from `spam.csv` (or directly from the UCI link).
2.  **Column Renaming**: Columns are renamed to `label` and `message`.
3.  **Label Encoding**: 'ham' is encoded as 0 and 'spam' as 1.
4.  **Text Preprocessing**: Messages are converted to lowercase.
5.  **Train-Test Split**: Data is split into training and testing sets (70% train, 30% test) with stratification to maintain class distribution.
6.  **TF-IDF Vectorization**: Text messages are transformed into numerical feature vectors using `TfidfVectorizer`.

## Baseline Model
A Logistic Regression model is trained on the original, imbalanced training data to establish a baseline performance. Performance is evaluated using `classification_report` and `confusion_matrix`.

## Imbalance Handling Techniques
Four different techniques are applied to the training data to mitigate the effects of class imbalance. A Logistic Regression model is trained for each technique, and its performance is evaluated on the test set.

### 1. Random Undersampling
The majority class samples are randomly removed until the class distribution is more balanced.

### 2. Random Oversampling
The minority class samples are randomly duplicated until the class distribution is more balanced.

### 3. SMOTE (Synthetic Minority Over-sampling Technique)
Synthetic samples of the minority class are generated based on the feature space similarities between existing minority class samples, rather than simply duplicating them.

### 4. Class Weighting
Instead of resampling the data, the Logistic Regression model is configured to assign different weights to the classes during training, giving more importance to the minority class.

## Evaluation
For each model (baseline and those trained with imbalance handling techniques), the following metrics are presented:
- **Classification Report**: Precision, Recall, F1-score, and Support for both 'Ham' and 'Spam' classes.
- **Confusion Matrix**: Visual representation of true positives, true negatives, false positives, and false negatives.

## Summary
This notebook provides a comprehensive comparison of different strategies for handling imbalanced datasets. The choice of the best technique often depends on the specific dataset, problem, and evaluation metrics prioritized. The results demonstrate how these methods can improve the detection of the minority class (spam in this case) compared to a simple baseline model.