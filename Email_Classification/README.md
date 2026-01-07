## Goal
Classify emails into Spam or Ham (non-spam) using a machine learning classifier trained on text data.
## Dataset
SpamAssasin Dataset (from Kaggle)

## Notebook Summary

### Step 1: Import Necessary Libraries
The notebook starts by importing essential libraries for data manipulation (pandas, numpy), regular expressions (re), natural language processing (nltk), machine learning (sklearn), and visualization (matplotlib, seaborn).

### Step 2: Load and Explore the Dataset
- The SpamAssassin dataset is downloaded from KaggleHub and loaded into a pandas DataFrame.
- The dataset contains `email_text` and `target` (renamed to `label`) columns.
- Basic data exploration is performed using `df.head()`, `df.tail()`, `df.info()`, and `df['label'].value_counts()` to understand its structure and class distribution.

### Data Cleaning
- NLTK stopwords are downloaded.
- A `Clean_text` function is defined to preprocess email text, which includes:
    - Removing links.
    - Removing email addresses.
    - Removing punctuation.
    - Removing digits.
    - Removing stopwords.
    - Performing lemmatization.
    - Performing stemming.
- A new column `Clean_email` is added to the DataFrame by applying this cleaning function to the `email_text` column.

### Data Visualization
- The distribution of 'label' (Spam vs Ham) is visualized using a histogram, a pie chart, and a count plot to show the class imbalance.

### Step 3: Preprocess the Data
- The dataset is split into training and testing sets (`X_train`, `X_test`, `y_train`, `y_test`).
- TF-IDF Vectorizer is used to convert the textual `email_text` into numerical features, excluding English stopwords.

### Step 4: Train a Naive Bayes Classifier
- A Multinomial Naive Bayes model is initialized and trained on the TF-IDF transformed training data.
- Predictions are made on the test set.

### Step 5: Evaluate the Model
- The model's performance is evaluated using:
    - **Accuracy Score**.
    - **Classification Report** (including precision, recall, and F1-score).
    - **Confusion Matrix** visualized using a heatmap.

### Step 6: Model Interpretation and Discussion
- This section prompts discussion on model performance, potential improvements, the importance of precision in spam detection, and how Naive Bayes handles text data.

### Extension: Experiment with Other Models
- The notebook suggests trying other models like Logistic Regression or Support Vector Machines.
- It demonstrates an example of training and evaluating a **Logistic Regression** model, showing its accuracy.