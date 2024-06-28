#!/usr/bin/env python
# coding: utf-8

import os
import sys
import re
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from multiprocessing import Pool, cpu_count
import time
from sklearn.preprocessing import MinMaxScaler

# Add the src paths to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "Model")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "Features")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "Data")))

print(sys.path)

import LoadData as LD
import DataPreProcessing as dp
import Models

# Data preprocessing
processed_file_path = "../Data/Process/sample_preprocessed_data.xlsx"
if not os.path.exists(processed_file_path):
    print("inside loop")
    path = '../Data/Raw/IMDB Dataset sample.csv'
    df = LD.load_data(path)
    df['review'] = df['review'].apply(lambda x: re.sub(r'<br /><br />', '', x))
    df['review'] = df['review'].apply(lambda x: re.sub(r'\'', '', x))
    df['cleaned_review'] = df['review'].apply(lambda x: dp.preprocess_text(x))
    df.to_excel(processed_file_path, index=False)

# Load preprocessed data
df = pd.read_excel(processed_file_path)
label_encoder = LabelEncoder()
df['sentiment_encoded'] = label_encoder.fit_transform(df['sentiment'])

vectorizer_dict = [
    {'vectorizer': 'count', 'review_column': 'cleaned_review', 'pretrained': None},
    {'vectorizer': 'tf-idf', 'review_column': 'cleaned_review', 'pretrained': None},
    {'vectorizer': 'word2vec', 'review_column': 'cleaned_review', 'pretrained': False},
    {'vectorizer': 'fasttext', 'review_column': 'cleaned_review', 'pretrained': False},
#     {'vectorizer': 'word2vec', 'review_column': 'cleaned_review', 'pretrained': True},
#     {'vectorizer': 'fasttext', 'review_column': 'cleaned_review', 'pretrained': True},
#     {'vectorizer': 'word2vec', 'review_column': 'review', 'pretrained': False},
#     {'vectorizer': 'fasttext', 'review_column': 'review', 'pretrained': False},
    # {'vectorizer': 'word2vec', 'review_column': 'review', 'pretrained': True},
    # {'vectorizer': 'fasttext', 'review_column': 'review', 'pretrained': True},
]


# Run one model at a time
models = list({
    # 'Logistic Regression',
    # 'Multinomial Naive Bayes',
#     'Decision Tree',
    'Random Forest',
    'SVC',
    'KNN'
})

# Function to process a single combination of vectorizer and model
def process_combination(params):
    dic, model_name, df = params
    t = time.time()
    df_ = dp.vectorization(df, vectorizer=dic['vectorizer'], review_column=dic['review_column'], pretrained=dic['pretrained'])
    X_train, X_test, y_train, y_test = Models.train_test_data_split(df_)

    if model_name=='Multinomial Naive Bayes' and (dic['vectorizer']=='word2vec' or dic['vectorizer']=='fasttext'):
      # Apply MinMaxScaler to transform the feature vectors to be non-negative
      scaler = MinMaxScaler()
      X_train = scaler.fit_transform(X_train)
      X_test = scaler.transform(X_test)
    Models.fit_models(X_train, y_train, model_name, dic['vectorizer'])
    accuracy, precision, recall, f1 = Models.predict_models_summary(X_test, y_test, model_name, dic['vectorizer'])
    return dic['vectorizer'], model_name, accuracy, precision, recall, f1, time.time() - t

# Prepare parameters for multiprocessing
params = [(dic, model_name, df) for dic in vectorizer_dict for model_name in models]

# Use multiprocessing to process combinations
if __name__ == '__main__':
    print(cpu_count())
    with Pool(cpu_count() - 1) as pool:
        results = pool.map(process_combination, params)

    # Create summary DataFrame
    df_summary = pd.DataFrame(results, columns=['vectorizer', 'model_name', 'accuracy', 'precision', 'recall', 'f1', 'time_taken'])

    summary_file_path = f'../Data/Process/{models[0].replace(" ", "_")}.xlsx'
#     print(df_summary)
    df_summary.to_excel(summary_file_path, index=False)

