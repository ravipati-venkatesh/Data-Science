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
import gensim.downloader as api
import gensim.models.fasttext as fasttext

# Add the src paths to sys.path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "Model")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "Features")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "Data")))

print(sys.path)

import LoadData as LD
import DataPreProcessing as dp
import Models



import requests
import shutil
import gzip

url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz'
target_file = 'cc.en.300.bin.gz'

# Download the file
with requests.get(url, stream=True) as r:
    with open(target_file, 'wb') as f:
        shutil.copyfileobj(r.raw, f)

# Decompress the file if needed
with gzip.open(target_file, 'rb') as f_in:
    with open('cc.en.300.bin', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Remove the compressed file
os.remove(target_file)



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
#     {'vectorizer': 'count', 'review_column': 'cleaned_review', 'pretrained': None},
#     {'vectorizer': 'tf-idf', 'review_column': 'cleaned_review', 'pretrained': None},
#     {'vectorizer': 'word2vec', 'review_column': 'cleaned_review', 'pretrained': False},
#     {'vectorizer': 'fasttext', 'review_column': 'cleaned_review', 'pretrained': False},
    {'vectorizer': 'word2vec', 'review_column': 'cleaned_review', 'pretrained': True},
    {'vectorizer': 'fasttext', 'review_column': 'cleaned_review', 'pretrained': True},
#     {'vectorizer': 'word2vec', 'review_column': 'review', 'pretrained': False},
#     {'vectorizer': 'fasttext', 'review_column': 'review', 'pretrained': False},
    {'vectorizer': 'word2vec', 'review_column': 'review', 'pretrained': True},
    {'vectorizer': 'fasttext', 'review_column': 'review', 'pretrained': True},
]


# Run one model at a time
models = list({
#     'Logistic Regression',
    'Multinomial Naive Bayes',
    'Decision Tree',
#     'Random Forest',
#     'SVC',
#     'KNN'
})

# Function to process a single combination of vectorizer and model
def process_combination(params):
    dic, model_name, df, word2vec_model, fasttext_model = params
    t = time.time()
    pretrained = dic['pretrained']
    if pretrained and dic['vectorizer']=='word2vec':
        model = word2vec_model
    elif pretrained and dic['vectorizer']=='fasttext':
        model = fasttext_model
    else:
        model = None
        
    df_ = dp.vectorization(df, vectorizer=dic['vectorizer'], review_column=dic['review_column'], pretrained=dic['pretrained'], model=model)
    X_train, X_test, y_train, y_test = Models.train_test_data_split(df_)

    if model_name=='Multinomial Naive Bayes' and (dic['vectorizer']=='word2vec' or dic['vectorizer']=='fasttext'):
      # Apply MinMaxScaler to transform the feature vectors to be non-negative
      scaler = MinMaxScaler()
      X_train = scaler.fit_transform(X_train)
      X_test = scaler.transform(X_test)
    Models.fit_models(X_train, y_train, model_name, dic['vectorizer'])
    accuracy, precision, recall, f1 = Models.predict_models_summary(X_test, y_test, model_name, dic['vectorizer'])
    return dic['vectorizer'], model_name, accuracy, precision, recall, f1, dic['pretrained'], dic['review_column']


# Use multiprocessing to process combinations
if __name__ == '__main__':
    
    # Load the pretrained Word2Vec and FastText models once
    word2vec_model = api.load('word2vec-google-news-300')
    fasttext_model = fasttext.load_facebook_vectors('cc.en.300.bin')
    # Prepare parameters for multiprocessing
    params = [(dic, model_name, df, word2vec_model, fasttext_model) for dic in vectorizer_dict for model_name in models]



    print(cpu_count())
    with Pool(cpu_count() - 1) as pool:
        results = pool.map(process_combination, params)

    # Create summary DataFrame
    df_summary = pd.DataFrame(results, columns=['vectorizer', 'model_name', 'accuracy', 'precision', 'recall', 'f1', 'pretrained', 'review_col'])

    summary_file_path = f'../Data/Process/{models[0].replace(" ", "_")}.xlsx'
#     print(df_summary)
    df_summary.to_excel(summary_file_path, index=False)

