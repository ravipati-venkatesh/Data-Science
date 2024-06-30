#!/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, FastText
from gensim.models import KeyedVectors
import spacy


def preprocess_text(text):
    """
    Preprocess the input text using spaCy.

    Args:
    text (str): The text to preprocess.

    Returns:
    List[str]: A list of cleaned and lemmatized tokens.
    """

    # Load the large language model
    nlp = spacy.load("en_core_web_lg")

    # Process the text with spaCy
    doc = nlp(text)
    
    # Initialize an empty list to hold the cleaned tokens
    cleaned_tokens = []
    
    for token in doc:
        # Filter out stop words, punctuation, and tokens with less than 2 characters, spaces, numbers
        if not token.is_stop and not token.is_punct and len(token.text) > 1 and not token.is_space and not token.like_num:
            # Lemmatize the token and add to the cleaned tokens list
            cleaned_tokens.append(token.lemma_)
    
    return cleaned_tokens


def word_embedding(df, model, review_column='review'):

    # Tokenization function
    def preprocess(text):
        return re.findall(r'\b\w+\b', text.lower())
    
    # Function to get vector representation of a review
    def get_review_vector(model, tokens):
        # Get vectors for each word in the review
        word_vectors = [model.wv[word] for word in tokens if word in model.wv]
        # Calculate the mean of the vectors
        if len(word_vectors) == 0:  # Handle empty word_vectors case
            return np.zeros(model.vector_size)
        return np.mean(word_vectors, axis=0)

    # Apply tokenization to each review
    df['tokens'] = df[review_column].apply(preprocess)

    if model == "word2vec":
        # Train Word2Vec model
        word2vec_model = Word2Vec(sentences=df['tokens'], vector_size=100, window=5, min_count=1, workers=4)

        # Apply the function to each review for Word2Vec
        df['vector'] = df['tokens'].apply(lambda tokens: get_review_vector(word2vec_model, tokens))
        
    if model == "fasttext":

        # Train FastText model
        fasttext_model = FastText(sentences=df['tokens'], vector_size=100, window=5, min_count=1, workers=4)

        # Apply the function to each review for FastText
        df['vector'] = df['tokens'].apply(lambda tokens: get_review_vector(fasttext_model, tokens))

    return df[['review', 'vector', 'sentiment_encoded']]


def pre_trained_word_embedding(df, model, review_column='review'):

    # Tokenization function
    def preprocess(text):
        return re.findall(r'\b\w+\b', text.lower())
    
    # Function to get vector representation of a review
    def get_review_vector(model, tokens):
        # Get vectors for each word in the review
        word_vectors = [model[word] for word in tokens if word in model]
        # Calculate the mean of the vectors
        if len(word_vectors) == 0:  # Handle empty word_vectors case
            return np.zeros(model.vector_size)
        return np.mean(word_vectors, axis=0)

    # Apply tokenization to each review
    df['tokens'] = df[review_column].apply(preprocess)

#     if model == "word2vec" or model == "fasttext":

    # Apply the function to each review for Word2Vec
    df['vector'] = df['tokens'].apply(lambda tokens: get_review_vector(model, tokens))

    return df[['review', 'vector', 'sentiment_encoded']]


def vectorization(df, vectorizer, review_column='cleaned_review', pretrained=False, model=None):

    if vectorizer == 'tf-idf':
        # Create TfidfVectorizer instance
        vectorizer_model = TfidfVectorizer()
    elif vectorizer == 'count':
        # Create CountVectorizer instance
        vectorizer_model = CountVectorizer()
    else:
        if pretrained:
            return pre_trained_word_embedding(df, model, review_column=review_column)
        else:
            return word_embedding(df, model=vectorizer, review_column=review_column)

    # Fit and transform the data
    X = vectorizer_model.fit_transform(df[review_column])

    # Convert sparse matrix to dense format
    dense_matrix = X.toarray()

    # Get feature names (i.e., words)
    feature_names = vectorizer_model.get_feature_names_out()

    # Create a DataFrame for better readability
    df_counts = pd.DataFrame(dense_matrix, columns=feature_names)
    df['vector'] = df_counts.values.tolist()
    return df[['review', 'vector', 'sentiment_encoded']]


    
    