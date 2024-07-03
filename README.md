# Machine Learning and Deep Learning work

This repository contains implementations of various machine learning and deep learning models, as well as projects related to sentiment analysis.

## Table of Contents

1. [Introduction](#introduction)
2. [Supervised Learning](#supervised-learning)
    - [Regression](#regression)
    - [Classification](#classification)
3. [Deep Learning](#deep-learning)
    - [Neural Networks](#neural-networks)
    - [CNN and Autoencoders](#cnn-and-autoencoders)
4. [Projects](#projects)
    - [Sentiment Analysis](#sentiment-analysis)
5. [Usage](#usage)

---

## Introduction

This repository showcases implementations of various machine learning and deep learning models, along with projects focusing on sentiment analysis. Each implementation includes detailed explanations, code samples, and performance evaluations where applicable.

---

## Supervised Learning

### Regression

1. **Linear Regression**
   - Implemented linear regression from scratch using Python and numpy.
   - Implemented linear regression using `scikit-learn`.
   - Implemented Lasso, Ridge, and Elastic Net regression techniques.

### Classification

1. **Logistic Regression**
   - Implemented logistic regression using `scikit-learn`.
   
2. **Decision Tree Classifier**
   - Implemented decision tree classifier using `scikit-learn`.

3. **Random Forest Classification**
   - Implemented random forest classifier using `scikit-learn`.

---

## Deep Learning

### Neural Networks

1. **Single Layer Neural Network**
   - Implemented a basic neural network with one hidden layer from scratch using numpy.
   - Implemented a multiple layer neural network from scratch.

2. **Feed Forward on MNIST Data**
   - Implemented a feedforward neural network for digit recognition using Keras.

### CNN and Autoencoders

1. **Convolutional Neural Network (CNN)**
   - Implemented a CNN for digit recognition on MNIST data using Keras.
   
2. **Autoencoders**
   - Implemented single layer encoder and convolutional autoencoders using Keras.

---

## Projects

### Sentiment Analysis

1. **Using Machine Learning Models and Vectorizers**
   - Implemented sentiment analysis using SVM, Naive Bayes, and various vectorization techniques (e.g., CountVectorizer, TF-IDF, Word2Vec, FastText).

- **Top Performing Models**: Support Vector Classifier (SVC), Logistic Regression, Multinomial Naive Bayes, and Random Forest consistently performed better across different vectorizers (tf-idf, count, word2vec, fasttext). These models achieved accuracies ranging from 82% to 85%.
  
- **Performance Variation**: Models like K-Nearest Neighbors (KNN) and Decision Trees generally showed lower performance compared to the top models, with accuracies ranging from 50% to 68%.

- **Impact of Vectorization**: tf-idf and count vectorization methods generally produced better results compared to non pretrained word embeddings (word2vec, fasttext) across most models.

- **Recommendations**: Based on these results, SVC, Logistic Regression, Multinomial Naive Bayes, and Random Forest are recommended for sentiment analysis tasks due to their higher accuracy and consistent performance across different vectorization techniques.

This summary helps guide the selection of models and vectorization methods for sentiment analysis tasks based on their performance metrics in this study.


2. **Using Deep Learning Models**
   - Implemented sentiment analysis using RNN, LSTM, GRU, and their bidirectional variants.

---

## Usage

To run any of the implementations, follow these steps:

1. Clone the repository:
   ```bash
   !git clone https://github.com/ravipati-venkatesh/Data-Science.git
   cd Data-Science
