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
    - [Usage](#usage)

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

# Projects

## Sentiment Analysis (Project 1)

1. **Using Machine Learning Models and Vectorizers**
   - Implemented sentiment analysis using SVM, Naive Bayes, and various vectorization techniques (e.g., CountVectorizer, TF-IDF, Word2Vec, FastText).

   - **Top Performing Models**: Support Vector Classifier (SVC), Logistic Regression, Multinomial Naive Bayes, and Random Forest consistently performed better across different vectorizers (tf-idf, count, word2vec, fasttext). These models achieved accuracies ranging from 82% to 85%.
  
   - **Performance Variation**: Models like K-Nearest Neighbors (KNN) and Decision Trees generally showed lower performance compared to the top models, with accuracies ranging from 50% to 68%.

   - **Impact of Vectorization**: tf-idf and count vectorization methods generally produced better results compared to non-pretrained word embeddings (word2vec, fasttext) across most models.

   - **Recommendations**: Based on these results, SVC, Logistic Regression, Multinomial Naive Bayes, and Random Forest are recommended for sentiment analysis tasks due to their higher accuracy and consistent performance across different vectorization techniques. And as the reviews were of higher length KNN can't focus on important features and decision tree is also overfitting the data.

2. **Using Deep Learning Models**
   - Implemented sentiment analysis using RNN, LSTM, GRU, and their bidirectional variants.
   - These models accuracies ranging from 80% to 85%
   - RNN take less time with performance od around 82% while LSTM and GRU achieve the accuracy of around 85%
   - These models are getting overfit if ran for more iterations
   - **Recommendations**: Based on these results if the length of the input strings were higher then the deep learning models will overfit, it can't able to learn the context of the paragraph.

3. **Using transformer based Models (BERT, Roberta)**
   - Implemented sentiment analysis using BERT, Roberta
   - These models accuracies ranging from 92% to 95%
   - **Recommendations**: Based on these results transformer based Models outperforms all the other models.

4. **Using LLM Models (flan-t5-large)**
   - Implemented sentiment analysis using Langchain and HuggingFaceHub
   - These models accuracies ranging from 98%
   - **Recommendations**: Based on these results LLM Models outperforms all the other models but it has the issue of free usage limit. 

#### Usage

To run any of the implementations, follow these steps:

1. Clone the repository:
   ```bash
   !git clone https://github.com/ravipati-venkatesh/Data-Science.git
   cd Data-Science/Projects/SentimentAnalysis/src
   
   # To run machine learning models 
   !python Machine_learning_Main.py
   
   # To run Deep learning models 
   !python Deep_Learning_main.py
   
   # To run transformer based models
   !python bert_roberta_main.py
   
   # To run transformer based models using PEFT(LORA)
   !python Bert_Roberta_Lora_main.py (WIP)
   
   # To run on top of LLM model
   !python Sentiment_analysis_langchain.py
   
#### Acknowledgements

- **Transformers (Hugging Face)**: Provides state-of-the-art natural language processing models and tools.
- **TensorFlow**: A powerful library for numerical computation and Deep learning.
- **Keras**: High-level neural networks API, written in Python and capable of running on top of TensorFlow.
- **Gensim**: Library for topic modeling and document similarity analysis.
- **LangChain**: Framework for building applications with language models.
- **Pandas**: Essential for data manipulation and analysis.
- **Scikit-Learn**: A comprehensive library for machine learning in Python.
- **Multiprocessing**: Provides support for concurrent execution of code using processes.

## Question Answering Application (Project 2)

This application processes articles from URLs, splits the text into chunks, generates embeddings, creates a vector database, and answers user questions based on the provided context using a language model.

#### Features

- **Load Data**: Fetches and processes data from provided URLs.
- **Text Splitting**: Splits the text into manageable chunks for better processing.
- **Vector Database Creation**: Creates a FAISS vector database from the processed text.
- **LLM Model**: Uses a pre-trained language model (Flan-T5) from HuggingFace for generating answers.
- **Question Answering**: Allows users to input questions and get answers based on the processed data.

#### Usage

To run any of the implementations, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/ravipati-venkatesh/Data-Science.git
    ```
   
2. Set up your environment variables:

    Create a `.env` file in the project directory and add your HuggingFace API token:

    ```plaintext
    HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
    ```

3. Running the App

    ```bash
    cd Data-Science/Projects/QA_based_on_articles
    streamlit run Main.py
    ```

4. Open your web browser and go to `http://localhost:8501`.

#### Example URLs

- https://www.ibm.com/topics/large-language-models
- https://www.cloudflare.com/en-in/learning/ai/what-is-large-language-model/
- https://www.elastic.co/what-is/large-language-models
- https://en.wikipedia.org/wiki/Large_language_model

#### Acknowledgements

- **LangChain**: A framework for building applications with language models. LangChain simplifies the process of working with various language models and integrating them into applications.

- **Hugging Face**: Provides a vast collection of pre-trained language models and tools for natural language processing. Their models and libraries are fundamental to the implementation of state-of-the-art NLP solutions.

- **Streamlit**: An open-source app framework for Machine Learning and Data Science projects. Streamlit allows for the rapid creation of interactive web applications to visualize and share results.

- **SentenceTransformers**: Offers pre-trained models and tools for generating sentence embeddings. These embeddings are useful for various NLP tasks, including semantic textual similarity and clustering.

- **FAISS**: A library for efficient similarity search and clustering of dense vectors. FAISS enables fast and scalable retrieval of similar items in large datasets.

