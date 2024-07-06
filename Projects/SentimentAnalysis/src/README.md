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
   cd Data-Science/Projects/SentimentAnalysis/src
   
   
   !python Machine_learning_Main.py
   !python Deep_Learning_main.py
   !python bert_roberta_main.py
   !python Bert_Roberta_Lora_main.py