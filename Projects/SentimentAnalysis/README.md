
---

## Projects

### Sentiment Analysis

1. **Using Machine Learning Models and Vectorizers**
   - Implemented sentiment analysis using SVM, Naive Bayes, and various vectorization techniques (e.g., CountVectorizer, TF-IDF, Word2Vec, FastText).

   - **Top Performing Models**: Support Vector Classifier (SVC), Logistic Regression, Multinomial Naive Bayes, and Random Forest consistently performed better across different vectorizers (tf-idf, count, word2vec, fasttext). These models achieved accuracies ranging from 82% to 85%.
  
   - **Performance Variation**: Models like K-Nearest Neighbors (KNN) and Decision Trees generally showed lower performance compared to the top models, with accuracies ranging from 50% to 68%.

   - **Impact of Vectorization**: tf-idf and count vectorization methods generally produced better results compared to non-pretrained word embeddings (word2vec, fasttext) across most models.

   - **Recommendations**: Based on these results, SVC, Logistic Regression, Multinomial Naive Bayes, and Random Forest are recommended for sentiment analysis tasks due to their higher accuracy and consistent performance across different vectorization techniques.

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

---

## Usage

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
   !python Bert_Roberta_Lora_main.py