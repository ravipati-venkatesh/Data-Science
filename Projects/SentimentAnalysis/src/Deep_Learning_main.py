#!/usr/bin/env python
# coding: utf-8

# In[65]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, RNN, Dense, BatchNormalization, SimpleRNN
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[66]:


def preprocess_data(df):
    label_encoder = LabelEncoder()
    df['sentiment_encoded'] = label_encoder.fit_transform(df['sentiment'])
    X_train, X_test, y_train, y_test = train_test_split(df['review'], df['sentiment_encoded'], test_size=0.2, random_state=42)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train.values)
    X_train_seq = tokenizer.texts_to_sequences(X_train.values)
    X_test_seq = tokenizer.texts_to_sequences(X_test.values)
    maxlen = max([item for sublist in X_train_seq for item in sublist])
    # Pad the sequences to the same length
    X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
    X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)
    return X_train_pad, X_test_pad, y_train, y_test, label_encoder, maxlen


# In[67]:


def model_def_and_compile(model_name, maxlen):
    
    # Define the model
    model = Sequential()
    model.add(Embedding(input_dim=maxlen+1, output_dim=100))
    model.add(BatchNormalization())
    
    if model_name == "LSTM":
        model.add(LSTM(200, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
    elif model_name == "GRU":
        model.add(GRU(200, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
    elif model_name == "RNN":
        model.add(SimpleRNN(200, dropout=0.2, recurrent_dropout=0.2, return_sequences=False))
        
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# In[68]:


def predict_sentiment(model, X_test_pad, y_test):

    y_pred = model.predict(X_test_pad)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    return accuracy, precision, recall, f1


# In[ ]:


model_names = ['LSTM', 'RNN', 'GRU']

if __name__ == '__main__':
    
    # preprocessing  Data path
    processed_file_path = "../Data/Process/sample_preprocessed_data.xlsx"
    
    # Load preprocessed data
    df = pd.read_excel(processed_file_path)
    X_train_pad, X_test_pad, y_train, y_test, label_encoder, maxlen = preprocess_data(df)
    
    results = []
    for model_name in model_names:
        model = model_def_and_compile(model_name, maxlen)
        print(maxlen)
        # Train the model
        model.fit(X_train_pad, y_train, epochs=5, batch_size=32)
        
        accuracy, precision, recall, f1  = predict_sentiment(model, X_test_pad, y_test)
        results.append([model_name, accuracy, precision, recall, f1])
        
    df_summary = pd.DataFrame(results, columns=['model_name', 'accuracy', 'precision', 'recall', 'f1'])
    summary_file_path = f'../Data/Process/deep_learning.xlsx'
    df_summary.to_excel(summary_file_path, index=False)
        
    


# In[ ]:




