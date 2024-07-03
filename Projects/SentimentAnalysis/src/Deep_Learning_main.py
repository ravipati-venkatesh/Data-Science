#!/usr/bin/env python
# coding: utf-8

# In[65]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, RNN, Dense, BatchNormalization, SimpleRNN, Dropout
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.utils import to_categorical

def preprocess_data(df):
    # label_encoder = LabelEncoder()
    # df['sentiment_encoded'] = label_encoder.fit_transform(df['sentiment'])
        # Convert sentiment labels to categorical encoding
    df['sentiment'] = pd.Categorical(df['sentiment'])
    df['sentiment_encoded'] = df['sentiment'].cat.codes
    y = to_categorical(df['sentiment_encoded'])
    X_train, X_test, y_train, y_test = train_test_split(df['review'], y, test_size=0.2, random_state=42)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train.values)
    X_train_seq = tokenizer.texts_to_sequences(X_train.values)
    X_test_seq = tokenizer.texts_to_sequences(X_test.values)
    maxlen = max([len(sublist) for sublist in X_train_seq])
    # Pad the sequences to the same length
    X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
    X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)
    vocab_size = max([item for sublist in X_train_seq for item in sublist])
    return X_train_pad, X_test_pad, y_train, y_test, vocab_size


def model_def_and_compile(model_name, vocab_size):
    
    # Define the model
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size+1, output_dim=100))
    model.add(BatchNormalization())
    
    if model_name == "LSTM":
        model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(LSTM(100, dropout=0.3, recurrent_dropout=0.3, return_sequences=False))
    elif model_name == "GRU":
        model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(GRU(100, dropout=0.3, recurrent_dropout=0.3, return_sequences=False))
    elif model_name == "RNN":
        model.add(Conv1D(filters=64, kernel_size=5, activation='relu'))
        model.add(MaxPooling1D(pool_size=4))
        model.add(SimpleRNN(100, dropout=0.3, recurrent_dropout=0.3, return_sequences=False))
        
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    model.add(Dense(2, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


def predict_sentiment(model, X_test_pad, y_test):

    y_pred = model.predict(X_test_pad)
    y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)
    
    return accuracy, precision, recall, f1


model_names = ['LSTM', 'RNN', 'GRU']
if __name__ == '__main__':
    
    # preprocessing  Data path
    processed_file_path = "../Data/Process/sample_preprocessed_data.xlsx"
    
    # Load preprocessed data
    df = pd.read_excel(processed_file_path)
    X_train_pad, X_test_pad, y_train, y_test, vocab_size = preprocess_data(df)
    
    results = []
    for model_name in model_names:
        model = model_def_and_compile(model_name, vocab_size)
        # Train the model
        model.fit(X_train_pad, y_train, epochs=5, batch_size=200, validation_data=(X_test_pad, y_test))

        accuracy, precision, recall, f1  = predict_sentiment(model, X_test_pad, y_test)
        print(model_name, accuracy, precision, recall, f1)
        results.append([model_name, accuracy, precision, recall, f1])
        
    df_summary = pd.DataFrame(results, columns=['model_name', 'accuracy', 'precision', 'recall', 'f1'])
    summary_file_path = f'../Data/Process/deep_learning.xlsx'
    df_summary.to_excel(summary_file_path, index=False)

