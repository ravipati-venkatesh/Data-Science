import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification, RobertaTokenizer, TFRobertaForSequenceClassification
import tensorflow as tf

# Load your dataset
processed_file_path = "../Data/Process/sample_preprocessed_data.xlsx"
df = pd.read_excel(processed_file_path)

# Preprocess data
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Use BERT tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_data(df, tokenizer):
    return tokenizer(
        df['review'].tolist(),
        max_length=5000,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

train_encodings = tokenize_data(train_df, bert_tokenizer)
test_encodings = tokenize_data(test_df, bert_tokenizer)

# Convert encodings to TensorFlow datasets
train_labels = tf.convert_to_tensor(train_df['sentiment'].values)
test_labels = tf.convert_to_tensor(test_df['sentiment'].values)

# Load BERT model
bert_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased')

# Define training arguments
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

bert_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Fine-tune the model
bert_model.fit(
    x=dict(train_encodings),
    y=train_labels,
    epochs=3,
    batch_size=16,
    validation_data=(dict(test_encodings), test_labels)
)





# Use RoBERTa tokenizer
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

train_encodings = tokenize_data(train_df, roberta_tokenizer)
test_encodings = tokenize_data(test_df, roberta_tokenizer)

# Convert encodings to TensorFlow datasets
train_labels = tf.convert_to_tensor(train_df['sentiment'].values)
test_labels = tf.convert_to_tensor(test_df['sentiment'].values)

# Load RoBERTa model
roberta_model = TFRobertaForSequenceClassification.from_pretrained('roberta-base')

# Define training arguments
roberta_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Fine-tune the model
roberta_model.fit(
    x=dict(train_encodings),
    y=train_labels,
    epochs=3,
    batch_size=16,
    validation_data=(dict(test_encodings), test_labels)
)



# Evaluate BERT model
bert_results = bert_model.evaluate(dict(test_encodings), test_labels)
print(f"BERT - Loss: {bert_results[0]}, Accuracy: {bert_results[1]}")

# Evaluate RoBERTa model
roberta_results = roberta_model.evaluate(dict(test_encodings), test_labels)
print(f"RoBERTa - Loss: {roberta_results[0]}, Accuracy: {roberta_results[1]}")


