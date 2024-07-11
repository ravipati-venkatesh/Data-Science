import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, TFBertForSequenceClassification, RobertaTokenizer, TFRobertaForSequenceClassification
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
        max_length=512,
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
    epochs=2,
    batch_size=256,
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
    epochs=2,
    batch_size=256,
    validation_data=(dict(test_encodings), test_labels)
)



 # Define a function to calculate additional evaluation metrics
def evaluate_model(model, test_encodings, test_labels):
    y_pred_logits = model.predict(dict(test_encodings)).logits
    y_pred = tf.argmax(y_pred_logits, axis=1).numpy()
    y_true = test_labels.numpy()

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return accuracy, precision, recall, f1


# Evaluate BERT model
bert_accuracy, bert_precision, bert_recall, bert_f1 = evaluate_model(bert_model, test_encodings, test_labels)
print(f"BERT - Accuracy: {bert_accuracy}, Precision: {bert_precision}, Recall: {bert_recall}, F1 Score: {bert_f1}")

# Evaluate RoBERTa model
roberta_accuracy, roberta_precision, roberta_recall, roberta_f1 = evaluate_model(roberta_model, test_encodings,
                                                                                 test_labels)
print(
    f"RoBERTa - Accuracy: {roberta_accuracy}, Precision: {roberta_precision}, Recall: {roberta_recall}, F1 Score: {roberta_f1}")

