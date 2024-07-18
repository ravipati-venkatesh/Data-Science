#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import time
from dotenv import load_dotenv
from langchain import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Access the Hugging Face API key
huggingface_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')

# Initialize the LLM with Hugging Face
flan_t5_llm = HuggingFaceHub(
    huggingfacehub_api_token=huggingface_api_key,
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.1},
    task="text2text-generation"
)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["text"],
    template="Given the review, determine the sentiment of the below review: {text}"
)

# Create the chain
chain = LLMChain(
    llm=flan_t5_llm,
    prompt=prompt_template
)

# Function to classify sentiment
def classify_review(review_):
    result = chain.run(text=review_)
    return result.strip().lower()

# File paths
processed_file_path = "../Data/Process/sample_test_data.xlsx"
batch_status_file_path = "../Data/Process/batch_status.xlsx"

# Load test data
df_test = pd.read_excel(processed_file_path)

# Load or create the batch status file
if os.path.exists(batch_status_file_path):
    df_status = pd.read_excel(batch_status_file_path)
else:
    df_status = pd.DataFrame(columns=["batch_start", "batch_end", "status", "accuracy"])

accuracy_list = []

# Process each batch
for i in range(0, len(df_test), 50):
    batch_start = i
    batch_end = min(i + 50, len(df_test))
    
    # Check if the batch has already been processed
    if not ((df_status['batch_start'] == batch_start) & (df_status['batch_end'] == batch_end)).any():
        batch_df = df_test.iloc[batch_start:batch_end]
        batch_df['predicted_sentiment'] = batch_df['review'].apply(classify_review)
        
        batch_accuracy = batch_df[batch_df['sentiment'] == batch_df['predicted_sentiment']].shape[0] / batch_df.shape[0]
        accuracy_list.append(batch_accuracy)
        
        print(f"Batch {batch_start//50 + 1} accuracy: {batch_accuracy}")
        
        # Create a new DataFrame for the current batch status
        batch_status_df = pd.DataFrame({
            "batch_start": [batch_start],
            "batch_end": [batch_end],
            "status": ["processed"],
            "accuracy": [batch_accuracy]
        })
        
        # Concatenate the new batch status DataFrame with the existing status DataFrame
        df_status = pd.concat([df_status, batch_status_df], ignore_index=True)
        
        # Pause for a few seconds to avoid HTTP limit errors
        time.sleep(2)

        # Save the updated status DataFrame
        df_status.to_excel(batch_status_file_path, index=False)

# Calculate overall accuracy
if accuracy_list:
    overall_accuracy = sum(accuracy_list) / len(accuracy_list)
    print(f"Overall accuracy: {overall_accuracy}")
else:
    print("No batches processed.")