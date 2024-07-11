#!/usr/bin/env python
# coding: utf-8

# In[63]:



# from langchain.llms import GooglePalm
import os
import pandas as pd
from dotenv import load_dotenv
from langchain import HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


processed_file_path = "../Data/Process/sample_test_data.xlsx"
df_test = pd.read_excel(processed_file_path)

# Load environment variables from .env file
load_dotenv()



# Access the API key
# google_palm_api_key = os.getenv('Google_palm_api_key')
# llm = GooglePalm(google_api_key=api_key, temperature=0.4)
#
#
# # classification_chain = load_qa_chain(llm=llm)
# def get_sentiment(text):
#     print(text)
#     # Define the prompt template
#     prompt_template = """Given the review, ask as the NLP model to determine the sentiment of the below review: "{text}"."""
#
#     # Generate the complete prompt by inserting the review text
#     prompt = prompt_template.format(text=text)
#
#     # Pass the prompt to the LLM
#     classification = llm(prompt)
#
#     return classification

# df_test['pred_review'] = df_test['review'].apply(lambda x: get_sentiment(x))


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

# Apply the classification function to the DataFrame
df_test['predicted_sentiment'] = df_test['review'].apply(classify_review)


accuracy = df_test[df_test['sentiment']==df_test['predicted_sentiment']].shape[0]/df_test.shape[0]

print(accuracy)