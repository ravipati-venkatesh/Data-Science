#!/usr/bin/env python
# coding: utf-8

# # Loaders

from langchain import HuggingFaceHub
from langchain.document_loaders import UnstructuredURLLoader
from sentence_transformers import SentenceTransformer
import pickle
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import langchain
import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Data Loading
def load_data(urls):
    loader = UnstructuredURLLoader(urls=["https://www.ibm.com/topics/large-language-models",
                                         "https://www.cloudflare.com/en-in/learning/ai/what-is-large-language-model/",
                                         "https://www.elastic.co/what-is/large-language-models",
                                         "https://en.wikipedia.org/wiki/Large_language_model"
                                         ])
    data = loader.load()
    return data

# # Text splitters
def split_data(data):

    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "],
                                                   chunk_size=500,
                                                   chunk_overlap=200
                                                   )

    chunks = text_splitter.split_documents(data)
    return chunks

# # Vector Database
def create_vectordatabase(chunks):
    # Load a pre-trained Sentence-BERT model
    model = SentenceTransformer('all-roberta-large-v1')
    embeddings = model.encode([chunk.page_content for chunk in chunks])

    # Create FAISS vector store from documents and embeddings

    text_embedding_pairs = zip([chunk.page_content for chunk in chunks], embeddings)
    vectorstore = FAISS.from_embeddings(text_embedding_pairs, model)

    file_path = "vector_index.pkl"
    with open(file_path, 'wb') as f:
        pickle.dump(vectorstore, f)
    return vectorstore

def get_context(query_text, embedding_model, vectorstore, chunks):
    # Encode the query text
    query_text = "What is LLM, what are its uses?"
    query_vector = embedding_model.encode([query_text])

    # Perform the search
    k = 5  # Number of nearest neighbors to retrieve
    distances, indices = vectorstore.index.search(query_vector, k)

    docs = []
    # Retrieve and print the results
    for i in range(k):
        doc_index = indices[0][i]
        distance = distances[0][i]
        #         print(f"Result {i+1}:")
        #         print(f"Text: {chunks[doc_index].page_content}")
        #         print(f"Distance: {distance}")
        docs.append(chunks[doc_index].page_content)
    return ". ".join(docs)

def llm_chain():
    # Define a simple prompt template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are an AI assistant tasked with answering questions based on the given context. 
        Provide a concise and accurate answer to the question asked.
    
        Context:
        {context}
    
        Question:
        {question}
    
        Answer:
        """
    )

    # Load environment variables from .env file
    load_dotenv()

    # Access the Hugging Face API key
    huggingface_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')

    llm = HuggingFaceHub(huggingfacehub_api_token=huggingface_api_key,
                         repo_id="google/flan-t5-large",
                         model_kwargs={"temperature": 0.8, "max_length": 500},
                         task="text2text-generation"
                         )


    # Create an LLM chain with the prompt template and the LLM
    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    return llm_chain

def get_answer(llm_chain, query_text, model, vectorstore, chunks):
    context = get_context(query_text, model, vectorstore, chunks)
    response = llm_chain.invoke({"context": context, "question": query_text})

    return response["text"]




