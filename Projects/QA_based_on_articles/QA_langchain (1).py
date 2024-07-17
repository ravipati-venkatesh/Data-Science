#!/usr/bin/env python
# coding: utf-8

# # Loaders

# In[227]:


from langchain.document_loaders import UnstructuredURLLoader
loader = UnstructuredURLLoader(urls=["https://www.ibm.com/topics/large-language-models",
                                    "https://www.cloudflare.com/en-in/learning/ai/what-is-large-language-model/",
                                    "https://www.elastic.co/what-is/large-language-models",
                                    "https://en.wikipedia.org/wiki/Large_language_model"
                                    ])
data = loader.load()


# In[228]:


len(data)


# # Text splitters

# In[229]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " "],
                                            chunk_size=500,
                                            chunk_overlap=200
)

chunks = text_splitter.split_documents(data)


# In[230]:


len(chunks)


# # Vector Database

# In[231]:


from sentence_transformers import SentenceTransformer
import pickle
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
# Load a pre-trained Sentence-BERT model
model = SentenceTransformer('all-roberta-large-v1')
embeddings = model.encode([chunk.page_content for chunk in chunks])

# Create FAISS vector store from documents and embeddings

text_embedding_pairs = zip([chunk.page_content for chunk in chunks], embeddings)
vectorstore = FAISS.from_embeddings(text_embedding_pairs, model)
    
file_path = "vector_index.pkl"
with open(file_path, 'wb') as f:
    pickle.dump(vectorstore, f)


# In[ ]:


from dotenv import load_dotenv
import langchain
import os

# Load environment variables from .env file
load_dotenv()


# In[ ]:


# Access the Hugging Face API key
huggingface_api_key = os.getenv('HUGGINGFACEHUB_API_TOKEN')


# In[ ]:


llm = HuggingFaceHub(huggingfacehub_api_token=huggingface_api_key,
                     repo_id="google/flan-t5-large",
                     model_kwargs={"temperature": 0.8, "max_length": 500},
                     task="text2text-generation"
                    )


# In[ ]:


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


# In[ ]:


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

# Create an LLM chain with the prompt template and the LLM
llm_chain = LLMChain(prompt=prompt_template, llm=llm)

query_text = "What is LLM, what are its uses?"
context = get_context(query_text, model, vectorstore, chunks)
response = llm_chain.invoke({"context": context, "question": query_text})

# Print the response
print("Answer:", response['text'])


# In[ ]:


query_text = "How LLM Model performance can be increased"
context = get_context(query_text, model, vectorstore, chunks)
response = llm_chain.invoke({"context": context, "question": query_text})

# Print the response
print("Answer:", response['text'])


# In[ ]:


response


# In[ ]:


query_text = "when was transformer architecture invented?"
context = get_context(query_text, model, vectorstore, chunks)
response = llm_chain.invoke({"context": context, "question": query_text})

# Print the response
print("Answer:", response)


# In[ ]:




