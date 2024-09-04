## Question Answering Application

This application processes articles from URLs, splits the text into chunks, generates embeddings, creates a vector database, and answers user questions based on the provided context using a language model.

#### Features

- **Load Data**: Fetches and processes data from provided URLs.
- **Text Splitting**: Splits the text into manageable chunks for better processing.
- **Vector Database Creation**: Creates a FAISS vector database from the processed text.
- **LLM Model**: Uses a pre-trained language model (Flan-T5) from HuggingFace for generating answers.
- **Question Answering**: Allows users to input questions and get answers based on the processed data.

#### Usage

To run any of the implementations, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/ravipati-venkatesh/Data-Science.git
    ```
   
2. Set up your environment variables:

    Create a `.env` file in the project directory and add your HuggingFace API token:

    ```plaintext
    HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token
    ```

3. Running the App

    ```bash
    cd Data-Science/Projects/QA_based_on_articles
    streamlit run Main.py
    ```

4. Open your web browser and go to `http://localhost:8501`.

#### Example URLs

- https://www.ibm.com/topics/large-language-models
- https://www.cloudflare.com/en-in/learning/ai/what-is-large-language-model/
- https://www.elastic.co/what-is/large-language-models
- https://en.wikipedia.org/wiki/Large_language_model

#### Acknowledgements

- **LangChain**: A framework for building applications with language models. LangChain simplifies the process of working with various language models and integrating them into applications.

- **Hugging Face**: Provides a vast collection of pre-trained language models and tools for natural language processing. Their models and libraries are fundamental to the implementation of state-of-the-art NLP solutions.

- **Streamlit**: An open-source app framework for Machine Learning and Data Science projects. Streamlit allows for the rapid creation of interactive web applications to visualize and share results.

- **SentenceTransformers**: Offers pre-trained models and tools for generating sentence embeddings. These embeddings are useful for various NLP tasks, including semantic textual similarity and clustering.

- **FAISS**: A library for efficient similarity search and clustering of dense vectors. FAISS enables fast and scalable retrieval of similar items in large datasets.

