import streamlit as st
import time
from QA_langchain import load_data, split_data, create_vectordatabase, get_context, llm_chain, get_answer


# Define function to simulate fetching data and answering question
def fetch_and_answer(url, question):
    time.sleep(2)  # Simulate network delay
    return f"Answer from {url}: {question}"


def display_progress(message, progress):
    st.write(message)
    for i in range(1, 101):
        time.sleep(0.01)
        progress.progress(i)


# Streamlit layout
st.title("URL Input and Question Answering App")

# Sidebar for URL inputs
st.sidebar.header("Enter URLs")
url1 = st.sidebar.text_input("Enter URL 1")
url2 = st.sidebar.text_input("Enter URL 2")
url3 = st.sidebar.text_input("Enter URL 3")

# URL submission and processing
urls = []
if st.sidebar.button("Submit URLs"):
    if url1:
        urls.append(url1)
    if url2:
        urls.append(url2)
    if url3:
        urls.append(url3)
    if urls:
        st.write("URLs submitted successfully!")
    else:
        st.sidebar.write("No URLs were entered")

# Load and process URL data if URLs are submitted
if urls:
    progress_bar = st.progress(0)

    display_progress("Loading URL data...", progress_bar)
    data = load_data(urls)

    display_progress("Splitting data into chunks...", progress_bar)
    chunks = split_data(data)

    display_progress("Creating Vector Database...", progress_bar)
    vectorstore = create_vectordatabase(chunks)

    display_progress("Creating LLM Model...", progress_bar)
    llm_chain = llm_chain()

    st.success("Data and model preparation completed!")

    # Main area for question input and displaying results
    st.header("Ask a Question")

    question = st.text_input("Enter your question here")

    if st.button("Submit"):
        if question:
            st.write("Fetching answer...")
            answer = get_answer(llm_chain, question, vectorstore, chunks)

            # Display answers
            st.subheader("Results")
            st.write(answer)
        else:
            st.write("Please enter the question.")
else:
    st.write("Please submit URLs first.")
