import streamlit as st      # Library for web application
from langchain_community.document_loaders import PyPDFLoader    # Tools to load PDF files and extract text
from langchain.text_splitter import RecursiveCharacterTextSplitter  # A tool to split long text into smaller chunks
from langchain_community.embeddings import OllamaEmbeddings     # Generate text embeddings using Ollama
from langchain_community.vectorstores import Chroma     # Vector database storage
from langchain_community.llms import Ollama         # Ollama LLM interface
from langchain.chains import RetrievalQA        # Implementing a question-answer chain
import tempfile     # Temporary file handling
import os           # File system operations
import time         # Added module for time measurement

# Streamlit page basics
st.set_page_config(
        page_title="AI Paper Analyst",
        layout="wide"
)

# Set the main page title
st.title("AI Paper Analyst with deepseek-r1_1.5b")

# Create a PDF file upload widget 
uploaded_file = st.file_uploader("Upload Paper PDF File.", type=['pdf'])

# Logic to be executed when the file is uploaded
if uploaded_file is not None:
    # Save uploaded files as temporary files
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        temp_path = tmp_file.name  # Save the temporary file path

    try:
        # Load the PDF file and extract text
        loader = PyPDFLoader(temp_path)
        pages = loader.load()  # Load each page of the PDF

        # Split the extracted text into small chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Maximum number of characters in each chunk
            chunk_overlap=200  # Number of characters that overlap between
        )
        splits = text_splitter.split_documents(pages)

        # Set up text embeddings
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text",  # Model to use for embeddings
            base_url="http://localhost:11434"  # Ollama server URL
        )
        
        # Set up vector store and save documents
        vectorstore = Chroma.from_documents(
            documents=splits,  # Split documents
            embedding=embeddings,  # Embedding function
            persist_directory="./.chroma"  # Location of vector store
        )

        # Set up LLM 
        llm = Ollama(
            model="deepseek-r1:1.5b",  # LLM model to use
            temperature=0,  # Creativity of the response (0: most deterministic)
            base_url="http://localhost:11434"  # Ollama server URL
        )

        # Create a question-answer chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,  # LLM to use
            chain_type="stuff",  # QA chain type
            retriever=vectorstore.as_retriever()  # Document searcher
        )

        # Get user question
        user_question = st.text_input("Ask a question about the paper:")
        
        # Execute when question is entered
        if user_question:
            # Record start time
            start_time = time.time()
            
            with st.spinner('Generating answer...'):  # Loading indicator
                # Progress indicator
                progress_placeholder = st.empty()
                progress_placeholder.text("Creating an answer...")
                
                # Generate an answer to a question
                response = qa_chain.invoke({"query": user_question})
                
                # Record the end time and calculate the time taken
                end_time = time.time()
                elapsed_time = round(end_time - start_time, 2)
                
                # Remove progress text
                progress_placeholder.empty()
                
                # Display the answer and time taken
                st.write("Answer:")
                st.write(response['result'])
                st.info(f"Time taken to generate an answer: {elapsed_time} seconds")

    finally:
        # Delete temporary file after processing is complete
        os.unlink(temp_path)

# Add usage instructions to the sidebar
with st.sidebar:
    st.header("How to use")
    st.write("""
    1. Upload a PDF paper file.
    2. Ask questions about the paper. "ex. explain about xxx using uploaded paper.."
    3. AI analyzes the paper and provices answers.
    """)
