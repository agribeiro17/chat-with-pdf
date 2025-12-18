import streamlit as st
import os
import asyncio
import tempfile
from rag_pipeline import RAGPipeline

# Set up the Streamlit page
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("Chat with PDF using Multimodal RAG")

# Initialize the RAG pipeline
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()

# Track processed files
if 'processed_file' not in st.session_state:
    st.session_state.processed_file = None

# --- Helper function to run async code in Streamlit ---
def run_async(func):
    return asyncio.run(func)

# Sidebar for PDF upload
with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Check if this is a new file or the same file
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if st.session_state.processed_file != file_id:
            # New file - process it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                tmpfile.write(uploaded_file.getvalue())
                file_path = tmpfile.name
            
            st.success(f"File '{uploaded_file.name}' uploaded successfully!")
            
            # Process the PDF and create the vector store
            with st.spinner("Processing PDF... This may take a few minutes."):
                run_async(st.session_state.rag_pipeline.process_pdf(file_path))
            
            st.success("✓ PDF processed and vector store created!")
            st.info(f"Processed {len(st.session_state.rag_pipeline.all_docs)} document chunks")
            
            # Mark this file as processed
            st.session_state.processed_file = file_id
            
            # Clean up the temporary file
            os.remove(file_path)
        else:
            # Same file already processed
            st.success(f"✓ '{uploaded_file.name}' is ready")
            st.info(f"Using {len(st.session_state.rag_pipeline.all_docs)} document chunks")
    
    # Add a button to clear and reprocess
    if st.session_state.processed_file is not None:
        if st.button("Clear & Upload New PDF"):
            st.session_state.processed_file = None
            st.session_state.rag_pipeline = RAGPipeline()
            st.session_state.messages = []
            st.rerun()

# Main chat interface
st.header("Chat")

# Check if a PDF has been processed
if st.session_state.processed_file is None:
    st.info("👈 Please upload a PDF file to start chatting")
else:
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question about the PDF"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get response from the RAG pipeline (queries vector store only)
        with st.spinner("Searching document..."):
            response = run_async(st.session_state.rag_pipeline.ask_question(prompt))
            
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})