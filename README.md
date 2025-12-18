# Chat with PDF using Multimodal RAG

This project implements a multimodal RAG (Retrieval-Augmented Generation) pipeline using LangChain and Streamlit. It allows you to chat with complex PDF documents containing text and images, leveraging the multimodal capabilities of GPT-4 with vision.

## Code Overview (src folder)

The `src` folder contains two main files:

-   `app.py`: This is the Streamlit application that provides the user interface for interacting with the RAG pipeline. It handles PDF uploads, displays chat messages, and sends user queries to the `RAGPipeline`.
-   `rag_pipeline.py`: This file defines the `RAGPipeline` class, which encapsulates the core logic for processing PDFs, embedding text and images, creating a vector store, and answering questions using a large language model (LLM). It utilizes asynchronous operations for efficient PDF processing.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/chat-with-pdf.git
    cd chat-with-pdf
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Configurations

1.  **OpenAI API Key:**
    Obtain an OpenAI API key from the [OpenAI website](https://platform.openai.com/account/api-keys).

2.  **Environment File:**
    Create a `.env` file in the root of the project directory and add your OpenAI API key:
    ```
    OPENAI_API_KEY="your_openai_api_key_here"
    ```
    Replace `"your_openai_api_key_here"` with your actual OpenAI API key.

## Usage

1.  **Run the Streamlit application:**
    ```bash
    python -m streamlit run src/app.py
    ```

2.  **Interact with the application:**
    Open your web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).
    
    -   Upload a PDF file using the sidebar.
    -   Once the PDF is processed, you can ask questions about its content in the chat interface.