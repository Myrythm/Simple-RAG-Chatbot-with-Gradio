# Simple RAG with LangChain and Gradio

This project is a simple Retrieval-Augmented Generation (RAG) application that uses LangChain and Gradio to create a chatbot that can answer questions about a given document.

## Features

- Uses a local vector store (ChromaDB) to store document embeddings.
- Uses Google's Gemini models for embeddings and chat.
- Provides a Gradio web interface for interacting with the chatbot.
- Includes a script for ingesting PDF documents into the vector store.

## Getting Started

### Prerequisites

- Python 3.10
- A Google API key

### Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/Myrythm/Simple-RAG-Chatbot-with-Gradio.git
   cd RAG-INTRO
   ```

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv venv
   ```

   ```bash
    venv\Scripts\activate
   ```

3. **Install the required packages:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your environment variables:**
   Create a `.env` file in the root of the project and add your Google API key:
   ```
   GOOGLE_API_KEY="YOUR_API_KEY"
   ```

## Usage

1. **Add your documents:**
   Place your PDF documents in the `data` directory.

2. **Ingest the documents:**
   Run the ingestion script to process the documents and create the vector store:

   ```bash
   python ingest_database.py
   ```

3. **Run the chatbot:**

   ```bash
   python chatbot.py
   ```

4. **Interact with the chatbot:**
   Open the URL provided in the terminal in your web browser to start chatting with your documents.

## File Descriptions

- `chatbot.py`: The main application file that runs the Gradio chatbot.
- `ingest_database.py`: A script for ingesting PDF documents into the ChromaDB vector store.
- `requirements.txt`: A list of the Python packages required to run the application.
- `.env.example`: An example of the `.env` file.
- `data/`: A directory to store the PDF documents to be ingested.
- `chroma_db/`: A directory where the ChromaDB vector store is saved.
