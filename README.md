# AI Ingestion & Search (RAG System)

This project implements a Retrieval-Augmented Generation (RAG) system using Python, LangChain, and PostgreSQL with the `pgvector` extension. It allows you to ingest PDF documents, split them into chunks, generate embeddings, and store them in a vector database for semantic search and interactive chat.

## Project Structure

The source code is located in the `src/` directory:

- `src/ingest.py`: Main script for loading PDFs, processing text, and ingesting chunks into the vector store.
- `src/chat.py`: Interactive CLI tool to chat with your documents using semantic search results.
- `src/search.py`: Contains the logic for similarity search and context retrieval.
- `src/factory.py`: Centralized factory for initializing AI models (embeddings/chat) and the database connection.
- `src/os_util.py`: Utility for environment variable handling and validation.

## Prerequisites

- **Docker & Docker Compose**: To run the PostgreSQL/pgvector database.
- **Python 3.10+**: The project is built with Python.
- **Provider API Key**: Required for AI models (e.g., Google Gemini or OpenAI).

## Setup Instructions

### 1. Database Setup
First, start the PostgreSQL database with `pgvector` using Docker Compose:

```bash
docker-compose up -d
```

### 2. Python Environment (venv)
It is highly recommended to use a virtual environment to manage dependencies.

**Create the virtual environment:**
```bash
python3 -m venv venv
```

**Activate the virtual environment:**
- **On Linux/macOS:**
  ```bash
  source venv/bin/activate
  ```
- **On Windows:**
  ```bash
  .\venv\Scripts\activate
  ```

### 3. Install Dependencies
With the virtual environment active, install the required packages:

```bash
pip install -r requirements.txt
```

### 4. Configuration
Create a `.env` file in the root directory (you can copy `.env.example` as a starting point):

```bash
cp .env.example .env
```

Edit the `.env` file and provide your **API Keys**. You can configure the `LLM_PROVIDER` (e.g., `google` or `openai`) and other settings like database connection strings and PDF paths.

## Environment Variables

The project uses the following environment variables defined in the `.env` file:

### Provider Configuration
- `LLM_PROVIDER`: The AI provider to use (`google`, `openai`, or `openai-openrouter`).

### Google Gemini Configuration
- `GOOGLE_API_KEY`: Your Google AI API key.
- `GOOGLE_LLM_MODEL`: The Gemini model for chat (e.g., `gemini-1.5-flash`).
- `GOOGLE_EMBEDDING_MODEL`: The model for generating embeddings (e.g., `models/embedding-001`).

### OpenAI Configuration
- `OPENAI_API_KEY`: Your OpenAI API key.
- `OPENAI_LLM_MODEL`: The GPT model for chat (e.g., `gpt-4o-mini`).
- `OPENAI_EMBEDDING_MODEL`: The model for generating embeddings (e.g., `text-embedding-3-small`).

### OpenRouter Configuration
- `OPENAI_API_KEY_EMBEDDING_FROM_OPENROUTER_KEY`: Your OpenRouter API key.
- `OPENROUTER_URL`: The OpenRouter API base URL (usually `https://openrouter.ai/api/v1`).
- `OPENAI_LLM_MODEL`: The model name on OpenRouter (e.g., `openai/gpt-4o-mini`).
- `OPENAI_EMBEDDING_MODEL`: The embedding model name on OpenRouter (Note: ensure the model supports OpenAI-compatible embeddings).

### Database & Storage
- `DATABASE_URL`: Connection string for PostgreSQL (e.g., `postgresql+psycopg://postgres:postgres@localhost:5432/rag`).
- `PG_VECTOR_COLLECTION_NAME`: The name of the collection in the vector database.
- `PDF_PATH`: The relative or absolute path to the PDF file to be ingested.

---

## How to Use

### Step 1: Ingest Documents
Place your PDF files in the directory specified by `PDF_PATH` in your `.env` (default is the project root). Then, run the ingestion script:

```bash
python src/ingest.py
```
This script will read the PDF, split it into chunks, and store the embeddings in Postgres.

### Step 2: Interactive Chat
Once the data is ingested, you can start asking questions about your documents:

```bash
python src/chat.py
```
Type your message and press **Enter** to get an AI-generated answer based on your documents. Type `exit` or `quit` to leave the chat.