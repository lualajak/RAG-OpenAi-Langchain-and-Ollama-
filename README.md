
===========================
## RAG-WEB: Retrieval-Augmented Generation with LangChain + Ollama + Deepseek
===========================

This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using:
- A public document from the South Sudan government
- Chroma for vector storage
- Ollama for embeddings (nomic-embed-text)
- DeepSeek chat model for reasoning
- LangChain for chaining prompts and retrieval

---------------------------
### Features
---------------------------
- Loads and chunks a webpage into smaller documents
- Generates embeddings using Ollama
- Stores vectors in a local Chroma DB
- Retrieves relevant chunks using MultiQueryRetriever
- Uses an LLM to answer questions based on context

---------------------------
### Project Structure
---------------------------
rag-web/
├── .env               # Your API keys and base URLs
├── rag.py             # Main script
├── venv/              # (Optional) Python virtual environment
├── requirements.txt   

---------------------------
### Requirements
---------------------------
- Python 3.8+
- Ollama running locally
- Chroma DB
- LangChain

### Install dependencies:

    pip install -r requirements.txt

(Generate requirements.txt with: pip freeze > requirements.txt)

---------------------------

### .env File
---------------------------
Create a .env file in the root with:

    OPENAI_API_KEY=your-openrouter-api-key
    BASE_URL=http://localhost:11434  # or your Ollama base URL

---------------------------
### How to Run
---------------------------

    python rag.py

---------------------------

### Example Prompt
---------------------------
    Summarise the document

### The system will:
1. Embed the document
2. Retrieve relevant chunks
3. Ask the LLM to generate a contextual answer

---------------------------
### Source Document
---------------------------
South Sudan Constitution & Government:  
https://mojca.gov.ss/constitution-and-government #or anyother document you want to use

---------------------------
### Models Used
---------------------------
- Embedding: nomic-embed-text:latest
- LLM: deepseek/deepseek-chat:free (via OpenRouter)

---------------------------

