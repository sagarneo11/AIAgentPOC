# RAG Knowledge Agent - Python Backend

FastAPI backend for the RAG Knowledge Agent application.

## Setup

1. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure Ollama is running:**
   ```bash
   # Make sure Ollama is running on http://localhost:11434
   ollama pull richardyoung/smolvlm2-2.2b-instruct
   ```

4. **Run the server:**
   ```bash
   python main.py
   ```

   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

The API will be available at `http://localhost:8000`

## API Endpoints

### GET `/`
Health check endpoint.

### GET `/api/ingest`
Test endpoint to verify the ingest API is working.

### POST `/api/ingest`
Ingest a document (PDF or URL) into the vector store.

**Form Data:**
- `file`: PDF file (optional)
- `url`: URL string (optional)

**Response:**
```json
{
  "success": true,
  "message": "Document indexed successfully",
  "chunksCount": 10
}
```

### POST `/api/chat`
Chat with the indexed documents using RAG.

**Request Body:**
```json
{
  "messages": [
    {"role": "user", "content": "What is the main topic?"}
  ],
  "query": "What is the main topic?"
}
```

**Response:**
Server-Sent Events (SSE) stream with chunks:
```
data: {"content": "The main topic is..."}
```

## Vector Store

The vector store is persisted to disk in the `vector_store/` directory using FAISS. It will be automatically created on first use and loaded on subsequent server restarts.
