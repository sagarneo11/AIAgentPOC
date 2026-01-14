# RAG Knowledge Agent

A Proof of Concept (POC) application that demonstrates Retrieval-Augmented Generation (RAG) capabilities using Next.js frontend, Python FastAPI backend, LangChain, and Ollama.

## Architecture

- **Frontend**: Next.js 14 (App Router) with React and Tailwind CSS
- **Backend**: Python FastAPI server
- **AI Logic**: LangChain (Python)
- **LLM Provider**: Ollama (local)
- **Vector Store**: FAISS (persisted to disk)

## Prerequisites

1. **Node.js** (v18 or higher)
2. **Python 3.9+**
3. **Ollama** installed and running locally on port 11434
4. Required Ollama models:
   - `richardyoung/smolvlm2-2.2b-instruct` (for chat and embeddings)

### Installing Ollama Models

```bash
# Install Ollama if you haven't already
# Visit https://ollama.ai for installation instructions

# Pull required model
ollama pull richardyoung/smolvlm2-2.2b-instruct
```

## Setup

### Backend Setup

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the backend server:
   ```bash
   python main.py
   ```
   
   Or using uvicorn directly:
   ```bash
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

   The API will be available at `http://localhost:8000`

### Frontend Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Start the development server:
   ```bash
   npm run dev
   ```

3. Open [http://localhost:3000](http://localhost:3000) in your browser

## Usage

### Indexing Documents

1. **Via URL**: Enter a URL in the left sidebar and click "Index URL"
2. **Via PDF**: Click the upload area in the left sidebar and select a PDF file

Once indexed, you'll see a success message indicating the document has been added to the knowledge base.

### Chatting with Documents

1. After indexing at least one document, use the chat interface on the right
2. Type your question and press Enter or click the Send button
3. The AI will retrieve relevant context from your indexed documents and provide an answer

## Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI application
│   ├── services/
│   │   ├── ingest_service.py # Document ingestion logic
│   │   └── chat_service.py   # RAG chat logic
│   ├── requirements.txt      # Python dependencies
│   └── vector_store/         # Persisted vector store (created automatically)
├── app/                      # Next.js frontend
│   ├── api/                  # (Not used - backend handles API)
│   ├── components/           # React components
│   └── page.tsx              # Main page
└── components/               # Shared components
```

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
    { "role": "user", "content": "What is the main topic?" }
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

The vector store is persisted to disk in the `backend/vector_store/` directory using FAISS. It will be automatically created on first use and loaded on subsequent server restarts.

## Troubleshooting

### Backend Issues

- **Ollama Connection**: Ensure Ollama is running: `ollama list`
- **Model Not Found**: Pull the required models: `ollama pull richardyoung/smolvlm2-2.2b-instruct`
- **Port Already in Use**: Change the port in `main.py` or stop the conflicting service

### Frontend Issues

- **CORS Errors**: Ensure the backend CORS settings allow `http://localhost:3000`
- **Connection Refused**: Make sure the backend is running on port 8000

### Vector Store Issues

- If the vector store becomes corrupted, delete the `backend/vector_store/` directory and re-index documents

## License

This is a POC demonstration project.
