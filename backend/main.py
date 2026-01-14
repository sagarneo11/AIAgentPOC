from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import os
from dotenv import load_dotenv

from services.ingest_service import IngestService
from services.chat_service import ChatService
from services.document_tracker import DocumentTracker
from services.fine_tuning_service import FineTuningService

load_dotenv()

app = FastAPI(title="RAG Knowledge Agent API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize document tracker first (shared instance)
document_tracker = DocumentTracker()

# Initialize services with shared tracker
ingest_service = IngestService(document_tracker)
# Check if fine-tuned model exists, otherwise use base model
fine_tuned_model = os.getenv("FINE_TUNED_MODEL", "richardyoung/smolvlm2-2.2b-instruct")
chat_service = ChatService(model_name=fine_tuned_model)
fine_tuning_service = FineTuningService(document_tracker, ingest_service)


class ChatRequest(BaseModel):
    messages: List[dict]
    query: str
    doc_id: Optional[int] = None
    fine_tuned_model: Optional[str] = None


@app.get("/")
async def root():
    return {"message": "RAG Knowledge Agent API is running"}


@app.get("/api/ingest")
async def test_ingest():
    return {"message": "Ingest API is working"}


@app.get("/api/documents")
async def list_documents():
    """
    Get list of all indexed documents.
    """
    try:
        documents = document_tracker.get_documents()
        return {
            "success": True,
            "documents": documents,
            "total_documents": document_tracker.get_document_count(),
            "total_chunks": document_tracker.get_total_chunks()
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get documents: {str(e)}"
        )


@app.post("/api/fine-tune/prepare")
async def prepare_training_data(format: str = "alpaca", doc_id: Optional[int] = None):
    """
    Prepare training data from indexed documents.
    If doc_id is provided, prepares data for that specific document.
    """
    try:
        if document_tracker.get_document_count() == 0:
            raise HTTPException(
                status_code=400,
                detail="No documents indexed. Please index documents first."
            )
        
        if doc_id:
            output_file = fine_tuning_service.prepare_training_data_for_document(doc_id, format=format)
        else:
            output_file = fine_tuning_service.prepare_training_data(format=format)
        
        return {
            "success": True,
            "message": "Training data prepared successfully",
            "output_file": output_file,
            "format": format,
            "doc_id": doc_id
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to prepare training data: {str(e)}"
        )


@app.post("/api/fine-tune/create-modelfile")
async def create_modelfile(
    base_model: str = "phi3:mini",
    doc_id: Optional[int] = None
):
    """
    Create Ollama Modelfile for custom model configuration.
    If doc_id is provided, creates a model for that specific document.
    """
    try:
        result = fine_tuning_service.create_modelfile(base_model=base_model, doc_id=doc_id)
        return {
            "success": True,
            "message": "Modelfile created successfully",
            **result
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create Modelfile: {str(e)}"
        )


@app.get("/api/fine-tune/status")
async def get_training_status():
    """
    Get fine-tuning status.
    """
    try:
        status = fine_tuning_service.get_training_status()
        return {
            "success": True,
            **status
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get training status: {str(e)}"
        )


@app.get("/api/fine-tune/models")
async def get_fine_tuned_models():
    """
    Get all fine-tuned models mapped by document ID.
    """
    try:
        models = fine_tuning_service.get_fine_tuned_models()
        return {
            "success": True,
            "models": models
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get fine-tuned models: {str(e)}"
        )


@app.get("/api/fine-tune/training-data")
async def list_training_data():
    """
    List all training data files.
    """
    try:
        files = fine_tuning_service.list_training_data_files()
        return {
            "success": True,
            "files": files
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list training data: {str(e)}"
        )


@app.delete("/api/fine-tune/training-data/{filename}")
async def delete_training_data_file(filename: str):
    """
    Delete a specific training data file.
    """
    try:
        fine_tuning_service.delete_training_data_file(filename)
        return {
            "success": True,
            "message": f"Training data file {filename} deleted successfully"
        }
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete training data: {str(e)}"
        )


@app.delete("/api/fine-tune/training-data")
async def delete_all_training_data():
    """
    Delete all training data files.
    """
    try:
        count = fine_tuning_service.delete_all_training_data()
        return {
            "success": True,
            "message": f"Deleted {count} training data file(s)",
            "count": count
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete training data: {str(e)}"
        )


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: int):
    """
    Delete an indexed document and its chunks from the vector store.
    """
    try:
        result = await ingest_service.delete_document(doc_id)
        # Reload vector store in chat service after deletion
        chat_service.reload_vector_store()
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )


@app.get("/api/fine-tune/debug")
async def debug_vector_store():
    """
    Debug endpoint to inspect vector store contents.
    """
    try:
        vector_store = ingest_service.vector_store
        if vector_store is None:
            return {
                "success": False,
                "message": "Vector store is None"
            }
        
        index = vector_store.index
        num_vectors = index.ntotal
        
        # Try to get a sample of documents
        sample_docs = []
        try:
            docstore = vector_store.docstore
            # Get first 10 documents
            sample_ids = list(range(min(10, num_vectors)))
            sample_docs_raw = docstore.search(sample_ids)
            
            for idx, doc_item in enumerate(sample_docs_raw):
                if doc_item:
                    if hasattr(doc_item, 'metadata') and hasattr(doc_item, 'page_content'):
                        sample_docs.append({
                            "id": idx,
                            "has_metadata": True,
                            "metadata": dict(doc_item.metadata),
                            "content_preview": doc_item.page_content[:100] + "..." if len(doc_item.page_content) > 100 else doc_item.page_content,
                            "content_length": len(doc_item.page_content)
                        })
                    elif isinstance(doc_item, dict):
                        sample_docs.append({
                            "id": idx,
                            "has_metadata": True,
                            "metadata": doc_item.get("metadata", {}),
                            "content_preview": str(doc_item.get("page_content", ""))[:100],
                            "content_length": len(str(doc_item.get("page_content", "")))
                        })
        except Exception as e:
            sample_docs = [{"error": str(e)}]
        
        return {
            "success": True,
            "num_vectors": num_vectors,
            "has_docstore": hasattr(vector_store, 'docstore'),
            "sample_documents": sample_docs,
            "documents_tracked": document_tracker.get_document_count()
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.post("/api/ingest")
async def ingest_document(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None)
):
    """
    Ingest a document (PDF file or URL) into the vector store.
    """
    try:
        if not file and not url:
            raise HTTPException(
                status_code=400,
                detail="Either file or URL must be provided"
            )

        if file:
            # Handle PDF file
            if file.content_type != "application/pdf":
                raise HTTPException(
                    status_code=400,
                    detail="Only PDF files are supported"
                )

            contents = await file.read()
            result = await ingest_service.ingest_pdf(contents, file.filename)
        elif url:
            # Handle URL
            result = await ingest_service.ingest_url(url)
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid input"
            )

        return {
            "success": True,
            "message": "Document indexed successfully",
            "chunksCount": result.get("chunks_count", 0)
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Ingest error: {error_trace}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest document: {str(e)}"
        )


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Chat with the indexed documents using RAG.
    """
    try:
        if not request.query:
            raise HTTPException(
                status_code=400,
                detail="Query is required"
            )

        # Update model if fine_tuned_model is specified
        # Default to phi3:mini for better instruction following on 8GB RAM
        model_to_use = request.fine_tuned_model or "phi3:mini"
        if chat_service.current_model != model_to_use:
            chat_service._update_llm(model_to_use)
            print(f"Switched to model: {model_to_use}")

        # Stream the response
        async def generate_response():
            async for chunk in chat_service.chat_stream(
                messages=request.messages,
                query=request.query,
                doc_id=request.doc_id
            ):
                yield f"data: {chunk}\n\n"

        return StreamingResponse(
            generate_response(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat: {str(e)}"
        )


@app.delete("/api/clear-all")
async def clear_all_data():
    """
    Clear all data: documents, vector store, training data, and modelfiles.
    WARNING: This is irreversible!
    """
    try:
        # Clear vector store and documents
        ingest_result = await ingest_service.clear_all_data()
        
        # Clear fine-tuning data
        fine_tune_result = fine_tuning_service.clear_all_fine_tuning_data()
        
        # Reload vector store in chat service
        chat_service.reload_vector_store()
        
        return {
            "success": True,
            "message": "All data cleared successfully",
            "details": {
                "vector_store": "cleared",
                "documents": "cleared",
                **fine_tune_result
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear all data: {str(e)}"
        )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
