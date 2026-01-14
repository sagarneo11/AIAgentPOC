import json
import os
from typing import List, Dict
from datetime import datetime

class DocumentTracker:
    def __init__(self):
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.tracker_file = os.path.join(backend_dir, "documents.json")
        self.documents = []
        self._load_documents()

    def _load_documents(self):
        """Load documents from file."""
        if os.path.exists(self.tracker_file):
            try:
                with open(self.tracker_file, 'r') as f:
                    self.documents = json.load(f)
            except Exception as e:
                print(f"Error loading documents tracker: {e}")
                self.documents = []
        else:
            self.documents = []

    def _save_documents(self):
        """Save documents to file."""
        try:
            with open(self.tracker_file, 'w') as f:
                json.dump(self.documents, f, indent=2)
        except Exception as e:
            print(f"Error saving documents tracker: {e}")

    def add_document(self, doc_type: str, source: str, chunks_count: int, filename: str = None):
        """Add a document to the tracker."""
        doc = {
            "id": len(self.documents) + 1,
            "type": doc_type,  # "pdf" or "url"
            "source": source,
            "filename": filename if doc_type == "pdf" else None,
            "chunks_count": chunks_count,
            "indexed_at": datetime.now().isoformat(),
            "fine_tuned_model": None  # Will be set when model is created
        }
        self.documents.append(doc)
        self._save_documents()
        return doc
    
    def update_document_model(self, doc_id: int, model_name: str):
        """Update a document with its fine-tuned model name."""
        doc = next((d for d in self.documents if d["id"] == doc_id), None)
        if doc:
            doc["fine_tuned_model"] = model_name
            self._save_documents()
            return True
        return False

    def get_documents(self) -> List[Dict]:
        """Get all indexed documents."""
        return self.documents

    def get_document_count(self) -> int:
        """Get total number of indexed documents."""
        return len(self.documents)

    def get_total_chunks(self) -> int:
        """Get total number of chunks across all documents."""
        return sum(doc.get("chunks_count", 0) for doc in self.documents)
    
    def delete_document(self, doc_id: int) -> bool:
        """Delete a document from the tracker."""
        original_count = len(self.documents)
        self.documents = [d for d in self.documents if d["id"] != doc_id]
        if len(self.documents) < original_count:
            self._save_documents()
            return True
        return False
    
    def get_document(self, doc_id: int) -> Dict:
        """Get a specific document by ID."""
        return next((d for d in self.documents if d["id"] == doc_id), None)
    
    def clear_all_documents(self):
        """Clear all documents from the tracker."""
        self.documents = []
        self._save_documents()
        print("All documents cleared from tracker")