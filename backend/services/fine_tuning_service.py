import os
import json
from typing import List, Dict
from services.document_tracker import DocumentTracker
from services.ingest_service import IngestService
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings

class FineTuningService:
    def __init__(self, document_tracker: DocumentTracker, ingest_service: IngestService):
        self.document_tracker = document_tracker
        self.ingest_service = ingest_service
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.training_data_dir = os.path.join(backend_dir, "training_data")
        self.models_dir = os.path.join(backend_dir, "fine_tuned_models")
        os.makedirs(self.training_data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        
    def prepare_training_data(self, format: str = "alpaca") -> str:
        """
        Prepare training data from indexed documents.
        Formats: 'alpaca', 'chatml', 'jsonl'
        """
        # Load vector store to get all chunks
        vector_store = self.ingest_service.vector_store
        if vector_store is None:
            raise Exception("No documents indexed. Please index documents first.")
        
        # Get all documents from tracker
        documents = self.document_tracker.get_documents()
        
        if not documents:
            raise Exception("No documents tracked. Please index documents first.")
        
        print(f"Preparing training data from {len(documents)} documents...")
        
        # Get all chunks from FAISS
        training_data = []
        chunks_by_doc = {}
        
        try:
            # Get total number of vectors in FAISS index
            index = vector_store.index
            num_vectors = index.ntotal
            print(f"FAISS index contains {num_vectors} vectors")
            
            if num_vectors == 0:
                raise Exception("Vector store is empty. No chunks found.")
            
            # Access docstore to get all documents
            docstore = vector_store.docstore
            
            # Method 1: Try to get all documents by ID
            print(f"Attempting to retrieve {num_vectors} documents from docstore...")
            
            try:
                # Access docstore._dict directly to avoid search() issues
                all_docs = []
                if hasattr(docstore, '_dict'):
                    # Access the internal dictionary directly
                    docstore_dict = docstore._dict
                    for doc_id_idx in range(num_vectors):
                        try:
                            # Try different key formats
                            doc_item = None
                            if doc_id_idx in docstore_dict:
                                doc_item = docstore_dict[doc_id_idx]
                            elif str(doc_id_idx) in docstore_dict:
                                doc_item = docstore_dict[str(doc_id_idx)]
                            
                            if doc_item:
                                all_docs.append(doc_item)
                        except Exception as e:
                            print(f"Error retrieving document at index {doc_id_idx}: {e}")
                            continue
                else:
                    # Fallback: try to get all values from docstore
                    print("Warning: docstore._dict not available, trying alternative method")
                    # Use similarity search as fallback
                    all_docs = []
                
                print(f"Retrieved {len(all_docs)} documents from docstore")
                
                # Process each document
                for idx, doc_item in enumerate(all_docs):
                    try:
                        if doc_item is None:
                            continue
                        
                        # Handle different document formats
                        metadata = {}
                        page_content = ""
                        
                        if hasattr(doc_item, 'metadata') and hasattr(doc_item, 'page_content'):
                            # Ensure metadata is a dict, not a list or other type
                            raw_metadata = doc_item.metadata
                            if isinstance(raw_metadata, dict):
                                metadata = raw_metadata
                            elif isinstance(raw_metadata, (list, tuple)) and len(raw_metadata) > 0:
                                # If metadata is a list, try to get the first item if it's a dict
                                if isinstance(raw_metadata[0], dict):
                                    metadata = raw_metadata[0]
                                else:
                                    metadata = {}
                            else:
                                metadata = {}
                            page_content = doc_item.page_content
                        elif isinstance(doc_item, dict):
                            raw_metadata = doc_item.get("metadata", {})
                            if isinstance(raw_metadata, dict):
                                metadata = raw_metadata
                            elif isinstance(raw_metadata, (list, tuple)) and len(raw_metadata) > 0:
                                if isinstance(raw_metadata[0], dict):
                                    metadata = raw_metadata[0]
                                else:
                                    metadata = {}
                            else:
                                metadata = {}
                            page_content = doc_item.get("page_content", "")
                        else:
                            print(f"Warning: Unknown document format at index {idx}: {type(doc_item)}")
                            continue
                        
                        if not isinstance(metadata, dict):
                            metadata = {}
                        
                        # Get doc_id from metadata and ensure it's an int or None
                        doc_id_raw = metadata.get("doc_id") if isinstance(metadata, dict) else None
                        doc_id = None
                        
                        if doc_id_raw is not None:
                            # Handle different types
                            if isinstance(doc_id_raw, int):
                                doc_id = doc_id_raw
                            elif isinstance(doc_id_raw, (list, tuple)) and len(doc_id_raw) > 0:
                                try:
                                    doc_id = int(doc_id_raw[0])
                                except (ValueError, TypeError):
                                    doc_id = None
                            elif isinstance(doc_id_raw, (str, float)):
                                try:
                                    doc_id = int(doc_id_raw)
                                except (ValueError, TypeError):
                                    doc_id = None
                        
                        if doc_id:
                            if doc_id not in chunks_by_doc:
                                chunks_by_doc[doc_id] = []
                            chunks_by_doc[doc_id].append(page_content)
                        elif documents:
                            # If no doc_id, assign to first document
                            first_doc_id = documents[0]["id"]
                            if first_doc_id not in chunks_by_doc:
                                chunks_by_doc[first_doc_id] = []
                            chunks_by_doc[first_doc_id].append(page_content)
                    except Exception as item_error:
                        print(f"Error processing document at index {idx}: {item_error}")
                        import traceback
                        traceback.print_exc()
                        continue
                
            except Exception as e:
                print(f"Error accessing docstore.search: {e}")
                import traceback
                traceback.print_exc()
                
                # Fallback: Try accessing _dict directly
                if hasattr(docstore, '_dict'):
                    print("Trying docstore._dict access...")
                    all_docstore_items = docstore._dict
                    print(f"Found {len(all_docstore_items)} items in docstore._dict")
                    
                    for key, doc_item in all_docstore_items.items():
                        doc_id = None
                        page_content = None
                        
                        metadata = {}
                        page_content = None
                        
                        if hasattr(doc_item, 'metadata') and hasattr(doc_item, 'page_content'):
                            raw_metadata = doc_item.metadata
                            if isinstance(raw_metadata, dict):
                                metadata = raw_metadata
                            elif isinstance(raw_metadata, (list, tuple)) and len(raw_metadata) > 0:
                                if isinstance(raw_metadata[0], dict):
                                    metadata = raw_metadata[0]
                            page_content = doc_item.page_content
                        elif isinstance(doc_item, dict):
                            raw_metadata = doc_item.get("metadata", {})
                            if isinstance(raw_metadata, dict):
                                metadata = raw_metadata
                            elif isinstance(raw_metadata, (list, tuple)) and len(raw_metadata) > 0:
                                if isinstance(raw_metadata[0], dict):
                                    metadata = raw_metadata[0]
                            page_content = doc_item.get("page_content")
                        else:
                            continue
                        
                        if not isinstance(metadata, dict):
                            metadata = {}
                        
                        doc_id_raw = metadata.get("doc_id") if isinstance(metadata, dict) else None
                        
                        # Ensure doc_id is an int or None
                        if doc_id_raw is not None:
                            if isinstance(doc_id_raw, int):
                                doc_id = doc_id_raw
                            elif isinstance(doc_id_raw, (list, tuple)) and len(doc_id_raw) > 0:
                                try:
                                    doc_id = int(doc_id_raw[0])
                                except (ValueError, TypeError):
                                    doc_id = None
                            elif isinstance(doc_id_raw, (str, float)):
                                try:
                                    doc_id = int(doc_id_raw)
                                except (ValueError, TypeError):
                                    doc_id = None
                        
                        if doc_id and page_content:
                            if doc_id not in chunks_by_doc:
                                chunks_by_doc[doc_id] = []
                            chunks_by_doc[doc_id].append(page_content)
            
            # Method 2: Fallback - Use multiple similarity searches if direct access failed
            if not chunks_by_doc:
                print("Using similarity search fallback method...")
                search_terms = ["", "the", "a", "is", "and", "of", "to", "in", "for", "on", "with", "that", "this", "smolvlm", "model"]
                # Use dict with content hash as key to avoid duplicates
                all_docs_dict = {}
                
                for term in search_terms:
                    try:
                        k = min(1000, num_vectors)
                        docs = vector_store.similarity_search(term, k=k)
                        for doc_item in docs:
                            content = doc_item.page_content
                            if not content:
                                continue
                            
                            # Get metadata safely
                            raw_metadata = doc_item.metadata if hasattr(doc_item, 'metadata') else {}
                            metadata = {}
                            if isinstance(raw_metadata, dict):
                                metadata = raw_metadata
                            elif isinstance(raw_metadata, (list, tuple)) and len(raw_metadata) > 0:
                                if isinstance(raw_metadata[0], dict):
                                    metadata = raw_metadata[0]
                            
                            # Get doc_id and ensure it's an int or None
                            doc_id_raw = metadata.get("doc_id") if isinstance(metadata, dict) else None
                            doc_id = None
                            
                            if doc_id_raw is not None:
                                # Handle different types
                                if isinstance(doc_id_raw, int):
                                    doc_id = doc_id_raw
                                elif isinstance(doc_id_raw, (list, tuple)) and len(doc_id_raw) > 0:
                                    # If it's a list/tuple, take the first element
                                    doc_id = int(doc_id_raw[0]) if isinstance(doc_id_raw[0], (int, float)) else None
                                elif isinstance(doc_id_raw, (str, float)):
                                    try:
                                        doc_id = int(doc_id_raw)
                                    except (ValueError, TypeError):
                                        doc_id = None
                            
                            # Use content hash as key to avoid duplicates
                            content_hash = hash(content)
                            if content_hash not in all_docs_dict:
                                all_docs_dict[content_hash] = (content, doc_id)
                    except Exception as e:
                        print(f"Error in similarity search with term '{term}': {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                print(f"Retrieved {len(all_docs_dict)} unique chunks via similarity search")
                
                # Group by doc_id
                for content_hash, (content, doc_id) in all_docs_dict.items():
                    if doc_id:
                        if doc_id not in chunks_by_doc:
                            chunks_by_doc[doc_id] = []
                        chunks_by_doc[doc_id].append(content)
                    elif documents:
                        # Assign to first document if no doc_id
                        first_doc_id = documents[0]["id"]
                        if first_doc_id not in chunks_by_doc:
                            chunks_by_doc[first_doc_id] = []
                        chunks_by_doc[first_doc_id].append(content)
                        
        except Exception as e:
            print(f"Error accessing vector store: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Failed to extract chunks from vector store: {str(e)}")
        
        print(f"Grouped chunks: {len(chunks_by_doc)} documents with chunks")
        for doc_id, chunks in chunks_by_doc.items():
            print(f"  Document {doc_id}: {len(chunks)} chunks")
        
        if not chunks_by_doc:
            raise Exception("No chunks found in vector store. Make sure documents were indexed with doc_id metadata.")
        
        # Create training examples from actual document chunks
        for doc in documents:
            doc_id = doc["id"]
            doc_chunks = chunks_by_doc.get(doc_id, [])
            
            if not doc_chunks:
                print(f"Warning: No chunks found for document {doc_id} ({doc.get('filename', doc.get('source', 'Unknown'))})")
                continue
            
            # Combine chunks for this document
            doc_name = doc.get('filename') or doc.get('source', 'Unknown')
            
            print(f"Processing document {doc_id}: {doc_name} ({len(doc_chunks)} chunks)")
            
            # Create multiple training examples per document
            if format == "alpaca":
                # Create training examples from chunks
                # Split into smaller examples if content is too long
                max_chunk_length = 2000
                chunk_size = 3  # Number of chunks per training example
                
                for i in range(0, len(doc_chunks), chunk_size):
                    chunk_batch = doc_chunks[i:i+chunk_size]
                    batch_content = "\n\n".join(chunk_batch)
                    
                    if len(batch_content) > max_chunk_length:
                        batch_content = batch_content[:max_chunk_length] + "..."
                    
                    # Create Q&A examples
                    example = {
                        "instruction": "Answer questions based on the following document content.",
                        "input": f"Document: {doc_name}\n\nContent:\n{batch_content}",
                        "output": batch_content
                    }
                    training_data.append(example)
                    
                    # Add question-answer pairs
                    questions = [
                        "What is this document about?",
                        "Summarize the key information in this document.",
                        "What are the main points discussed?",
                    ]
                    
                    for question in questions:
                        example = {
                            "instruction": question,
                            "input": f"Document: {doc_name}\n\nContent:\n{batch_content}",
                            "output": batch_content[:1000] if len(batch_content) > 1000 else batch_content
                        }
                        training_data.append(example)
                    
            elif format == "chatml":
                # Create chat format examples
                for i in range(0, len(doc_chunks), 3):
                    chunk_batch = doc_chunks[i:i+3]
                    batch_content = "\n\n".join(chunk_batch)
                    
                    example = {
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant trained on the indexed documents."},
                            {"role": "user", "content": f"Tell me about: {doc_name}"},
                            {"role": "assistant", "content": batch_content[:2000] if len(batch_content) > 2000 else batch_content}
                        ]
                    }
                    training_data.append(example)
            else:  # jsonl
                for chunk in doc_chunks:
                    example = {
                        "text": f"Document: {doc_name}\n\n{chunk}"
                    }
                    training_data.append(example)
        
        if not training_data:
            raise Exception("No training data generated. Check if documents have chunks with doc_id metadata.")
        
        print(f"Generated {len(training_data)} training examples")
        
        # Save training data
        output_file = os.path.join(self.training_data_dir, f"training_data_{format}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            if format == "jsonl":
                for item in training_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            else:
                json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved training data to {output_file}")
        return output_file
    
    def prepare_training_data_for_document(self, doc_id: int, format: str = "alpaca") -> str:
        """
        Prepare training data for a specific document.
        """
        vector_store = self.ingest_service.vector_store
        if vector_store is None:
            raise Exception("No documents indexed. Please index documents first.")
        
        documents = self.document_tracker.get_documents()
        doc = next((d for d in documents if d["id"] == doc_id), None)
        
        if not doc:
            raise Exception(f"Document with ID {doc_id} not found.")
        
        print(f"Preparing training data for document {doc_id}: {doc.get('filename', doc.get('source', 'Unknown'))}...")
        
        # Get chunks for this specific document
        training_data = []
        chunks = []
        
        try:
            index = vector_store.index
            num_vectors = index.ntotal
            docstore = vector_store.docstore
            
            # Get all documents and filter by doc_id
            # Use similarity search as a more reliable method
            print(f"Using similarity search to find chunks for document {doc_id}...")
            all_docs = []
            
            # Try multiple search terms to get all chunks
            search_terms = ["", "the", "a", "document", "text", "content"]
            seen_content = set()
            
            for term in search_terms:
                try:
                    # Get many results
                    k = min(1000, num_vectors)
                    docs = vector_store.similarity_search(term, k=k)
                    for doc_item in docs:
                        if hasattr(doc_item, 'page_content'):
                            content = doc_item.page_content
                            # Use content hash to avoid duplicates
                            content_hash = hash(content)
                            if content_hash not in seen_content:
                                seen_content.add(content_hash)
                                all_docs.append(doc_item)
                except Exception as e:
                    print(f"Error in similarity search with term '{term}': {e}")
                    continue
            
            print(f"Found {len(all_docs)} unique chunks via similarity search")
            
            for doc_item in all_docs:
                if doc_item is None:
                    continue
                
                metadata = {}
                page_content = ""
                
                if hasattr(doc_item, 'metadata') and hasattr(doc_item, 'page_content'):
                    # Ensure metadata is a dict, not a list or other type
                    raw_metadata = doc_item.metadata
                    if isinstance(raw_metadata, dict):
                        metadata = raw_metadata
                    elif isinstance(raw_metadata, (list, tuple)) and len(raw_metadata) > 0:
                        # If metadata is a list, try to get the first item if it's a dict
                        if isinstance(raw_metadata[0], dict):
                            metadata = raw_metadata[0]
                        else:
                            metadata = {}
                    else:
                        metadata = {}
                    page_content = doc_item.page_content
                elif isinstance(doc_item, dict):
                    raw_metadata = doc_item.get("metadata", {})
                    if isinstance(raw_metadata, dict):
                        metadata = raw_metadata
                    elif isinstance(raw_metadata, (list, tuple)) and len(raw_metadata) > 0:
                        if isinstance(raw_metadata[0], dict):
                            metadata = raw_metadata[0]
                        else:
                            metadata = {}
                    else:
                        metadata = {}
                    page_content = doc_item.get("page_content", "")
                else:
                    continue
                
                # Final check to ensure metadata is a dict
                if not isinstance(metadata, dict):
                    metadata = {}
                
                if not page_content:
                    continue
                
                # Get doc_id and ensure it's an int
                item_doc_id_raw = metadata.get("doc_id") if isinstance(metadata, dict) else None
                item_doc_id = None
                
                if item_doc_id_raw is not None:
                    if isinstance(item_doc_id_raw, int):
                        item_doc_id = item_doc_id_raw
                    elif isinstance(item_doc_id_raw, (list, tuple)) and len(item_doc_id_raw) > 0:
                        try:
                            item_doc_id = int(item_doc_id_raw[0])
                        except (ValueError, TypeError):
                            item_doc_id = None
                    elif isinstance(item_doc_id_raw, (str, float)):
                        try:
                            item_doc_id = int(item_doc_id_raw)
                        except (ValueError, TypeError):
                            item_doc_id = None
                
                if item_doc_id == doc_id:
                    chunks.append(page_content)
                    print(f"Found chunk for doc_id {doc_id}: {len(page_content)} chars")
            
            print(f"Total chunks found for document {doc_id}: {len(chunks)}")
            
            # If no chunks found with doc_id, try to get all chunks and assign them
            if not chunks and all_docs:
                print(f"Warning: No chunks found with doc_id={doc_id}, but found {len(all_docs)} total chunks")
                print("This might mean doc_id wasn't set during indexing. Using all chunks as fallback.")
                # Use all chunks as fallback
                for doc_item in all_docs:
                    if hasattr(doc_item, 'page_content'):
                        chunks.append(doc_item.page_content)
                    elif isinstance(doc_item, dict):
                        page_content = doc_item.get("page_content", "")
                        if page_content:
                            chunks.append(page_content)
                
        except Exception as e:
            print(f"Error extracting chunks: {e}")
            import traceback
            traceback.print_exc()
            raise Exception(f"Failed to extract chunks: {str(e)}")
        
        if not chunks:
            raise Exception(f"No chunks found for document {doc_id}. Make sure the document was indexed with doc_id metadata.")
        
        print(f"Found {len(chunks)} chunks for document {doc_id}")
        
        doc_name = doc.get('filename') or doc.get('source', 'Unknown')
        
        # Create training examples
        if format == "alpaca":
            chunk_size = 3
            max_chunk_length = 2000
            
            for i in range(0, len(chunks), chunk_size):
                chunk_batch = chunks[i:i+chunk_size]
                batch_content = "\n\n".join(chunk_batch)
                
                if len(batch_content) > max_chunk_length:
                    batch_content = batch_content[:max_chunk_length] + "..."
                
                example = {
                    "instruction": "Answer questions based on the following document content.",
                    "input": f"Document: {doc_name}\n\nContent:\n{batch_content}",
                    "output": batch_content
                }
                training_data.append(example)
        
        # Save training data
        output_file = os.path.join(self.training_data_dir, f"training_data_doc_{doc_id}_{format}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved training data to {output_file}")
        return output_file

    def create_modelfile(self, base_model: str = "phi3:mini", doc_id: int = None) -> Dict:
        """
        Create an Ollama Modelfile for fine-tuning.
        If doc_id is provided, creates a model for that specific document.
        """
        documents = self.document_tracker.get_documents()
        
        if doc_id:
            # Create model for specific document
            doc = next((d for d in documents if d["id"] == doc_id), None)
            if not doc:
                raise Exception(f"Document with ID {doc_id} not found.")
            
            doc_name = doc.get('filename') or doc.get('source', 'Unknown')
            # Create safe model name
            model_name = f"rag-smolvlm-doc-{doc_id}"
            modelfile_path = os.path.join(self.models_dir, f"Modelfile_doc_{doc_id}")
            
            modelfile_content = f"""FROM {base_model}

# Modelfile for document: {doc_name}
# {doc['chunks_count']} chunks
# NOTE: This Modelfile sets a system prompt. For actual training, use the training data with Unsloth or similar.
SYSTEM \"\"\"You are a document Q&A assistant specialized in answering questions about the following document:
- Document: {doc_name}
- Chunks indexed: {doc['chunks_count']}

IMPORTANT INSTRUCTIONS:
1. When answering questions, use ONLY information from the indexed document chunks
2. The document has been split into {doc['chunks_count']} chunks that are retrieved via RAG
3. Always base your answers on the retrieved context from these chunks
4. If the context doesn't contain the answer, say "Based on the document, this information is not available"
5. Do NOT use general knowledge - ONLY use what's in the document chunks
6. When asked for summaries, provide comprehensive summaries based on all relevant chunks

The RAG system will provide you with relevant chunks from this document. Use them to answer questions accurately.\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
"""
        else:
            # Create model for all documents
            total_chunks = self.document_tracker.get_total_chunks()
            model_name = "rag-smolvlm"
            modelfile_path = os.path.join(self.models_dir, "Modelfile")
            
            modelfile_content = f"""FROM {base_model}

# Modelfile for {len(documents)} documents with {total_chunks} total chunks
# NOTE: This Modelfile sets a system prompt. For actual training, use the training data with Unsloth or similar.
SYSTEM \"\"\"You are a document Q&A assistant specialized in answering questions about the following indexed documents:
{chr(10).join([f"- {doc.get('filename', doc.get('source', 'Unknown'))} ({doc['chunks_count']} chunks)" for doc in documents])}

IMPORTANT INSTRUCTIONS:
1. When answering questions, use ONLY information from the indexed document chunks
2. The documents have been split into {total_chunks} total chunks that are retrieved via RAG
3. Always base your answers on the retrieved context from these chunks
4. If the context doesn't contain the answer, say "Based on the documents, this information is not available"
5. Do NOT use general knowledge - ONLY use what's in the document chunks
6. When asked for summaries, provide comprehensive summaries based on all relevant chunks

The RAG system will provide you with relevant chunks from these documents. Use them to answer questions accurately.\"\"\"

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER top_k 40
"""
        
        with open(modelfile_path, 'w') as f:
            f.write(modelfile_content)
        
        # Update document tracker with model name
        if doc_id:
            self.document_tracker.update_document_model(doc_id, model_name)
        
        return {
            "modelfile_path": modelfile_path,
            "model_name": model_name,
            "instructions": f"Run: ollama create {model_name} -f {modelfile_path}"
        }
    
    def create_training_script(self) -> str:
        """
        Create a Python script for fine-tuning using unsloth or similar.
        """
        script_path = os.path.join(self.models_dir, "train_model.py")
        
        script_content = '''#!/usr/bin/env python3
"""
Fine-tuning script for smolvlm2-2.2b-instruct using Unsloth.
Install: pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
"""

import os
import json
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

# Model configuration
model_name = "richardyoung/smolvlm2-2.2b-instruct"
max_seq_length = 2048
dtype = None  # Auto detection
load_in_4bit = True  # Use 4-bit quantization

# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Prepare LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
)

# Load training data
training_data_path = "../training_data/training_data_alpaca.json"
with open(training_data_path, 'r') as f:
    training_data = json.load(f)

# Format for training
def format_prompt(example):
    instruction = example["instruction"]
    input_text = example.get("input", "")
    output = example["output"]
    
    if input_text:
        prompt = f"### Instruction:\\n{instruction}\\n\\n### Input:\\n{input_text}\\n\\n### Response:\\n{output}"
    else:
        prompt = f"### Instruction:\\n{instruction}\\n\\n### Response:\\n{output}"
    
    return {"text": prompt}

# Create dataset
dataset = Dataset.from_list(training_data)
dataset = dataset.map(format_prompt)

# Training arguments
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=not True,
        logging_steps=1,
        output_dir="outputs",
        optim="adamw_torch",
    ),
)

# Train
trainer.train()

# Save model
model.save_pretrained("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model")

print("Fine-tuning complete! Model saved to fine_tuned_model/")
'''
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        os.chmod(script_path, 0o755)  # Make executable
        
        return script_path
    
    def get_training_status(self) -> Dict:
        """Get status of fine-tuning."""
        training_data_exists = os.path.exists(self.training_data_dir) and len(os.listdir(self.training_data_dir)) > 0
        model_exists = os.path.exists(self.models_dir) and len(os.listdir(self.models_dir)) > 0
        
        return {
            "training_data_prepared": training_data_exists,
            "model_trained": model_exists,
            "documents_count": self.document_tracker.get_document_count(),
            "total_chunks": self.document_tracker.get_total_chunks()
        }
    
    def get_fine_tuned_models(self) -> Dict[int, str]:
        """Get all fine-tuned models mapped by document ID."""
        documents = self.document_tracker.get_documents()
        models = {}
        
        for doc in documents:
            if doc.get("fine_tuned_model"):
                models[doc["id"]] = doc["fine_tuned_model"]
        
        return models
    
    def list_training_data_files(self) -> List[Dict]:
        """List all training data files."""
        files = []
        if os.path.exists(self.training_data_dir):
            for filename in os.listdir(self.training_data_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.training_data_dir, filename)
                    stat = os.stat(filepath)
                    files.append({
                        "filename": filename,
                        "path": filepath,
                        "size": stat.st_size,
                        "modified": stat.st_mtime
                    })
        return sorted(files, key=lambda x: x["modified"], reverse=True)
    
    def delete_training_data_file(self, filename: str) -> bool:
        """Delete a training data file."""
        if not filename.endswith('.json'):
            raise ValueError("Only JSON files can be deleted")
        
        filepath = os.path.join(self.training_data_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File {filename} not found")
        
        # Security check: ensure file is in training_data_dir
        if not os.path.abspath(filepath).startswith(os.path.abspath(self.training_data_dir)):
            raise ValueError("Invalid file path")
        
        os.remove(filepath)
        return True
    
    def delete_all_training_data(self) -> int:
        """Delete all training data files."""
        count = 0
        if os.path.exists(self.training_data_dir):
            for filename in os.listdir(self.training_data_dir):
                if filename.endswith('.json'):
                    filepath = os.path.join(self.training_data_dir, filename)
                    os.remove(filepath)
                    count += 1
        return count
    
    def clear_all_modelfiles(self) -> int:
        """Delete all Modelfile files."""
        count = 0
        if os.path.exists(self.models_dir):
            for filename in os.listdir(self.models_dir):
                file_path = os.path.join(self.models_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    count += 1
        return count
    
    def clear_all_fine_tuning_data(self) -> Dict:
        """Clear all fine-tuning related data: training data and modelfiles."""
        training_count = self.delete_all_training_data()
        model_count = self.clear_all_modelfiles()
        return {
            "training_files_deleted": training_count,
            "modelfiles_deleted": model_count
        }
