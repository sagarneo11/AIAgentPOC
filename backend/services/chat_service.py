from langchain_community.chat_models import ChatOllama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import os
import json
import asyncio
from typing import List, Dict, AsyncGenerator

class ChatService:
    def __init__(self, model_name: str = "phi3:mini"):
        # Default to phi3:mini - excellent instruction following, fits in 8GB RAM
        # Alternatives: "qwen2.5:3b", "llama3.2:3b", "gemma2:2b"
        self.base_model = "phi3:mini"
        self.current_model = model_name
        self._update_llm(model_name)
    
    def _update_llm(self, model_name: str):
        """Update the LLM with a new model. Falls back to base model if model doesn't exist."""
        original_model = model_name
        
        # Check if model exists in Ollama before trying to use it
        try:
            import httpx
            response = httpx.get(f"http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]
                
                # Check if the exact model name exists
                if model_name not in model_names:
                    # Check if it's a fine-tuned model (starts with "rag-")
                    if model_name.startswith("rag-"):
                        print(f"\n⚠️  WARNING: Fine-tuned model '{model_name}' not found in Ollama.")
                        print(f"   Available models: {', '.join(model_names[:5])}...")
                        print(f"   Falling back to base model: {self.base_model}")
                        print(f"\n   To create the fine-tuned model, run:")
                        # Extract doc_id from model name (e.g., "rag-smolvlm-doc-2" -> "2")
                        doc_id = model_name.split("-")[-1] if "-" in model_name else "2"
                        print(f"   ollama create {model_name} -f backend/fine_tuned_models/Modelfile_doc_{doc_id}")
                        print(f"   (See backend/CREATE_FINE_TUNED_MODEL.md for details)\n")
                        model_name = self.base_model
                    else:
                        print(f"⚠️  Warning: Model '{model_name}' not found. Falling back to {self.base_model}")
                        model_name = self.base_model
        except Exception as e:
            print(f"⚠️  Could not verify model existence: {e}. Using model name as-is.")
        
        # Update current_model AFTER the check
        self.current_model = model_name
        
        self.llm = ChatOllama(
            model=model_name,
            base_url="http://localhost:11434",
            temperature=0.7
        )
        
        if original_model != model_name:
            print(f"✅ Using model: {model_name} (requested: {original_model})")
        # For embeddings, we can still use a smaller model or keep smolvlm2
        # Note: phi3:mini doesn't support embeddings, so we use smolvlm2 for embeddings
        self.embeddings = OllamaEmbeddings(
            model="richardyoung/smolvlm2-2.2b-instruct",  # Keep for embeddings
            base_url="http://localhost:11434"
        )
        # Get the backend directory path
        backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.store_path = os.path.join(backend_dir, "vector_store")
        self.vector_store = None
        self._load_vector_store()

    def _load_vector_store(self):
        """Load the vector store from disk."""
        store_path = self.store_path
        if os.path.exists(store_path) and os.listdir(store_path):
            try:
                self.vector_store = FAISS.load_local(
                    store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded vector store from {store_path}")
            except Exception as e:
                print(f"Error loading vector store: {e}")
                self.vector_store = None
        else:
            print(f"Vector store not found at {store_path}")
            self.vector_store = None
    
    def reload_vector_store(self):
        """Reload the vector store (useful after deletion)."""
        self._load_vector_store()

    async def chat_stream(
        self,
        messages: List[Dict],
        query: str,
        doc_id: int = None
    ) -> AsyncGenerator[str, None]:
        """Stream chat responses using RAG."""
        try:
            if self.vector_store is None:
                yield json.dumps({
                    "error": "No documents indexed. Please index a document first."
                })
                return
            
            # Verify vector store has documents
            try:
                index = self.vector_store.index
                num_vectors = index.ntotal
                print(f"Vector store contains {num_vectors} vectors")
                if num_vectors == 0:
                    yield json.dumps({
                        "error": "Vector store is empty. Please index documents first."
                    })
                    return
            except Exception as e:
                print(f"Error checking vector store: {e}")
                yield json.dumps({
                    "error": f"Error accessing vector store: {str(e)}"
                })
                return

            # Create retriever with optional document filter
            # Increase k to get more context for better answers
            # For summary queries, we want more chunks
            is_summary_query = any(word in query.lower() for word in ["summary", "summarize", "summarise", "overview"])
            k_value = 20 if is_summary_query else 15
            search_kwargs = {"k": k_value}
            
            if doc_id is not None:
                # Filter by document ID using metadata
                # FAISS filter function receives metadata dict
                def filter_func(metadata):
                    doc_id_in_meta = metadata.get("doc_id")
                    # Handle different types
                    if isinstance(doc_id_in_meta, int):
                        return doc_id_in_meta == doc_id
                    elif isinstance(doc_id_in_meta, (list, tuple)) and len(doc_id_in_meta) > 0:
                        try:
                            return int(doc_id_in_meta[0]) == doc_id
                        except:
                            return False
                    elif isinstance(doc_id_in_meta, (str, float)):
                        try:
                            return int(doc_id_in_meta) == doc_id
                        except:
                            return False
                    return False
                search_kwargs["filter"] = filter_func
                print(f"Filtering chunks for document ID: {doc_id}")
            
            retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
            
            # For summary queries or when filtering, create a fallback retriever without filter
            fallback_retriever = None
            if doc_id is not None:
                # Create fallback without filter in case filter is too strict
                fallback_search_kwargs = {"k": k_value}
                fallback_retriever = self.vector_store.as_retriever(search_kwargs=fallback_search_kwargs)
                print(f"Created fallback retriever (no filter) for document ID: {doc_id}")
            
            # Create prompt template with VERY explicit instructions
            # Use a format that makes context impossible to ignore - put context in human message
            # This forces the model to see the context as part of the conversation
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful assistant that answers questions based on provided text. 
You MUST use the text provided in the user's message to answer their question.
NEVER say you don't have access - the text IS provided in the message.
NEVER use general knowledge - ONLY use what's in the provided text."""),
                ("human", """Here is the document text:

{context}

Now answer this question using ONLY the text above: {input}

Remember: The text above IS the document. Use it to answer the question.""")
            ])

            # Create document chain with explicit document formatting
            # This ensures all retrieved documents are included in the context
            document_chain = create_stuff_documents_chain(
                llm=self.llm,
                prompt=prompt,
                # Ensure all documents are included
                document_variable_name="context"
            )

            # Create retrieval chain
            retrieval_chain = create_retrieval_chain(
                retriever=retriever,
                combine_docs_chain=document_chain
            )

            # Run the chain in a thread pool to avoid blocking
            def run_chain():
                try:
                    # First, verify retrieval is working
                    retrieved_docs = retriever.get_relevant_documents(query)
                    print(f"\n=== RAG DEBUG INFO ===")
                    print(f"Query: {query}")
                    print(f"Retrieved {len(retrieved_docs)} documents with filter")
                    
                    # If no documents retrieved and we have a filter, try without filter
                    if len(retrieved_docs) == 0 and fallback_retriever is not None:
                        print("No documents found with filter, trying without filter...")
                        retrieved_docs = fallback_retriever.get_relevant_documents(query)
                        print(f"Retrieved {len(retrieved_docs)} documents without filter")
                        
                        # If we got documents without filter, manually filter by doc_id
                        if len(retrieved_docs) > 0 and doc_id is not None:
                            print(f"Manually filtering {len(retrieved_docs)} documents for doc_id {doc_id}...")
                            filtered_docs = []
                            for doc in retrieved_docs:
                                doc_id_in_meta = doc.metadata.get("doc_id") if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict) else None
                                if doc_id_in_meta is not None:
                                    # Handle different types
                                    if isinstance(doc_id_in_meta, int) and doc_id_in_meta == doc_id:
                                        filtered_docs.append(doc)
                                    elif isinstance(doc_id_in_meta, (list, tuple)) and len(doc_id_in_meta) > 0:
                                        try:
                                            if int(doc_id_in_meta[0]) == doc_id:
                                                filtered_docs.append(doc)
                                        except:
                                            pass
                                    elif isinstance(doc_id_in_meta, (str, float)):
                                        try:
                                            if int(doc_id_in_meta) == doc_id:
                                                filtered_docs.append(doc)
                                        except:
                                            pass
                            
                            if len(filtered_docs) > 0:
                                print(f"Found {len(filtered_docs)} documents after manual filtering")
                                retrieved_docs = filtered_docs
                            else:
                                print(f"Warning: No documents match doc_id {doc_id} after manual filtering")
                                # For summary queries, use all retrieved docs as fallback
                                if is_summary_query:
                                    print(f"Using all {len(retrieved_docs)} retrieved documents for summary (doc_id filter too strict)")
                                else:
                                    # Still use them but warn
                                    print(f"Using all {len(retrieved_docs)} retrieved documents as fallback")
                    
                    # If still no documents, try with a generic query
                    if len(retrieved_docs) == 0:
                        print("Trying with empty/generic query to get any documents...")
                        try:
                            # Try to get documents using empty query or common terms
                            generic_queries = ["", "document", "text", "content", "information", "the", "a"]
                            for gen_query in generic_queries:
                                try:
                                    retrieved_docs = self.vector_store.similarity_search(gen_query, k=min(k_value, 20))
                                    if len(retrieved_docs) > 0:
                                        print(f"Found {len(retrieved_docs)} documents using generic query: '{gen_query}'")
                                        
                                        # If we have doc_id filter, try to filter these manually
                                        if doc_id is not None:
                                            filtered_docs = []
                                            for doc in retrieved_docs:
                                                doc_id_in_meta = doc.metadata.get("doc_id") if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict) else None
                                                if doc_id_in_meta is not None:
                                                    if isinstance(doc_id_in_meta, int) and doc_id_in_meta == doc_id:
                                                        filtered_docs.append(doc)
                                                    elif isinstance(doc_id_in_meta, (list, tuple)) and len(doc_id_in_meta) > 0:
                                                        try:
                                                            if int(doc_id_in_meta[0]) == doc_id:
                                                                filtered_docs.append(doc)
                                                        except:
                                                            pass
                                            
                                            if len(filtered_docs) > 0:
                                                retrieved_docs = filtered_docs
                                                print(f"Filtered to {len(filtered_docs)} documents matching doc_id {doc_id}")
                                            elif is_summary_query:
                                                # For summaries, use all if filter fails
                                                print(f"Using all {len(retrieved_docs)} documents for summary (filter too strict)")
                                        
                                        break
                                except Exception as e:
                                    print(f"Error with generic query '{gen_query}': {e}")
                                    continue
                        except Exception as e:
                            print(f"Error in fallback retrieval: {e}")
                    
                    if len(retrieved_docs) == 0:
                        print("ERROR: No documents retrieved even with fallback methods!")
                        # Try to get total count
                        try:
                            index = self.vector_store.index
                            num_vectors = index.ntotal
                            print(f"Vector store has {num_vectors} total vectors")
                            if num_vectors > 0:
                                error_msg = f"Documents are indexed ({num_vectors} vectors) but retrieval failed."
                                if doc_id is not None:
                                    error_msg += f"\n\nThe document filter (doc_id={doc_id}) might be too strict."
                                    error_msg += "\nTry: Deselect the document and ask again, or check if the document was indexed correctly."
                                else:
                                    error_msg += "\n\nThis might be due to:\n1. Query doesn't match any chunks\n2. Embedding model issues\n3. Vector store corruption"
                                error_msg += "\n\nCheck backend logs for more details."
                                return error_msg
                        except Exception as e:
                            print(f"Error checking vector count: {e}")
                        return "No relevant information found in the indexed documents. Please make sure documents are indexed and try again."
                    
                    # Log context details
                    total_context_length = sum(len(doc.page_content) for doc in retrieved_docs)
                    print(f"Total context length: {total_context_length} chars")
                    print(f"Number of chunks: {len(retrieved_docs)}")
                    
                    # Show preview of retrieved chunks
                    for i, doc in enumerate(retrieved_docs[:3]):
                        print(f"\nChunk {i+1} ({len(doc.page_content)} chars):")
                        print(f"  Preview: {doc.page_content[:150]}...")
                        if hasattr(doc, 'metadata'):
                            print(f"  Metadata: {doc.metadata}")
                    
                    # Build full context to verify
                    full_context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
                    print(f"\n=== CONTEXT BEING SENT TO LLM ===")
                    print(f"Context length: {len(full_context)} chars")
                    print(f"Number of chunks: {len(retrieved_docs)}")
                    print(f"Context preview (first 1500 chars):\n{full_context[:1500]}...")
                    print(f"Context ends with: ...{full_context[-200:]}")
                    
                    # Verify context contains actual content (not just metadata)
                    if len(full_context.strip()) < 100:
                        print("⚠️  WARNING: Context seems too short! This might cause issues.")
                        return "Error: Retrieved context is too short. Please try re-indexing the document."
                    
                    # Check for common words that should appear in answers if context is used
                    context_sample = full_context[:2000].lower()
                    context_has_content = len([w for w in context_sample.split() if len(w) > 3]) > 50
                    if not context_has_content:
                        print("⚠️  WARNING: Context doesn't seem to have substantial content!")
                    
                    # Invoke the chain with explicit input
                    print(f"\n=== INVOKING RETRIEVAL CHAIN ===")
                    print(f"Query: '{query}'")
                    print(f"Using model: {self.current_model}")
                    
                    try:
                        # Debug: Check what the chain receives
                        print(f"Invoking chain with input: '{query}'")
                        result = retrieval_chain.invoke({"input": query})
                        answer = result.get("answer", "I couldn't generate a response.")
                        
                        # Debug: Check if context was in the result
                        if "context" in result:
                            print(f"Context was passed to chain: {len(str(result.get('context', '')))} chars")
                        else:
                            print("⚠️  WARNING: 'context' not found in chain result!")
                            print(f"Result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
                        
                        print(f"\n=== GENERATED ANSWER ===")
                        print(f"Answer length: {len(answer)} chars")
                        print(f"Answer preview: {answer[:300]}...")
                        
                        # Verify answer uses context - check for generic phrases that indicate context was ignored
                        generic_phrases = [
                            "i don't have access",
                            "i'm an ai language model",
                            "i don't have real-time data",
                            "i can provide general information",
                            "as an ai language model",
                            "i don't have access to real-time data",
                            "i can't create a summary",
                            "i'm sorry, but i can't",
                            "there is no actual information",
                            "without additional context"
                        ]
                        answer_lower = answer.lower()
                        found_generic = [phrase for phrase in generic_phrases if phrase in answer_lower]
                        
                        if found_generic:
                            print(f"\n❌ ERROR: Answer contains generic phrases indicating context was ignored!")
                            print(f"Found phrases: {found_generic}")
                            print(f"Full answer: {answer}")
                            print(f"\nThis means the model is NOT using the context. Possible causes:")
                            print(f"  1. Model is too small/weak to follow instructions")
                            print(f"  2. Context format is wrong")
                            print(f"  3. Prompt structure needs to be different")
                            
                            # Try a more direct approach - manually format the prompt
                            # Use a very simple format that's impossible to ignore
                            print(f"\n⚠️  Attempting fallback: manually formatting prompt with context...")
                            manual_prompt = f"""Here is document text:

{full_context[:8000]}

Question: {query}

Answer using ONLY the text above. The text above IS the document. Do not say you don't have access."""
                            
                            # Use the LLM directly with the manual prompt
                            from langchain_core.messages import HumanMessage
                            try:
                                manual_response = self.llm.invoke([HumanMessage(content=manual_prompt)])
                                answer = manual_response.content if hasattr(manual_response, 'content') else str(manual_response)
                                print(f"Fallback answer: {answer[:300]}...")
                                
                                # Check if fallback also has generic phrases
                                fallback_lower = answer.lower()
                                fallback_generic = [p for p in generic_phrases if p in fallback_lower]
                                if fallback_generic:
                                    print(f"⚠️  Fallback also contains generic phrases: {fallback_generic}")
                                    print(f"This suggests the model ({self.current_model}) is too small/weak to follow instructions.")
                                    print(f"Consider using a larger model or fine-tuning.")
                            except Exception as fallback_error:
                                print(f"Fallback also failed: {fallback_error}")
                                # Return the original answer even if it's generic
                                pass
                        else:
                            # Check if answer references context keywords
                            context_keywords = [w.lower() for w in full_context[:1000].split() if len(w) > 4][:30]
                            answer_words = set(answer_lower.split())
                            matching_keywords = [kw for kw in context_keywords if kw in answer_words]
                            
                            if len(matching_keywords) > 0:
                                print(f"✅ Answer references context keywords: {matching_keywords[:5]}...")
                            elif len(answer) > 100:
                                print(f"⚠️  WARNING: Answer is long but doesn't reference context keywords")
                                print(f"Context keywords sample: {context_keywords[:10]}")
                            
                    except Exception as chain_error:
                        print(f"❌ Error in retrieval chain: {chain_error}")
                        import traceback
                        traceback.print_exc()
                        raise
                    
                    # Check if answer seems generic (not using context)
                    generic_phrases = [
                        "i don't have access",
                        "i'm an ai language model",
                        "i don't have real-time data",
                        "i can provide general information",
                        "as an ai language model",
                        "i don't have access to real-time data"
                    ]
                    answer_lower = answer.lower()
                    if any(phrase in answer_lower for phrase in generic_phrases):
                        print(f"\n⚠️  WARNING: Answer seems generic and might not be using context!")
                        print(f"Answer: {answer[:300]}...")
                        print(f"\nThis suggests the model is ignoring the context. Check:")
                        print(f"  1. Is the context being passed correctly?")
                        print(f"  2. Is the model following the prompt instructions?")
                        print(f"  3. Try increasing k (currently {search_kwargs.get('k', 10)})")
                    else:
                        print(f"\n✓ Answer appears to be using context")
                    
                    print(f"Generated answer length: {len(answer)} chars")
                    print(f"=== END RAG DEBUG ===\n")
                    
                    return answer
                except Exception as e:
                    print(f"Error in run_chain: {e}")
                    import traceback
                    traceback.print_exc()
                    raise

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, run_chain)

            # Stream the response word by word
            words = response.split()
            for i, word in enumerate(words):
                chunk_data = {
                    "content": word + (" " if i < len(words) - 1 else "")
                }
                yield json.dumps(chunk_data)
                # Small delay to simulate streaming
                await asyncio.sleep(0.02)

        except Exception as e:
            error_data = {
                "error": f"Failed to generate response: {str(e)}"
            }
            yield json.dumps(error_data)
