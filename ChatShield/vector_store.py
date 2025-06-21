import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
import logging
from config import config

class SecureVectorStore:
    def __init__(self, index_path: str = "vector_store"):
        self.index_path = index_path
        self.embeddings = OpenAIEmbeddings(
            model=config.embedding_model,
            openai_api_key=config.openai_api_key
        )
        self.dimension = 1536  # OpenAI ada-002 embedding dimension
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Store metadata separately
        self.metadata_store = []
        
        # Load existing index if it exists
        self.load_index()
    
    def load_index(self):
        """Load existing FAISS index and metadata"""
        index_file = f"{self.index_path}/faiss_index.bin"
        metadata_file = f"{self.index_path}/metadata.pkl"
        
        if os.path.exists(index_file) and os.path.exists(metadata_file):
            try:
                # Load FAISS index
                self.index = faiss.read_index(index_file)
                
                # Load metadata
                with open(metadata_file, 'rb') as f:
                    self.metadata_store = pickle.load(f)
                
                logging.info(f"Loaded existing index with {len(self.metadata_store)} documents")
            except Exception as e:
                logging.error(f"Error loading existing index: {e}")
                # Reinitialize if loading fails
                self.index = faiss.IndexFlatL2(self.dimension)
                self.metadata_store = []
    
    def save_index(self):
        """Save FAISS index and metadata"""
        os.makedirs(self.index_path, exist_ok=True)
        
        try:
            # Save FAISS index
            faiss.write_index(self.index, f"{self.index_path}/faiss_index.bin")
            
            # Save metadata
            with open(f"{self.index_path}/metadata.pkl", 'wb') as f:
                pickle.dump(self.metadata_store, f)
            
            logging.info(f"Saved index with {len(self.metadata_store)} documents")
        except Exception as e:
            logging.error(f"Error saving index: {e}")
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Add document chunks to the vector store"""
        if not chunks:
            return
        
        # Extract content for embedding
        contents = [chunk['content'] for chunk in chunks]
        
        # Generate embeddings
        try:
            embeddings = self.embeddings.embed_documents(contents)
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Store metadata
            for chunk in chunks:
                self.metadata_store.append({
                    'content': chunk['content'],
                    'access_level': chunk['access_level'],
                    'source': chunk['source'],
                    'chunk_id': chunk['chunk_id'],
                    'chunk_size': chunk['chunk_size']
                })
            
            logging.info(f"Added {len(chunks)} chunks to vector store")
            
            # Save after adding
            self.save_index()
            
        except Exception as e:
            logging.error(f"Error adding documents to vector store: {e}")
            raise
    
    def search(self, query: str, user_role: str, k: int = 5) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Search for relevant documents with access control.
        Returns a tuple of (accessible_results, inaccessible_sources).
        """
        try:
            if self.index.ntotal == 0:
                return [], []

            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            query_vector = np.array([query_embedding]).astype('float32')
            
            # Search in FAISS index, get more results to allow for filtering
            search_k = min(self.index.ntotal, k * 5)
            distances, indices = self.index.search(query_vector, search_k)
            
            # Filter results based on access level
            filtered_results = []
            inaccessible_sources = set()
            
            for i, idx in enumerate(indices[0]):
                if idx < 0 or idx >= len(self.metadata_store):
                    continue

                metadata = self.metadata_store[idx]
                
                if self._can_access(user_role, metadata['access_level']):
                    if len(filtered_results) < k:
                        result = {
                            'content': metadata['content'],
                            'access_level': metadata['access_level'],
                            'source': metadata['source'],
                            'chunk_id': metadata['chunk_id'],
                            'distance': float(distances[0][i])
                        }
                        filtered_results.append(result)
                else:
                    inaccessible_sources.add(metadata['source'])

            logging.info(f"Found {len(filtered_results)} accessible chunks and matched {len(inaccessible_sources)} inaccessible sources for role '{user_role}'.")
            return filtered_results, list(inaccessible_sources)
            
        except Exception as e:
            logging.error(f"Error searching vector store: {e}")
            return [], []
    
    def _can_access(self, user_role: str, chunk_access_level: str) -> bool:
        """Check if user can access a chunk based on role hierarchy"""
        access_levels = config.get_access_levels()
        
        if user_role not in access_levels:
            return False
        
        role_config = access_levels[user_role]
        
        # Manager has access to all levels
        if role_config.get('can_access') == 'all':
            return True
        
        # Employer can only access Employer-level chunks
        if user_role == 'Employer':
            return chunk_access_level == 'Employer'
        
        return False
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about stored documents"""
        if not self.metadata_store:
            return {
                'total_chunks': 0,
                'manager_chunks': 0,
                'employer_chunks': 0,
                'sources': []
            }
        
        manager_count = sum(1 for meta in self.metadata_store if meta['access_level'] == 'Manager')
        employer_count = sum(1 for meta in self.metadata_store if meta['access_level'] == 'Employer')
        sources = list(set(meta['source'] for meta in self.metadata_store))
        
        return {
            'total_chunks': len(self.metadata_store),
            'manager_chunks': manager_count,
            'employer_chunks': employer_count,
            'sources': sources
        }
    
    def clear_all(self):
        """Clear all documents from the vector store"""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata_store = []
        self.save_index()
        logging.info("Cleared all documents from vector store")
    
    def remove_document(self, source: str):
        """Remove all chunks from a specific document"""
        # This is a simplified implementation
        # In production, you'd want to rebuild the index without the specific chunks
        logging.warning("Document removal not implemented in this version") 