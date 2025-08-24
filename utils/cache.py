import os
import json
import pickle
from typing import List, Dict, Any, Optional, Union
import hashlib
from datetime import datetime

from langchain_core.documents import Document

class RAGCache:
    """Cache for RAG system to store embeddings and query results"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        """
        Initialize the cache
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = cache_dir
        self.embeddings_dir = os.path.join(cache_dir, "embeddings")
        self.query_results_dir = os.path.join(cache_dir, "query_results")
        
        # Create cache directories
        os.makedirs(self.embeddings_dir, exist_ok=True)
        os.makedirs(self.query_results_dir, exist_ok=True)
    
    def _get_embedding_path(self, text: str, model_name: str) -> str:
        """Get path for cached embedding"""
        # Create a unique hash for the text and model
        text_hash = hashlib.md5(text.encode()).hexdigest()
        model_hash = hashlib.md5(model_name.encode()).hexdigest()
        return os.path.join(self.embeddings_dir, f"{text_hash}_{model_hash}.pkl")
    
    def _get_query_result_path(self, query: str) -> str:
        """Get path for cached query result"""
        # Create a unique hash for the query
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return os.path.join(self.query_results_dir, f"{query_hash}.pkl")
    
    def get_embedding(self, text: str, model_name: str) -> Optional[List[float]]:
        """
        Get embedding from cache
        
        Args:
            text: Text to get embedding for
            model_name: Name of the model used for embedding
            
        Returns:
            Embedding vector or None if not found
        """
        embedding_path = self._get_embedding_path(text, model_name)
        if os.path.exists(embedding_path):
            with open(embedding_path, "rb") as f:
                return pickle.load(f)
        return None
    
    def cache_embedding(self, text: str, model_name: str, embedding: List[float]) -> None:
        """
        Cache embedding
        
        Args:
            text: Text that was embedded
            model_name: Name of the model used for embedding
            embedding: Embedding vector
        """
        embedding_path = self._get_embedding_path(text, model_name)
        with open(embedding_path, "wb") as f:
            pickle.dump(embedding, f)
    
    def get_query_result(self, query: str) -> Optional[List[Document]]:
        """
        Get query result from cache
        
        Args:
            query: Query text
            
        Returns:
            List of documents or None if not found
        """
        print(f"🔍 DEBUG [Cache.get_query_result]: Checking cache for query: '{query[:50]}...'")
        query_result_path = self._get_query_result_path(query)
        print(f"🔍 DEBUG [Cache.get_query_result]: Cache path: {query_result_path}")
        
        if os.path.exists(query_result_path):
            print(f"🔍 DEBUG [Cache.get_query_result]: Cache file found, loading...")
            with open(query_result_path, "rb") as f:
                result = pickle.load(f)
            print(f"🔍 DEBUG [Cache.get_query_result]: Loaded {len(result) if result else 0} documents from cache")
            return result
        
        print(f"🔍 DEBUG [Cache.get_query_result]: No cache file found")
        return None
    
    def cache_query_result(self, query: str, documents: List[Document]) -> None:
        """
        Cache query result
        
        Args:
            query: Query text
            documents: List of retrieved documents
        """
        query_result_path = self._get_query_result_path(query)
        with open(query_result_path, "wb") as f:
            pickle.dump(documents, f)
    
    def clear_cache(self, cache_type: Optional[str] = None) -> int:
        """
        Clear cache
        
        Args:
            cache_type: Type of cache to clear ("embeddings", "query_results", or None for all)
            
        Returns:
            Number of files deleted
        """
        files_deleted = 0
        
        if cache_type in [None, "embeddings"]:
            for file in os.listdir(self.embeddings_dir):
                os.remove(os.path.join(self.embeddings_dir, file))
                files_deleted += 1
        
        if cache_type in [None, "query_results"]:
            for file in os.listdir(self.query_results_dir):
                os.remove(os.path.join(self.query_results_dir, file))
                files_deleted += 1
        
        return files_deleted