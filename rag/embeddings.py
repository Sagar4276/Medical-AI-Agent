print(f"🔍 TRACKING: File = c:\\Users\\Sagar\\Desktop\\medical-rag\\rag\\embeddings.py | Starting file execution")

from typing import List, Dict, Any, Optional
from langchain_core.embeddings import Embeddings

from rag.custom_embeddings import SimpleMedicalEmbeddings, FastHashEmbeddings, WordCountEmbeddings
from utils.cache import RAGCache
from utils.logger import get_logger

logger = get_logger(__name__)

class CachedMedicalEmbeddings(Embeddings):
    """
    Custom medical embeddings with caching support
    Uses lightweight custom embeddings instead of pretrained models
    """
    
    def __init__(
        self,
        embedding_type: str = "simple",  # "simple", "fast", or "wordcount"
        embedding_dim: int = 128,
        cache: Optional[RAGCache] = None,
        cache_dir: str = "data/cache"
    ):
        """
        Initialize the custom embedding model
        
        Args:
            embedding_type: Type of custom embedding ("simple", "fast", or "wordcount")
            embedding_dim: Dimension of embedding vectors
            cache: Cache instance or None to create a new one
            cache_dir: Directory for cache storage if creating new cache
        """
        print(f"🔍 TRACKING: File = c:\\Users\\Sagar\\Desktop\\medical-rag\\rag\\embeddings.py | Function = CachedMedicalEmbeddings.__init__")
        print(f"🔍 DEBUG [Embeddings.__init__]: Starting initialization with type: {embedding_type}")
        logger.info(f"Initializing CachedMedicalEmbeddings with type: {embedding_type}")
        
        self.embedding_type = embedding_type
        self.embedding_dim = embedding_dim
        print(f"🔍 DEBUG [Embeddings.__init__]: Embedding type: {embedding_type}, dim: {embedding_dim}")
        
        print(f"🔍 DEBUG [Embeddings.__init__]: Creating cache with cache_dir: {cache_dir}")
        self.cache = cache or RAGCache(cache_dir=cache_dir)
        print(f"🔍 DEBUG [Embeddings.__init__]: Cache created successfully")
        
        # Initialize the appropriate custom embedding model
        if embedding_type == "simple":
            print(f"🔍 DEBUG [Embeddings.__init__]: Using SimpleMedicalEmbeddings")
            self.embedder = SimpleMedicalEmbeddings(embedding_dim=embedding_dim)
        elif embedding_type == "fast":
            print(f"🔍 DEBUG [Embeddings.__init__]: Using FastHashEmbeddings")
            self.embedder = FastHashEmbeddings(embedding_dim=embedding_dim)
        elif embedding_type == "wordcount":
            print(f"🔍 DEBUG [Embeddings.__init__]: Using WordCountEmbeddings")
            self.embedder = WordCountEmbeddings(embedding_dim=embedding_dim)
        else:
            print(f"🔍 DEBUG [Embeddings.__init__]: Unknown type, defaulting to SimpleMedicalEmbeddings")
            self.embedder = SimpleMedicalEmbeddings(embedding_dim=embedding_dim)
        
        print(f"🔍 DEBUG [Embeddings.__init__]: Custom embedding model initialization completed")
        logger.info(f"CachedMedicalEmbeddings initialized successfully")
    
    def _get_cached_embedding(self, text: str) -> Optional[List[float]]:
        """Try to get embedding from cache"""
        return self.cache.get_embedding(text, self.embedding_type)
    
    def _cache_embedding(self, text: str, embedding: List[float]) -> None:
        """Cache the embedding"""
        self.cache.cache_embedding(text, self.embedding_type, embedding)
    
    def embed_query(self, text: str) -> List[float]:
        """
        Generate embeddings for a query with caching
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        print(f"🔍 DEBUG [Embeddings.embed_query]: Starting embed_query for text: '{text[:50]}...'")
        
        # Try to get from cache
        print(f"🔍 DEBUG [Embeddings.embed_query]: Checking cache for query")
        cached_embedding = self._get_cached_embedding(text)
        if cached_embedding is not None:
            print(f"🔍 DEBUG [Embeddings.embed_query]: Found cached embedding")
            logger.debug(f"Using cached embedding for query: {text[:30]}...")
            return cached_embedding
        
        # Generate new embedding
        print(f"🔍 DEBUG [Embeddings.embed_query]: Cache miss, generating new embedding")
        logger.debug(f"Generating new embedding for query: {text[:30]}...")
        embedding = self.embedder.embed_query(text)
        print(f"🔍 DEBUG [Embeddings.embed_query]: New embedding generated, length: {len(embedding)}")
        
        # Cache the embedding
        print(f"🔍 DEBUG [Embeddings.embed_query]: Caching the embedding")
        self._cache_embedding(text, embedding)
        print(f"🔍 DEBUG [Embeddings.embed_query]: Embedding cached successfully")
        
        return embedding
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for documents with caching
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        results = []
        texts_to_embed = []
        positions = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached_embedding = self._get_cached_embedding(text)
            if cached_embedding is not None:
                # If found in cache, add directly to results at correct position
                results.append((i, cached_embedding))
            else:
                # If not found, add to list to be embedded
                texts_to_embed.append(text)
                positions.append(i)
        
        # Sort results by position
        results.sort(key=lambda x: x[0])
        
        # If there are texts to embed, embed them
        if texts_to_embed:
            logger.debug(f"Generating new embeddings for {len(texts_to_embed)} documents")
            new_embeddings = self.embedder.embed_documents(texts_to_embed)
            
            # Cache the new embeddings
            for text, embedding in zip(texts_to_embed, new_embeddings):
                self._cache_embedding(text, embedding)
            
            # Add new embeddings to results at correct positions
            for pos, embedding in zip(positions, new_embeddings):
                results.append((pos, embedding))
        
        # Sort results by position and extract just the embeddings
        results.sort(key=lambda x: x[0])
        return [embedding for _, embedding in results]
    
    def get_dimension(self) -> int:
        """Get the dimensionality of the embeddings"""
        return self.embedder.get_dimension()