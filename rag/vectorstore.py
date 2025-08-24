print(f"🔍 TRACKING: File = c:\\Users\\Sagar\\Desktop\\medical-rag\\rag\\vectorstore.py | Starting file execution")

from typing import List, Dict, Any, Optional, Tuple
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

from rag.embeddings import CachedMedicalEmbeddings
from utils.logger import get_logger

logger = get_logger(__name__)

class MedicalVectorStore:
    """Vector store for medical documents with specialized embedding model"""
    
    def __init__(
        self,
        embedding_model: Optional[CachedMedicalEmbeddings] = None,
        persist_directory: str = "data/vectorstore",
        collection_name: str = "medical_documents"
    ):
        """
        Initialize the medical vector store
        
        Args:
            embedding_model: Embedding model to use
            persist_directory: Directory to persist vector store
            collection_name: Name of the collection
        """
        print(f"🔍 TRACKING: File = c:\\Users\\Sagar\\Desktop\\medical-rag\\rag\\vectorstore.py | Function = MedicalVectorStore.__init__")
        print(f"🔍 DEBUG [VectorStore.__init__]: Starting vectorstore initialization")
        print(f"🔍 DEBUG [VectorStore.__init__]: persist_directory: {persist_directory}")
        print(f"🔍 DEBUG [VectorStore.__init__]: collection_name: {collection_name}")
        
        # Initialize embedding model if not provided
        if embedding_model is None:
            print(f"🔍 DEBUG [VectorStore.__init__]: Creating new embedding model")
            embedding_model = CachedMedicalEmbeddings()
        else:
            print(f"🔍 DEBUG [VectorStore.__init__]: Using provided embedding model")
        
        self.embedding_model = embedding_model
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Create directory if it doesn't exist
        print(f"🔍 DEBUG [VectorStore.__init__]: Creating directory if not exists")
        os.makedirs(persist_directory, exist_ok=True)
        print(f"🔍 DEBUG [VectorStore.__init__]: Directory created/verified")
        
        # Initialize vector store
        db_path = os.path.join(persist_directory, "chroma.sqlite3")
        print(f"🔍 DEBUG [VectorStore.__init__]: Checking for existing DB at: {db_path}")
        
        if os.path.exists(db_path):
            print(f"🔍 DEBUG [VectorStore.__init__]: Found existing database, loading...")
            logger.info(f"Loading existing Chroma database from {persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_model,
                collection_name=collection_name
            )
            print(f"🔍 DEBUG [VectorStore.__init__]: Existing database loaded successfully")
        else:
            print(f"🔍 DEBUG [VectorStore.__init__]: No existing database found, creating new...")
            logger.info(f"Creating new Chroma database at {persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embedding_model,
                collection_name=collection_name
            )
            print(f"🔍 DEBUG [VectorStore.__init__]: New database created successfully")
        
        print(f"🔍 DEBUG [VectorStore.__init__]: VectorStore initialization completed")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        Add documents to the vector store
        
        Args:
            documents: Documents to add
            
        Returns:
            List of document IDs
        """
        logger.info(f"Adding {len(documents)} documents to vector store")
        return self.vectorstore.add_documents(documents)
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[Document, float]]:
        """
        Search for similar documents with scores
        
        Args:
            query: Query text
            k: Number of results to return
            filter: Filter to apply
            
        Returns:
            List of (document, score) tuples
        """
        return self.vectorstore.similarity_search_with_score(
            query, k=k, filter=filter
        )
    
    def as_retriever(self, search_type: str = "similarity", search_kwargs: Optional[Dict[str, Any]] = None):
        """
        Create a retriever from the vector store
        
        Args:
            search_type: Type of search
            search_kwargs: Search parameters
            
        Returns:
            Retriever
        """
        print(f"🔍 DEBUG [VectorStore.as_retriever]: Creating retriever with search_type: {search_type}")
        print(f"🔍 DEBUG [VectorStore.as_retriever]: search_kwargs: {search_kwargs}")
        
        retriever = self.vectorstore.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
        
        print(f"🔍 DEBUG [VectorStore.as_retriever]: Retriever created successfully")
        return retriever
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        try:
            return {
                "document_count": len(self.vectorstore.get()),
                "embedding_dimension": self.embedding_model.get_dimension()
            }
        except:
            return {"document_count": 0, "embedding_dimension": 0}