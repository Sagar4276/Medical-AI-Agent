print(f"🔍 TRACKING: File = c:\\Users\\Sagar\\Desktop\\medical-rag\\rag\\retriever.py | Starting file execution")

from typing import List, Dict, Any, Optional, Union, Callable
import re

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers.document_compressors import EmbeddingsFilter

from pydantic import BaseModel, Field

from utils.logger import get_logger
from rag.vectorstore import MedicalVectorStore
from utils.cache import RAGCache

logger = get_logger(__name__)

class MedicalQueryPreprocessor:
    """Process medical queries to improve retrieval effectiveness"""
    
    def __init__(self):
        """Initialize medical query preprocessor"""
        # Common medical abbreviations and their expansions
        self.medical_abbreviations = {
            "MI": "myocardial infarction",
            "HTN": "hypertension",
            "DM": "diabetes mellitus",
            "CHF": "congestive heart failure",
            "COPD": "chronic obstructive pulmonary disease",
            "CVA": "cerebrovascular accident",
            "TIA": "transient ischemic attack",
            "UTI": "urinary tract infection",
            "ARF": "acute renal failure",
            "CKD": "chronic kidney disease",
            # Add more abbreviations as needed
        }
    
    def expand_medical_abbreviations(self, query: str) -> str:
        """Expand medical abbreviations in the query"""
        expanded_query = query
        
        # Find all words that might be abbreviations (all caps)
        abbreviations = re.findall(r'\b[A-Z]{2,}\b', query)
        
        # Replace abbreviations with their expansions
        for abbr in abbreviations:
            if abbr in self.medical_abbreviations:
                expanded_query = re.sub(
                    fr'\b{abbr}\b', 
                    f"{abbr} ({self.medical_abbreviations[abbr]})", 
                    expanded_query
                )
        
        return expanded_query
    
    def process_query(self, query: str) -> str:
        """Process a query to improve retrieval"""
        # Expand medical abbreviations
        query = self.expand_medical_abbreviations(query)
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        return query

class EnhancedMedicalRetriever(BaseRetriever, BaseModel):
    """
    Enhanced retriever for medical documents with caching and
    medical-specific query processing
    """
    # Declare all fields properly for Pydantic
    vectorstore: MedicalVectorStore = Field(..., description="Vector store to retrieve from")
    k: int = Field(default=4, description="Number of documents to retrieve")
    filter: Optional[Dict[str, Any]] = Field(default=None, description="Filter to apply to retrieval")
    cache: Optional[RAGCache] = Field(default=None, description="Cache instance")
    cache_dir: str = Field(default="data/cache", description="Directory for cache storage")
    preprocessor: Optional[MedicalQueryPreprocessor] = Field(default=None, description="Query preprocessor")
    reranking_enabled: bool = Field(default=True, description="Whether to enable reranking")
    relevance_threshold: float = Field(default=0.7, description="Threshold for relevance filtering")
    base_retriever: Any = Field(default=None, description="Base retriever")
    retriever: Any = Field(default=None, description="Enhanced retriever with reranking")
    
    class Config:
        arbitrary_types_allowed = True
    
    def model_post_init(self, __context):
        """Initialize complex components after basic initialization"""
        # Initialize cache if not provided
        if self.cache is None:
            self.cache = RAGCache(cache_dir=self.cache_dir)
        
        # Initialize preprocessor if not provided
        if self.preprocessor is None:
            self.preprocessor = MedicalQueryPreprocessor()
        
        # Create base retriever
        self.base_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.k * 2 if self.reranking_enabled else self.k, "filter": self.filter}
        )
        
        # Set up enhanced retrieval with reranking if enabled
        if self.reranking_enabled:
            # Create embeddings filter for reranking
            embeddings_filter = EmbeddingsFilter(
                embeddings=self.vectorstore.embedding_model,
                similarity_threshold=self.relevance_threshold,
                k=self.k
            )
            
            # Create pipeline for document compression
            pipeline = DocumentCompressorPipeline(
                transformers=[embeddings_filter]
            )
            
            # Create contextual compression retriever
            self.retriever = ContextualCompressionRetriever(
                base_compressor=pipeline,
                base_retriever=self.base_retriever
            )
        else:
            self.retriever = self.base_retriever
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Get relevant documents for a query
        
        Args:
            query: Query text
            run_manager: Callback manager
            
        Returns:
            List of relevant documents
        """
        print(f"🔍 DEBUG [Retriever._get_relevant_documents]: Starting retrieval for query: '{query[:50]}...'")
        
        # Process the query for better medical relevance
        print(f"🔍 DEBUG [Retriever._get_relevant_documents]: Processing query...")
        processed_query = self.preprocessor.process_query(query)
        print(f"🔍 DEBUG [Retriever._get_relevant_documents]: Query processed")
        
        logger.debug(f"Original query: {query}")
        logger.debug(f"Processed query: {processed_query}")
        
        # Check cache first
        print(f"🔍 DEBUG [Retriever._get_relevant_documents]: Checking cache...")
        cached_results = self.cache.get_query_result(processed_query)
        if cached_results is not None:
            print(f"🔍 DEBUG [Retriever._get_relevant_documents]: Found cached results")
            logger.debug(f"Using cached results for query: {processed_query[:30]}...")
            return cached_results
        
        print(f"🔍 DEBUG [Retriever._get_relevant_documents]: No cached results, retrieving from vectorstore...")
        
        # Retrieve documents
        if run_manager:
            print(f"🔍 DEBUG [Retriever._get_relevant_documents]: Using run_manager for retrieval")
            docs = self.retriever.invoke(
                processed_query, config={"callbacks": [run_manager.get_child()]}
            )
        else:
            print(f"🔍 DEBUG [Retriever._get_relevant_documents]: Direct retrieval call")
            docs = self.retriever.invoke(processed_query)
        
        print(f"🔍 DEBUG [Retriever._get_relevant_documents]: Retrieved {len(docs) if docs else 0} documents")
        
        # Cache the results
        print(f"🔍 DEBUG [Retriever._get_relevant_documents]: Caching results...")
        self.cache.cache_query_result(processed_query, docs)
        print(f"🔍 DEBUG [Retriever._get_relevant_documents]: Results cached")
        
        return docs
    
    def invoke(self, query: str, config=None) -> List[Document]:
        """
        Invoke the retriever (compatibility method)
        
        Args:
            query: Query text
            config: Configuration dict
            
        Returns:
            List of relevant documents
        """
        print(f"🔍 DEBUG [Retriever.invoke]: Invoke called with query: '{query[:50]}...'")
        result = self._get_relevant_documents(query)
        print(f"🔍 DEBUG [Retriever.invoke]: Invoke completed, returning {len(result) if result else 0} documents")
        return result