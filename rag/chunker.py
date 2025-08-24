from typing import List, Optional, Union, Dict, Any
import re
import nltk
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    SentenceTransformersTokenTextSplitter
)
from langchain.docstore.document import Document
from langchain_core.documents import Document as LangchainDocument

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class MedicalDocumentChunker:
    """
    Handles chunking of medical documents with various strategies
    optimized for medical content
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunking_strategy: str = "semantic",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the document chunker
        
        Args:
            chunk_size: Size of each chunk (characters or tokens depending on strategy)
            chunk_overlap: Overlap between chunks
            chunking_strategy: Strategy for chunking ("fixed", "semantic", "recursive")
            model_name: Name of the model for token counting (if using token-based splitting)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.model_name = model_name
        
        # Initialize the appropriate text splitter based on strategy
        if chunking_strategy == "fixed":
            self.text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len
            )
        elif chunking_strategy == "recursive":
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", ", ", " ", ""]
            )
        elif chunking_strategy == "semantic":
            # For semantic chunking, we'll use a token-based splitter
            # that's better at preserving semantic meaning
            self.text_splitter = SentenceTransformersTokenTextSplitter(
                model_name=model_name,
                chunk_size=chunk_size//4,  # Token counts are usually smaller than character counts
                chunk_overlap=chunk_overlap//4
            )
        else:
            raise ValueError(f"Unknown chunking strategy: {chunking_strategy}")
    
    def _preprocess_medical_text(self, text: str) -> str:
        """Preprocess medical text to improve chunking quality"""
        # Replace multiple newlines with double newline
        text = re.sub(r'\n+', '\n\n', text)
        
        # Ensure proper spacing after periods in abbreviations
        text = re.sub(r'(\w\.\w\.)', r'\1 ', text)
        
        # Make sure headers are properly separated
        text = re.sub(r'([A-Z][A-Z\s]+:)', r'\n\1', text)
        
        return text
    
    def _create_document_with_metadata(
        self, 
        chunk_text: str, 
        doc_metadata: Dict[str, Any],
        chunk_id: int
    ) -> LangchainDocument:
        """Create a Document object with metadata"""
        # Create a new metadata dict with chunk info
        metadata = doc_metadata.copy() if doc_metadata else {}
        metadata["chunk_id"] = chunk_id
        metadata["chunk_size"] = len(chunk_text)
        
        return LangchainDocument(page_content=chunk_text, metadata=metadata)
    
    def chunk_document(
        self, 
        document: Union[str, LangchainDocument],
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[LangchainDocument]:
        """
        Chunk a document into smaller pieces
        
        Args:
            document: Text content or Document object
            metadata: Metadata to associate with chunks
            
        Returns:
            List of Document objects with chunk metadata
        """
        # Extract text and metadata from document
        if isinstance(document, LangchainDocument):
            text = document.page_content
            doc_metadata = document.metadata.copy() if document.metadata else {}
            if metadata:
                doc_metadata.update(metadata)
        else:
            text = document
            doc_metadata = metadata or {}
        
        # Preprocess the text
        processed_text = self._preprocess_medical_text(text)
        
        # Split the document
        chunks = self.text_splitter.split_text(processed_text)
        
        # Create Document objects with metadata
        return [
            self._create_document_with_metadata(chunk, doc_metadata, i) 
            for i, chunk in enumerate(chunks)
        ]
    
    def chunk_documents(
        self, 
        documents: List[Union[str, LangchainDocument]],
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[LangchainDocument]:
        """
        Chunk multiple documents
        
        Args:
            documents: List of text contents or Document objects
            metadatas: List of metadata dicts (optional)
            
        Returns:
            List of Document objects with chunk metadata
        """
        all_chunks = []
        
        for i, doc in enumerate(documents):
            doc_metadata = metadatas[i] if metadatas and i < len(metadatas) else None
            chunks = self.chunk_document(doc, doc_metadata)
            all_chunks.extend(chunks)
            
        return all_chunks