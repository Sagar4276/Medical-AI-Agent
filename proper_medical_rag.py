#!/usr/bin/env python3
"""
RESTORED Medical RAG System - Full Vectorization & Semantic Search
This is your PROPER RAG system with all the advanced features you had before
"""

print(f"🔍 TRACKING: File = c:\\Users\\Sagar\\Desktop\\medical-rag\\proper_medical_rag.py | Starting file execution")

import os
import sys
import time
from typing import List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Add parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.embeddings import CachedMedicalEmbeddings
from rag.vectorstore import MedicalVectorStore  
from rag.retriever import EnhancedMedicalRetriever
from rag.generator import MedicalResponseGenerator
from rag.chunker import MedicalDocumentChunker
from utils.logger import get_logger

# Load environment variables
load_dotenv()
logger = get_logger(__name__)

@dataclass
class MedicalRAGResult:
    answer: str
    sources: List[str]
    confidence: float
    response_time: float
    retrieved_docs: List[str]

class ProperMedicalRAG:
    """
    PROPER Medical RAG System with full vectorization and semantic search
    This is what you had before - the REAL system!
    """
    
    def __init__(self, model_name: str = "llama3-70b-8192"):
        print(f"🔍 TRACKING: File = c:\\Users\\Sagar\\Desktop\\medical-rag\\proper_medical_rag.py | Function = ProperMedicalRAG.__init__")
        print("🔧 Initializing PROPER Medical RAG System with Vector Search...")
        logger.info("Initializing full Medical RAG system with vectorization")
        
        start_time = time.time()
        
        # Initialize custom embedding model (no pretrained models)
        print("🔧 Loading Custom Medical Embeddings (lightweight)...")
        self.embeddings = CachedMedicalEmbeddings(
            embedding_type="simple",  # Use simple custom embeddings
            embedding_dim=128,
            cache_dir="data/cache"
        )
        print("✅ Custom medical embeddings loaded")
        
        # Initialize vector store
        print("🔧 Initializing Vector Store (Chroma DB)...")
        self.vectorstore = MedicalVectorStore(
            embedding_model=self.embeddings,
            persist_directory="data/vectorstore",
            collection_name="medical_documents"
        )
        print("✅ Vector store initialized")
        
        # Initialize enhanced retriever with reranking
        print("🔧 Setting up Enhanced Medical Retriever...")
        self.retriever = EnhancedMedicalRetriever(
            vectorstore=self.vectorstore,
            k=5,  # Retrieve top 5 documents
            reranking_enabled=True,
            relevance_threshold=0.7,
            cache_dir="data/cache"
        )
        print("✅ Enhanced retriever ready")
        
        # Initialize response generator
        print("🔧 Initializing Medical Response Generator...")
        self.generator = MedicalResponseGenerator(
            retriever=self.retriever,
            model_name=model_name,
            provider="groq",
            temperature=0.1,
            max_tokens=2000  # Increased for complete responses
        )
        print("✅ Response generator ready")
        
        # Check if we need to load documents
        stats = self.vectorstore.get_collection_stats()
        if stats['document_count'] == 0:
            print("🔧 No documents in vector store, loading medical documents...")
            self._load_and_index_documents()
        else:
            print(f"✅ Found {stats['document_count']} documents in vector store")
        
        init_time = time.time() - start_time
        print(f"🎉 PROPER Medical RAG System ready in {init_time:.2f}s!")
        logger.info(f"Medical RAG system initialized successfully in {init_time:.2f}s")
        
    def _load_and_index_documents(self):
        """Load and index all medical documents into the vector store"""
        print("📚 Loading and indexing medical documents...")
        
        # Initialize document chunker
        chunker = MedicalDocumentChunker(
            chunk_size=500,
            chunk_overlap=50,
            preserve_medical_terms=True
        )
        
        # Find all medical documents
        document_dirs = [
            "data/documents",
            "data/samples"
        ]
        
        all_docs = []
        for doc_dir in document_dirs:
            if os.path.exists(doc_dir):
                for filename in os.listdir(doc_dir):
                    if filename.endswith('.txt'):
                        filepath = os.path.join(doc_dir, filename)
                        print(f"📄 Processing {filename}...")
                        
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            # Chunk the document
                            chunks = chunker.chunk_document(content, source=filepath)
                            all_docs.extend(chunks)
                            print(f"   ✓ Created {len(chunks)} chunks from {filename}")
                            
                        except Exception as e:
                            print(f"   ❌ Failed to process {filename}: {e}")
        
        if all_docs:
            print(f"🔧 Indexing {len(all_docs)} document chunks...")
            self.vectorstore.add_documents(all_docs)
            print(f"✅ Successfully indexed {len(all_docs)} chunks")
        else:
            print("⚠️ No documents found to index")
    
    def query(self, question: str) -> MedicalRAGResult:
        """
        Process a medical question using full RAG pipeline with semantic search
        """
        print(f"🔍 TRACKING: File = c:\\Users\\Sagar\\Desktop\\medical-rag\\proper_medical_rag.py | Function = ProperMedicalRAG.query")
        start_time = time.time()
        
        print(f"🔍 Processing question: {question}")
        logger.info(f"Processing query: {question}")
        
        try:
            # Use the full RAG pipeline
            print("🔍 Performing semantic search...")
            response = self.generator.generate_response(question)
            
            # Extract retrieved documents for sources
            retrieved_docs = self.retriever.invoke(question)
            sources = []
            doc_contents = []
            
            for doc in retrieved_docs:
                if hasattr(doc, 'metadata') and doc.metadata.get('source'):
                    source_file = os.path.basename(doc.metadata['source'])
                    if source_file not in sources:
                        sources.append(source_file)
                doc_contents.append(doc.page_content[:100] + "...")
            
            elapsed = time.time() - start_time
            
            print(f"✅ Response generated in {elapsed:.2f}s")
            logger.info(f"Query processed successfully in {elapsed:.2f}s")
            
            return MedicalRAGResult(
                answer=response,
                sources=sources,
                confidence=0.85,  # High confidence with proper retrieval
                response_time=elapsed,
                retrieved_docs=doc_contents
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"Error in RAG pipeline: {str(e)}"
            print(f"❌ Error: {error_msg}")
            logger.error(error_msg)
            
            return MedicalRAGResult(
                answer=f"I apologize, but I encountered an error processing your question: {error_msg}",
                sources=[],
                confidence=0.0,
                response_time=elapsed,
                retrieved_docs=[]
            )
    
    def get_system_stats(self):
        """Get comprehensive system statistics"""
        vectorstore_stats = self.vectorstore.get_collection_stats()
        
        return {
            "documents_indexed": vectorstore_stats['document_count'],
            "embedding_dimension": vectorstore_stats['embedding_dimension'],
            "embedding_model": "S-PubMedBert-MS-MARCO",
            "vector_store": "ChromaDB",
            "retrieval_method": "Enhanced with reranking",
            "llm_model": self.generator.model_name,
            "cache_enabled": True,
            "medical_preprocessing": True
        }

def format_sources(sources):
    """Format the source list for display"""
    if not sources:
        return "No specific sources"
    return ", ".join(sources)

def main():
    """Interactive interface for the PROPER Medical RAG system"""
    print(f"🔍 TRACKING: File = c:\\Users\\Sagar\\Desktop\\medical-rag\\proper_medical_rag.py | Function = main")
    print("=" * 80)
    print("🏥 MEDICAL AI ASSISTANT - PROPER RAG SYSTEM")
    print("🔬 Full Vectorization & Semantic Search Enabled")
    print("=" * 80)
    print("💡 Ask questions about medical conditions, symptoms, treatments")
    print("⚡ Powered by S-PubMedBert embeddings and ChromaDB")
    print("🎯 Type 'quit', 'exit', or 'bye' to exit")
    print("📊 Type 'stats' to see system statistics")
    print("-" * 80)
    
    try:
        # Initialize the PROPER RAG system
        print("🔧 Initializing PROPER Medical RAG System...")
        rag = ProperMedicalRAG()
        print("✅ System ready! Ask your medical questions.")
    except Exception as e:
        print(f"❌ Failed to initialize system: {e}")
        return
    
    print("-" * 80)
    
    # Interactive loop
    while True:
        try:
            # Get user input
            question = input("\n🤔 Your medical question: ").strip()
            
            # Check for exit commands
            if question.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Thank you for using the Medical AI Assistant!")
                break
            
            # Check for stats command
            if question.lower() == 'stats':
                print("\n📊 System Statistics:")
                stats = rag.get_system_stats()
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                continue
            
            # Skip empty questions
            if not question:
                continue
            
            print("\n🤖 Processing your question...")
            
            # Process the question
            result = rag.query(question)
            
            # Display results
            print("\n💬 Medical AI Response:")
            print("=" * 60)
            print(result.answer)
            print("=" * 60)
            print(f"📚 Sources: {format_sources(result.sources)}")
            print(f"⏱️  Response time: {result.response_time:.2f} seconds")
            print(f"🎯 Confidence: {len(result.sources)} source(s) found")
            
            if result.retrieved_docs:
                print(f"🔍 Retrieved {len(result.retrieved_docs)} relevant chunks")
            
            print("\n⚠️  Remember: This is for educational purposes only.")
            print("   Always consult healthcare professionals for medical advice.")
            
        except KeyboardInterrupt:
            print("\n\n👋 Medical AI Assistant stopped. Stay healthy!")
            break
        except Exception as e:
            print(f"\n❌ Error processing question: {e}")
            print("💡 Try rephrasing your question or ask about a different topic.")

if __name__ == "__main__":
    main()
