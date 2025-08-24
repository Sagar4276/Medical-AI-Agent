#!/usr/bin/env python3
"""
Initialize Vector Database with Custom Embeddings
Load medical documents and create fresh vector database
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.embeddings import CachedMedicalEmbeddings
from rag.vectorstore import MedicalVectorStore
from rag.chunker import MedicalDocumentChunker
from langchain_core.documents import Document

def load_medical_documents():
    """Load medical documents from data/documents"""
    docs = []
    docs_dir = Path("data/documents")
    
    if not docs_dir.exists():
        print("⚠️  No documents directory found")
        return docs
    
    for file_path in docs_dir.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    doc = Document(
                        page_content=content,
                        metadata={"source": str(file_path), "filename": file_path.name}
                    )
                    docs.append(doc)
                    print(f"✅ Loaded: {file_path.name}")
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    
    return docs

def initialize_vector_database():
    """Initialize vector database with custom embeddings"""
    print("🔧 Initializing Vector Database with Custom Embeddings")
    print("=" * 60)
    
    # Initialize custom embeddings
    print("\n1. Creating custom medical embeddings...")
    embeddings = CachedMedicalEmbeddings(
        embedding_type="simple",  # Use simple custom embeddings
        embedding_dim=128,
        cache_dir="data/cache"
    )
    print("✅ Custom embeddings ready")
    
    # Initialize vector store
    print("\n2. Creating vector store...")
    vectorstore = MedicalVectorStore(
        embedding_model=embeddings,
        persist_directory="data/vectorstore",
        collection_name="medical_documents"
    )
    print("✅ Vector store ready")
    
    # Load documents
    print("\n3. Loading medical documents...")
    documents = load_medical_documents()
    
    if not documents:
        print("⚠️  No documents found, creating sample document...")
        # Create a sample medical document
        sample_doc = Document(
            page_content="""
            Heart Disease Overview
            
            Heart disease is a term used to describe several types of heart conditions. The most common type of heart disease is coronary artery disease, which affects blood flow to the heart.
            
            Symptoms:
            - Chest pain or discomfort
            - Shortness of breath
            - Pain in neck, jaw, throat, upper abdomen or back
            - Pain, numbness, weakness or coldness in legs or arms
            
            Risk Factors:
            - High blood pressure
            - High cholesterol
            - Diabetes
            - Smoking
            - Obesity
            - Physical inactivity
            
            Treatment:
            Treatment for heart disease includes lifestyle changes, medications, and sometimes surgery. Early detection and treatment can significantly improve outcomes.
            
            Prevention:
            - Maintain a healthy diet
            - Exercise regularly
            - Don't smoke
            - Limit alcohol consumption
            - Manage stress
            - Get regular checkups
            """,
            metadata={"source": "sample_heart_disease.txt", "filename": "heart_disease_info.txt"}
        )
        documents = [sample_doc]
        print("✅ Sample document created")
    
    print(f"\n4. Processing {len(documents)} documents...")
    
    # Chunk documents for better retrieval
    chunker = MedicalDocumentChunker(chunk_size=500, chunk_overlap=50)
    chunked_docs = []
    
    for doc in documents:
        chunks = chunker.chunk_documents([doc])
        chunked_docs.extend(chunks)
        print(f"   📄 {doc.metadata.get('filename', 'Unknown')}: {len(chunks)} chunks")
    
    print(f"✅ Total chunks: {len(chunked_docs)}")
    
    # Add documents to vector store
    print("\n5. Adding documents to vector store...")
    doc_ids = vectorstore.add_documents(chunked_docs)
    print(f"✅ Added {len(doc_ids)} document chunks to vector store")
    
    # Get stats
    print("\n6. Vector store statistics:")
    stats = vectorstore.get_collection_stats()
    for key, value in stats.items():
        print(f"   📊 {key}: {value}")
    
    print("\n🎉 Vector database initialization complete!")
    print("✅ Ready to use with proper_medical_rag.py")
    
    return vectorstore

if __name__ == "__main__":
    initialize_vector_database()
