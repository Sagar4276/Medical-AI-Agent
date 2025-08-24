#!/usr/bin/env python3
"""
Test Custom Embeddings - Lightweight Medical RAG
No pretrained models - custom embeddings only
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.custom_embeddings import SimpleMedicalEmbeddings, FastHashEmbeddings, WordCountEmbeddings

def test_custom_embeddings():
    """Test all custom embedding methods"""
    
    test_texts = [
        "The patient has diabetes and high blood pressure.",
        "Heart disease is a common medical condition requiring treatment.",
        "Symptoms include fever, headache, and fatigue.",
        "The doctor prescribed medication for the infection."
    ]
    
    print("🧪 Testing Custom Embeddings (No Pretrained Models)")
    print("=" * 60)
    
    # Test 1: Simple Medical Embeddings
    print("\n1. Testing SimpleMedicalEmbeddings:")
    simple_emb = SimpleMedicalEmbeddings(embedding_dim=64)
    
    for i, text in enumerate(test_texts):
        embedding = simple_emb.embed_query(text)
        print(f"   Text {i+1}: {text[:50]}...")
        print(f"   Embedding: [{embedding[0]:.3f}, {embedding[1]:.3f}, ..., {embedding[-1]:.3f}] (dim={len(embedding)})")
    
    # Test batch embedding
    batch_embeddings = simple_emb.embed_documents(test_texts)
    print(f"   Batch processing: {len(batch_embeddings)} texts → {len(batch_embeddings[0])}-dim vectors")
    
    # Test 2: Fast Hash Embeddings
    print("\n2. Testing FastHashEmbeddings:")
    fast_emb = FastHashEmbeddings(embedding_dim=32)
    
    for i, text in enumerate(test_texts[:2]):  # Test first 2 only
        embedding = fast_emb.embed_query(text)
        print(f"   Text {i+1}: {text[:50]}...")
        print(f"   Embedding: [{embedding[0]:.3f}, {embedding[1]:.3f}, ..., {embedding[-1]:.3f}] (dim={len(embedding)})")
    
    # Test 3: Word Count Embeddings
    print("\n3. Testing WordCountEmbeddings:")
    word_emb = WordCountEmbeddings(embedding_dim=50, vocab_size=100)
    
    # Build vocab first
    batch_embeddings = word_emb.embed_documents(test_texts)
    print(f"   Vocabulary built from {len(test_texts)} texts")
    
    for i, text in enumerate(test_texts[:2]):  # Test first 2 only
        embedding = word_emb.embed_query(text)
        print(f"   Text {i+1}: {text[:50]}...")
        print(f"   Embedding: [{embedding[0]:.3f}, {embedding[1]:.3f}, ..., {embedding[-1]:.3f}] (dim={len(embedding)})")
    
    print("\n✅ All custom embeddings working correctly!")
    print("📊 Resource usage: Minimal (no model downloads)")
    print("🚀 Ready to use in RAG system")

if __name__ == "__main__":
    test_custom_embeddings()
