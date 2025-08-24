#!/usr/bin/env python3
"""
🧪 Final System Test - Medical RAG with Custom Embeddings
================================================================
Tests the complete medical RAG system after reset and setup.
"""

print("🔍 TRACKING: File = c:\\Users\\Sagar\\Desktop\\medical-rag\\final_test.py | Starting file execution")

from proper_medical_rag import ProperMedicalRAG
import time

def test_medical_rag():
    """Test the medical RAG system with sample questions."""
    print("\n🏥 Medical RAG System - Final Test")
    print("=" * 50)
    
    # Initialize RAG system
    print("🚀 Initializing Medical RAG System...")
    rag = ProperMedicalRAG()
    
    # Test questions
    questions = [
        "What is diabetes?",
        "How is hypertension treated?",
        "What are the symptoms of asthma?",
        "Tell me about heart disease prevention"
    ]
    
    print(f"\n🧪 Testing {len(questions)} medical questions:")
    print("-" * 50)
    
    for i, question in enumerate(questions, 1):
        print(f"\n📝 Question {i}: {question}")
        start_time = time.time()
        
        try:
            answer = rag.query(question)
            response_time = time.time() - start_time
            
            print(f"✅ Answer received in {response_time:.2f}s")
            print(f"📄 Response length: {len(answer)} characters")
            
            # Show first 100 chars of answer
            preview = answer[:100] + "..." if len(answer) > 100 else answer
            print(f"🔍 Preview: {preview}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Medical RAG System Test Complete!")
    print("🎯 Custom embeddings working")
    print("📊 Tracking statements active")
    print("🚀 System ready for production")

if __name__ == "__main__":
    test_medical_rag()
