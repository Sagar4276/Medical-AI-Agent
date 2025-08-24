#!/usr/bin/env python3
"""
Flask Backend for Medical RAG System
Serves the beautiful HTML interface and processes medical queries
"""

from flask import Flask, render_template, request, jsonify
import os
import sys
import traceback
from typing import Dict, Any

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the RAG system
from proper_medical_rag import ProperMedicalRAG, MedicalRAGResult

app = Flask(__name__)

# Global RAG instance
rag_system = None

def initialize_rag():
    """Initialize the RAG system once on startup"""
    global rag_system
    try:
        print("🔧 Initializing Medical RAG System...")
        rag_system = ProperMedicalRAG()
        print("✅ RAG System initialized successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return False

@app.route('/')
def index():
    """Serve the main HTML interface"""
    return render_template('medical_chat.html')

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process medical queries via API"""
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'error': 'No question provided'
            }), 400
        
        if not rag_system:
            return jsonify({
                'success': False,
                'error': 'RAG system not initialized'
            }), 500
        
        # Process the question using the RAG system
        result = rag_system.query(question)
        
        return jsonify({
            'success': True,
            'answer': result.answer,
            'sources': result.sources,
            'confidence': result.confidence,
            'response_time': result.response_time,
            'retrieved_docs_count': len(result.retrieved_docs)
        })
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Internal server error: {str(e)}'
        }), 500

@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    try:
        if not rag_system:
            return jsonify({
                'success': False,
                'error': 'RAG system not initialized'
            })
        
        stats = rag_system.get_system_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'rag_initialized': rag_system is not None
    })

if __name__ == '__main__':
    print("🏥 Starting Medical RAG Flask Server...")
    
    # Initialize RAG system
    if initialize_rag():
        print("🚀 Starting Flask server on http://localhost:5000")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("❌ Cannot start server - RAG system initialization failed")
