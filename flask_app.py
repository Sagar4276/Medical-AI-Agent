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
from corpus import GlaucomaCorpusBuilder, CorpusBuilderConfig, APIAuthManager

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

@app.route('/api/corpus/build', methods=['POST'])
def build_corpus():
    """Build glaucoma research corpus endpoint"""
    try:
        data = request.get_json() or {}
        
        # Create configuration from request
        config = CorpusBuilderConfig(
            enabled_sources=data.get('sources', ['pubmed']),
            max_results_per_source=data.get('max_results', 100),
            search_years_back=data.get('years_back', 5),
            min_relevance_score=data.get('min_relevance', 15.0),
            min_quality_score=data.get('min_quality', 0.5),
            remove_duplicates=data.get('remove_duplicates', True),
            extract_entities=data.get('extract_entities', True),
            calculate_statistics=data.get('calculate_statistics', True),
            output_directory=data.get('output_dir', './data/glaucoma_corpus'),
            export_formats=data.get('formats', ['json']),
            include_metadata=data.get('include_metadata', True),
            max_concurrent_requests=data.get('concurrent_requests', 3),
            request_delay=data.get('request_delay', 1.0),
            timeout_seconds=data.get('timeout', 30)
        )
        
        # Build corpus
        with GlaucomaCorpusBuilder(config) as builder:
            result = builder.build_corpus(
                custom_search_terms=data.get('custom_terms'),
                subtopics=data.get('subtopics')
            )
            
            return jsonify({
                'success': result.success,
                'total_documents': result.total_documents,
                'processed_documents': result.processed_documents,
                'filtered_documents': result.filtered_documents,
                'sources_used': result.sources_used,
                'processing_time': result.processing_time,
                'output_files': result.output_files,
                'error_summary': result.error_summary,
                'recommendations': result.recommendations,
                'statistics': result.statistics.__dict__ if result.statistics else None
            })
            
    except Exception as e:
        logger.error(f"Corpus building failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }), 500

@app.route('/api/corpus/status')
def corpus_status():
    """Get corpus builder status and health"""
    try:
        auth_manager = APIAuthManager()
        
        # Get available APIs
        available_apis = auth_manager.list_available_apis()
        
        # Test API connections
        api_status = {}
        for api in available_apis:
            if api['available']:
                try:
                    is_healthy = auth_manager.test_credential(api['name'])
                    api_status[api['name']] = {
                        'configured': True,
                        'healthy': is_healthy,
                        'description': api.get('description', '')
                    }
                except Exception as e:
                    api_status[api['name']] = {
                        'configured': True,
                        'healthy': False,
                        'error': str(e),
                        'description': api.get('description', '')
                    }
            else:
                api_status[api['name']] = {
                    'configured': False,
                    'healthy': False,
                    'description': api.get('description', ''),
                    'env_key': api.get('env_key', '')
                }
        
        return jsonify({
            'success': True,
            'api_status': api_status,
            'total_apis': len(available_apis),
            'configured_apis': sum(1 for api in available_apis if api['available']),
            'healthy_apis': sum(1 for status in api_status.values() if status.get('healthy', False))
        })
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/corpus')
def corpus_interface():
    """Serve corpus builder interface"""
    return render_template('corpus_builder.html')

if __name__ == '__main__':
    print("🏥 Starting Medical RAG Flask Server...")
    
    # Initialize RAG system
    if initialize_rag():
        # Get port from environment (for hosting services like Render)
        port = int(os.environ.get('PORT', 5000))
        debug_mode = os.environ.get('FLASK_ENV') != 'production'
        
        print(f"🚀 Starting Flask server on port {port}")
        app.run(debug=debug_mode, host='0.0.0.0', port=port)
    else:
        print("❌ Cannot start server - RAG system initialization failed")
