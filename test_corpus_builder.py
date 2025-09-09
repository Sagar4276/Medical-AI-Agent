#!/usr/bin/env python3
"""
Test script for Glaucoma Corpus Builder
Basic functionality testing without requiring API keys
"""

import os
import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from corpus import GlaucomaCorpusBuilder, CorpusBuilderConfig, APIAuthManager
from corpus.data_processor import GlaucomaDataProcessor
from corpus.error_manager import ErrorManager

def setup_logging():
    """Set up basic logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_auth_manager():
    """Test the authentication manager"""
    print("🔐 Testing Authentication Manager...")
    
    try:
        auth_manager = APIAuthManager()
        
        # List available APIs
        apis = auth_manager.list_available_apis()
        print(f"   ✅ Found {len(apis)} available APIs")
        
        for api in apis:
            status = "✅ Configured" if api['available'] else "❌ Not configured"
            print(f"   - {api['name']}: {status}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_data_processor():
    """Test the data processor"""
    print("🔄 Testing Data Processor...")
    
    try:
        processor = GlaucomaDataProcessor()
        
        # Test with sample data
        sample_docs = [
            {
                "id": "test1",
                "title": "Primary Open-Angle Glaucoma Treatment",
                "abstract": "This study evaluates the effectiveness of prostaglandin analogs in treating primary open-angle glaucoma with elevated intraocular pressure.",
                "authors": ["Dr. Jane Smith", "Dr. John Doe"],
                "publication_date": "2023-01-15",
                "journal": "Journal of Glaucoma Research",
                "doi": "10.1234/jgr.2023.001",
                "source": "Test",
                "keywords": ["glaucoma", "prostaglandin analogs", "intraocular pressure"],
                "mesh_terms": ["Glaucoma, Open-Angle", "Intraocular Pressure"]
            },
            {
                "id": "test2", 
                "title": "Visual Field Testing in Glaucoma Diagnosis",
                "abstract": "A comprehensive review of perimetry techniques for early detection of glaucomatous visual field defects.",
                "authors": ["Dr. Sarah Johnson"],
                "publication_date": "2023-02-20",
                "journal": "Ophthalmology Today",
                "doi": "10.1234/ot.2023.002",
                "source": "Test",
                "keywords": ["visual field testing", "perimetry", "glaucoma diagnosis"],
                "mesh_terms": ["Visual Field Tests", "Glaucoma"]
            }
        ]
        
        # Process documents
        processed = processor.process_documents(sample_docs)
        print(f"   ✅ Processed {len(processed)} documents")
        
        # Generate statistics
        stats = processor.generate_statistics(processed)
        print(f"   ✅ Generated statistics for {stats.total_documents} documents")
        print(f"   - Average relevance score: {stats.average_relevance_score:.1f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_error_manager():
    """Test the error manager"""
    print("🚨 Testing Error Manager...")
    
    try:
        error_manager = ErrorManager()
        
        # Test error categorization
        test_error = Exception("Test error")
        error_details = error_manager.categorize_error(test_error)
        print(f"   ✅ Categorized error: {error_details.category.value}")
        
        # Test error summary
        summary = error_manager.get_error_summary()
        print(f"   ✅ Generated error summary: {summary['total_errors']} errors")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def test_corpus_builder():
    """Test the corpus builder basic functionality"""
    print("🏗️ Testing Corpus Builder...")
    
    try:
        # Create minimal configuration
        config = CorpusBuilderConfig(
            enabled_sources=[],  # No actual API calls
            max_results_per_source=10,
            search_years_back=1,
            min_relevance_score=0.0,
            min_quality_score=0.0,
            remove_duplicates=True,
            extract_entities=True,
            calculate_statistics=True,
            output_directory="/tmp/test_corpus",
            export_formats=["json"],
            include_metadata=True,
            max_concurrent_requests=1,
            request_delay=0.0,
            timeout_seconds=10
        )
        
        # Initialize builder (without API clients)
        builder = GlaucomaCorpusBuilder(config)
        print("   ✅ Initialized corpus builder")
        
        # Test health status (will show no APIs configured, which is expected)
        health = builder.get_health_status()
        print(f"   ✅ Retrieved health status: {len(health.get('client_status', {}))} clients")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Running Glaucoma Corpus Builder Tests\n")
    
    setup_logging()
    
    tests = [
        ("Authentication Manager", test_auth_manager),
        ("Data Processor", test_data_processor),
        ("Error Manager", test_error_manager),
        ("Corpus Builder", test_corpus_builder)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASSED\n")
            else:
                print(f"❌ {test_name}: FAILED\n")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}\n")
    
    # Summary
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Glaucoma Corpus Builder is ready to use.")
        print("\n💡 Next steps:")
        print("   1. Set up API keys in .env file")
        print("   2. Run: python corpus/glaucoma_corpus_cli.py setup --interactive")
        print("   3. Test APIs: python corpus/glaucoma_corpus_cli.py test")
        print("   4. Build corpus: python corpus/glaucoma_corpus_cli.py build")
        print("   5. Or use the web interface at: http://localhost:5000/corpus")
        
        return 0
    else:
        print("🚨 Some tests failed. Please check the errors above.")
        return 1

if __name__ == '__main__':
    sys.exit(main())