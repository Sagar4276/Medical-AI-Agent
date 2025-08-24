#!/usr/bin/env python3
"""
COMPLETE SYSTEM RESET
Reset the medical RAG system to first-time setup state
"""

import os
import shutil
from pathlib import Path

def reset_system():
    """Reset the entire system to first-time state"""
    
    print("🔄 MEDICAL RAG SYSTEM RESET")
    print("=" * 50)
    print("⚠️  This will reset everything to first-time setup state!")
    
    # Confirm reset
    confirm = input("\n❓ Are you sure you want to reset? (type 'yes' to confirm): ")
    if confirm.lower() != 'yes':
        print("❌ Reset cancelled.")
        return
    
    print("\n🧹 Starting complete system reset...")
    
    # 1. Remove all generated data
    directories_to_remove = [
        "data/vectorstore",    # Vector database
        "data/cache",          # Cache files
        "logs",                # Log files
        "__pycache__",         # Python cache
        "rag/__pycache__",     # RAG module cache
        "utils/__pycache__",   # Utils module cache
    ]
    
    print("\n📂 Removing generated data directories...")
    for dir_path in directories_to_remove:
        if os.path.exists(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"   ✅ Removed: {dir_path}")
            except Exception as e:
                print(f"   ❌ Failed to remove {dir_path}: {e}")
        else:
            print(f"   ⚪ Not found: {dir_path}")
    
    # 2. Remove legacy/old files
    legacy_files_to_remove = [
        "app.py",                          # Old Streamlit app
        "enhanced_lightweight_rag.py",     # Old RAG version
        "medical_chat_app.py",             # Old chat app
        "medical_rag.py",                  # Old RAG version
        "ask_medical.py",                  # Old interface
        "launch_app.py",                   # Old launcher
        "main.py",                         # Might be old main
        "working_rag.py",                  # Old working version
    ]
    
    print("\n🗑️  Removing legacy/old files...")
    for file_path in legacy_files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"   ✅ Removed: {file_path}")
            except Exception as e:
                print(f"   ❌ Failed to remove {file_path}: {e}")
        else:
            print(f"   ⚪ Not found: {file_path}")
    
    # 3. Remove test/utility files (keep only essential ones)
    utility_files_to_keep = {
        "test_custom_embeddings.py",      # Essential test
        "initialize_database.py",         # Essential setup
    }
    
    utility_files_to_remove = [
        "analyze_files.py",
        "analyze_active_vs_legacy.py",
        "count_py_files.py",
        "file_classification.py",
        "file_status_report.py",
        "test_speed.py",
    ]
    
    print("\n🧪 Removing utility/analysis files...")
    for file_path in utility_files_to_remove:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"   ✅ Removed: {file_path}")
            except Exception as e:
                print(f"   ❌ Failed to remove {file_path}: {e}")
        else:
            print(f"   ⚪ Not found: {file_path}")
    
    # 4. Remove entire tests directory
    if os.path.exists("tests"):
        try:
            shutil.rmtree("tests")
            print(f"   ✅ Removed: tests/ directory")
        except Exception as e:
            print(f"   ❌ Failed to remove tests/: {e}")
    
    # 5. Create fresh data directories
    print("\n📁 Creating fresh data directories...")
    fresh_directories = [
        "data/documents",
        "data/vectorstore", 
        "data/cache",
        "logs",
    ]
    
    for dir_path in fresh_directories:
        try:
            os.makedirs(dir_path, exist_ok=True)
            print(f"   ✅ Created: {dir_path}")
        except Exception as e:
            print(f"   ❌ Failed to create {dir_path}: {e}")
    
    # 6. Create README for fresh setup
    readme_content = """# Medical RAG System - Fresh Setup

## 🚀 First Time Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Environment Variables
Create a `.env` file with:
```
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Add Medical Documents
- Place your medical text files in `data/documents/`
- Supported formats: .txt files

### 4. Initialize Database
```bash
python initialize_database.py
```

### 5. Run the System
```bash
python proper_medical_rag.py
```

## 📂 Core Files Structure
- `proper_medical_rag.py` - Main RAG system
- `rag/` - Core RAG components
- `utils/` - Utility functions
- `data/` - Data storage
- `initialize_database.py` - Database setup
- `test_custom_embeddings.py` - Test embeddings

## 🎯 System Features
- Custom lightweight embeddings (no pretrained models)
- ChromaDB vector storage
- Medical document processing
- Semantic search and retrieval
- Full tracking and logging
"""
    
    try:
        with open("README_FRESH_SETUP.md", "w") as f:
            f.write(readme_content)
        print(f"   ✅ Created: README_FRESH_SETUP.md")
    except Exception as e:
        print(f"   ❌ Failed to create README: {e}")
    
    # 7. Summary
    print("\n" + "=" * 50)
    print("🎉 SYSTEM RESET COMPLETE!")
    print("\n📊 What remains (Core System):")
    
    core_files = [
        "proper_medical_rag.py",
        "rag/embeddings.py",
        "rag/custom_embeddings.py", 
        "rag/vectorstore.py",
        "rag/retriever.py",
        "rag/generator.py",
        "rag/chunker.py",
        "utils/cache.py",
        "utils/logger.py",
        "initialize_database.py",
        "test_custom_embeddings.py",
        "requirements.txt",
        ".env (if exists)",
    ]
    
    for file in core_files:
        if os.path.exists(file):
            print(f"   ✅ {file}")
        else:
            print(f"   ⚪ {file} (missing)")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Check your .env file has GROQ_API_KEY")
    print("2. Add medical documents to data/documents/")
    print("3. Run: python initialize_database.py")
    print("4. Run: python proper_medical_rag.py")
    print("\n💡 Your system is now ready for first-time setup!")

if __name__ == "__main__":
    reset_system()
