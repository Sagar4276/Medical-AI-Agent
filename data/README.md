# Medical Data Sources for RAG

This directory contains medical datasets used by the Medical RAG system. You need to add your own medical datasets here for the system to function properly.

## Recommended Medical Datasets

1. **PubMed Abstracts**: Scientific paper abstracts from PubMed
   - Source: https://pubmed.ncbi.nlm.nih.gov/
   - Format: XML or JSON

2. **Medical Textbooks**: Public domain or licensed medical textbooks
   - Consider textbooks from OpenStax or other open educational resources

3. **MedlinePlus**: Consumer health information
   - Source: https://medlineplus.gov/

4. **UMLS (Unified Medical Language System)**: Medical terminology
   - Source: https://www.nlm.nih.gov/research/umls/index.html
   - Requires registration

5. **Medical Guidelines**: Clinical practice guidelines
   - Source: https://www.guideline.gov/

## Data Preparation

Place your medical data files in this directory using one of these formats:
- Plain text (.txt)
- Markdown (.md)
- PDF (.pdf)
- JSON (.json)
- CSV (.csv)

## Example Usage

To add your medical documents to the vector database:

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from app import MedicalRAG

# Initialize RAG system
rag = MedicalRAG()

# Load PDF documents
pdf_loader = DirectoryLoader("data/", glob="**/*.pdf", loader_cls=PyPDFLoader)
pdf_docs = pdf_loader.load()

# Load text documents
text_loader = DirectoryLoader("data/", glob="**/*.txt")
text_docs = text_loader.load()

# Add documents to vector store
rag.add_documents(pdf_docs + text_docs)
```

## Data Organization

Consider organizing your medical data into subdirectories by specialty or topic:
- `data/cardiology/`
- `data/oncology/`
- `data/pediatrics/`
- `data/pharmacology/`
- etc.