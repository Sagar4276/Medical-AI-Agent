# 🏥 Medical AI Agent - Advanced RAG System

**A sophisticated Medical Retrieval-Augmented Generation (RAG) system featuring ChatGPT-style interface, custom embeddings, professional Flask web application, and comprehensive glaucoma research corpus builder for medical education and research.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)](https://flask.palletsprojects.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

## 🌟 **Live Demo**
Experience the AI-powered medical assistant with real-time typing animations and professional healthcare interface.

![Medical AI Chat Interface](https://via.placeholder.com/800x400/1e40af/ffffff?text=ChatGPT-Style+Medical+Interface)

## ✨ **Key Features**

🧠 **Custom Medical Embeddings** - Lightweight, no pretrained models required  
🌐 **ChatGPT-Style Interface** - Professional typing animations and modern UI  
🏥 **Medical Knowledge Base** - Comprehensive diabetes, hypertension, cardiovascular, and glaucoma data  
🔍 **Semantic Vector Search** - Intelligent document retrieval with ChromaDB  
⚡ **Sub-2s Response Time** - Optimized for speed and accuracy  
📱 **Mobile Responsive** - Beautiful interface on all devices  
🎯 **Educational Focus** - Perfect for medical students and healthcare professionals  
💾 **Persistent Caching** - Smart embedding and query caching system  
🏗️ **Glaucoma Corpus Builder** - Advanced research data extraction from multiple medical APIs  
🔐 **Multi-API Integration** - PubMed, ClinicalTrials.gov, OpenFDA, WHO, Europe PMC  
🛡️ **Enterprise Security** - Encrypted credential storage and robust error handling  
📊 **Research Analytics** - Comprehensive data processing and quality assessment  

## 🚀 **Quick Start**

### Prerequisites
- Python 3.8+
- Groq API Key ([Get free key](https://console.groq.com/))

### Installation
```bash
# Clone the repository
git clone https://github.com/Sagar4276/Medical-AI-Agent.git
cd Medical-AI-Agent

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Set up environment
copy .env.example .env
# Edit .env and add your GROQ_API_KEY

# Initialize database
python initialize_database.py

# Start the Flask server
python flask_app.py
```

**Open http://localhost:5000 and start chatting with your Medical AI Agent!**

## 🎮 **Usage Examples**

### 💬 **Web Interface Features**
- **Real-time Typing Animation** - Watch responses appear like ChatGPT
- **Professional Medical Theme** - Clean, healthcare-focused design
- **Mobile Responsive** - Perfect on desktop, tablet, and mobile
- **Response Timing** - See exact query processing time
- **Auto-scrolling Chat** - Smooth conversation flow

### 🔬 **Sample Questions**
```
"What are the common symptoms of Type 2 diabetes?"
"How is hypertension diagnosed and managed?"
"Explain the treatment options for cardiovascular disease"
"What lifestyle changes help prevent heart disease?"
"What are the risk factors for stroke?"
```

### 💻 **Command Line Interface**
```bash
# Interactive mode
python proper_medical_rag.py

# Quick test
python quick_test.py
```

### 🐍 **Programmatic Usage**
```python
from proper_medical_rag import ProperMedicalRAG

# Initialize the system
rag = ProperMedicalRAG()

# Ask a question
result = rag.query("What is diabetes?")
print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Response Time: {result.response_time}s")
```

## 🛠️ **Technology Stack**

- **Backend**: Flask + Python
- **Frontend**: HTML5 + Tailwind CSS + Vanilla JavaScript
- **AI/ML**: LangChain + Groq LLM + Custom Embeddings
- **Database**: ChromaDB Vector Store
- **Caching**: Custom embedding and query cache system

## 📁 **Project Architecture**

```
Medical-AI-Agent/
├── 🌐 flask_app.py              # Flask web server & API
├── 🧠 proper_medical_rag.py     # Core RAG system
├── 📄 templates/
│   ├── medical_chat.html        # ChatGPT-style interface
│   └── corpus_builder.html      # Corpus builder interface
├── 🏗️ corpus/                   # Glaucoma Corpus Builder
│   ├── glaucoma_builder.py      # Main corpus builder orchestrator
│   ├── auth_manager.py          # Secure API authentication
│   ├── error_manager.py         # Comprehensive error handling
│   ├── data_processor.py        # Data processing and quality assessment
│   ├── glaucoma_corpus_cli.py   # Command line interface
│   └── api_clients/             # Multi-API client implementations
│       ├── base_client.py       # Base API client with retry logic
│       ├── pubmed_client.py     # PubMed/NCBI integration
│       ├── clinical_trials_client.py # ClinicalTrials.gov integration
│       ├── openfda_client.py    # OpenFDA integration
│       └── europepmc_client.py  # Europe PMC integration
├── 🔧 rag/                      # RAG components
│   ├── embeddings.py           # Custom medical embeddings
│   ├── vectorstore.py          # ChromaDB integration
│   ├── retriever.py            # Document retrieval
│   └── generator.py            # Response generation
├── 📚 data/
│   ├── documents/              # Medical knowledge base
│   ├── cache/                  # Embedding cache
│   └── vectorstore/            # ChromaDB database
├── 🛠️ utils/                   # Utilities
│   ├── logger.py               # Logging system
│   └── cache.py                # Query caching
└── 🧪 tests/                   # Testing utilities
    ├── test_corpus_builder.py  # Corpus builder tests
    ├── quick_test.py           # Quick system test
    └── final_test.py           # Comprehensive test
```

## 🧠 **Custom Embeddings**

Unlike other systems, this uses **custom medical embeddings** instead of heavy pretrained models:

### **SimpleMedicalEmbeddings** (Default - 128d)
- Medical-aware text analysis
- Statistical features + medical terms
- Optimized for healthcare content

### **FastHashEmbeddings** (32d)
- Ultra-lightweight for mobile/edge
- Hash-based feature extraction
- <1ms embedding generation

### **WordCountEmbeddings** (50d)
- Interpretable vocabulary-based
- TF-IDF style with medical focus
- Great for debugging and analysis

```python
# Switch embedding models
from rag.embeddings import CachedMedicalEmbeddings

# Choose your model
embeddings = CachedMedicalEmbeddings(
    embedding_type='simple',    # Default: medical-aware
    # embedding_type='fast',    # Lightweight option
    # embedding_type='vocab',   # Interpretable option
    embedding_dim=128
)
```

## 📊 **Performance Metrics**

| Metric | Value |
|--------|-------|
| ⚡ **Response Time** | <2 seconds average |
| 💾 **Memory Usage** | <100MB (no large models) |
| 🗄️ **Database Size** | ~600KB vector store |
| 🔍 **Embedding Speed** | <1ms per query |
| 🎯 **Medical Accuracy** | 85%+ confidence on queries |
| 📱 **Mobile Support** | 100% responsive |

## 🔧 **Configuration**

### Environment Variables (.env)
```bash
# Required: Groq API Key
GROQ_API_KEY=your_groq_api_key_here

# Optional: Medical Research APIs for Glaucoma Corpus Builder
PUBMED_API_KEY=your_pubmed_api_key_here           # For higher rate limits
CLINICAL_TRIALS_API_KEY=your_clinical_trials_api_key_here
OPENFDA_API_KEY=your_openfda_api_key_here         # For higher rate limits
WHO_API_KEY=your_who_api_key_here
UMLS_API_KEY=your_umls_api_key_here              # Medical terminology
EUROPEPMC_API_KEY=your_europepmc_api_key_here
CROSSREF_API_KEY=your_crossref_api_key_here

# Optional: System settings
LOG_LEVEL=INFO
CACHE_ENABLED=true
FLASK_DEBUG=false
FLASK_PORT=5000
```

## 🏗️ **Glaucoma Corpus Builder**

### **Advanced Research Data Extraction**

The system includes a comprehensive glaucoma research corpus builder that extracts and processes data from multiple medical APIs:

#### **Supported Data Sources**
- **PubMed** - NCBI biomedical literature database
- **ClinicalTrials.gov** - Clinical trial registry and results
- **OpenFDA** - FDA drug and device safety data  
- **WHO Global Health Observatory** - Global health statistics
- **Europe PMC** - European life sciences literature
- **UMLS** - Unified Medical Language System

#### **Key Features**
- 🔐 **Secure Authentication** - Encrypted API key storage
- 🔄 **Intelligent Retry Logic** - Robust error handling with exponential backoff
- ⚡ **Rate Limiting** - Respects API limits automatically
- 🎯 **Glaucoma-Specific Processing** - Specialized relevance scoring and entity extraction
- 📊 **Quality Assessment** - Document quality metrics and filtering
- 🗑️ **Duplicate Detection** - Content-based deduplication
- 📈 **Comprehensive Analytics** - Detailed statistics and recommendations

### **Using the Corpus Builder**

#### **Web Interface**
```bash
# Start the Flask server
python flask_app.py

# Navigate to corpus builder
# Visit: http://localhost:5000/corpus
```

#### **Command Line Interface**
```bash
# Setup API credentials interactively
python corpus/glaucoma_corpus_cli.py setup --interactive

# Test API connections
python corpus/glaucoma_corpus_cli.py test

# Build corpus with default settings
python corpus/glaucoma_corpus_cli.py build

# Advanced corpus building
python corpus/glaucoma_corpus_cli.py build \
  --sources pubmed clinical_trials openfda \
  --max-results 1000 \
  --years 5 \
  --min-relevance 20.0 \
  --output-dir ./data/glaucoma_research

# Check system health
python corpus/glaucoma_corpus_cli.py health
```

#### **Python API**
```python
from corpus import GlaucomaCorpusBuilder, CorpusBuilderConfig

# Configure corpus builder
config = CorpusBuilderConfig(
    enabled_sources=['pubmed', 'clinical_trials'],
    max_results_per_source=500,
    search_years_back=3,
    min_relevance_score=25.0,
    min_quality_score=0.7
)

# Build corpus
with GlaucomaCorpusBuilder(config) as builder:
    result = builder.build_corpus()
    
    print(f"Built corpus with {result.filtered_documents} documents")
    print(f"Processing time: {result.processing_time:.1f}s")
    print(f"Output files: {result.output_files}")
```

### **Corpus Builder Workflow**

1. **Authentication** - Securely manages API credentials
2. **Search** - Executes glaucoma-specific queries across multiple databases  
3. **Extraction** - Retrieves comprehensive document metadata
4. **Processing** - Cleans, normalizes, and structures the data
5. **Quality Assessment** - Scores documents for relevance and quality
6. **Entity Extraction** - Identifies medical entities and concepts
7. **Deduplication** - Removes duplicate content
8. **Analysis** - Generates statistics and recommendations
9. **Export** - Saves processed corpus in JSON format

### **Sample Output**
```json
{
  "id": "pmid_12345678",
  "title": "Efficacy of Prostaglandin Analogs in Primary Open-Angle Glaucoma",
  "abstract": "This randomized controlled trial evaluated...",
  "glaucoma_relevance_score": 85.2,
  "quality_metrics": {
    "overall_quality": 0.92,
    "completeness": 0.95,
    "content_quality": 0.90
  },
  "extracted_entities": {
    "conditions": ["primary open-angle glaucoma"],
    "medications": ["latanoprost", "timolol"],
    "measurements": ["intraocular pressure", "cup-to-disc ratio"]
  }
}
```

### Customization Options
```python
# Modify rag/embeddings.py for different models
embeddings = SimpleMedicalEmbeddings(
    embedding_dim=128,          # Vector dimensions
    medical_terms_weight=0.3,   # Medical terminology focus
    statistical_features=True   # Include text statistics
)

# Adjust retrieval in rag/retriever.py
retriever_config = {
    'search_type': 'similarity',
    'k': 5,                     # Number of documents
    'score_threshold': 0.7      # Relevance threshold
}
```

## 🔌 **API Endpoints**

### Main Routes
- `GET /` - Chat interface
- `GET /corpus` - Glaucoma corpus builder interface
- `POST /api/query` - Process medical questions
- `POST /api/corpus/build` - Build glaucoma research corpus
- `GET /api/corpus/status` - Check API status and health
- `GET /api/stats` - System statistics

### Query API Example
```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the symptoms of diabetes?"}'
```

### Corpus Builder API Example
```bash
curl -X POST http://localhost:5000/api/corpus/build \
  -H "Content-Type: application/json" \
  -d '{
    "sources": ["pubmed", "clinical_trials"],
    "max_results": 500,
    "years_back": 3,
    "min_relevance": 20.0
  }'
```

Response:
```json
{
  "success": true,
  "answer": "Type 2 diabetes symptoms include...",
  "confidence": 0.89,
  "response_time": 1.45,
  "retrieved_docs_count": 5
}
```

## 🧪 **Testing & Validation**

### Quick Tests
```bash
# Test corpus builder functionality
python test_corpus_builder.py

# System health check
python quick_test.py

# Comprehensive testing
python final_test.py

# Interactive mode
python proper_medical_rag.py
```

### Custom Tests
```python
# Test custom embeddings
python test_custom_embeddings.py

# Reset system
python reset_system.py
```

## 🚨 **Important Medical Disclaimer**

⚠️ **FOR EDUCATIONAL PURPOSES ONLY**

- **Not for actual medical diagnosis or treatment**
- **Always consult qualified healthcare professionals**
- **Do not use for emergency medical situations**
- **Information provided is for learning only**

## 🔄 **System Reset & Maintenance**

```bash
# Reset entire system
python reset_system.py

# Clear cache only
rm -rf data/cache/*

# Reinitialize database
python initialize_database.py
```

## 🐛 **Troubleshooting**

### Common Issues

1. **"No module named 'flask'"**
   ```bash
   pip install -r requirements.txt
   ```

2. **"GROQ_API_KEY not found"**
   ```bash
   copy .env.example .env
   # Edit .env and add your API key
   ```

3. **Empty response div**
   - Check browser console for errors
   - Verify GROQ API key is correct
   - Ensure database is initialized

4. **Port 5000 in use**
   ```bash
   # Find process using port
   netstat -ano | findstr :5000
   # Kill process or change port in flask_app.py
   ```

### Performance Issues
- Clear cache: `rm -rf data/cache/*`
- Restart Flask server
- Check internet connection for LLM calls


## 📈 **Roadmap**

- [x] ✅ Flask web interface with ChatGPT-style UI
- [x] ✅ Custom medical embeddings system
- [x] ✅ Real-time response streaming
- [x] ✅ Comprehensive glaucoma corpus builder
- [x] ✅ Multi-API integration (PubMed, ClinicalTrials, OpenFDA, etc.)
- [x] ✅ Secure authentication and credential management
- [x] ✅ Advanced error handling and retry mechanisms
- [x] ✅ Professional web interface for corpus building
- [ ] 🔄 Multi-language medical knowledge
- [ ] 🔄 Advanced caching strategies
- [ ] 🔄 Docker deployment
- [ ] 🔄 User conversation history
- [ ] 🔄 Medical image analysis
- [ ] 🔄 Voice interface integration

## 📞 **Support & Community**

- 🐛 **Report Issues**: [GitHub Issues](https://github.com/Sagar4276/Medical-AI-Agent/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/Sagar4276/Medical-AI-Agent/discussions)
- 📧 **Contact**: Create an issue for questions
- 📖 **Documentation**: Check code comments and README


## 🙏 **Acknowledgments**

- **[LangChain](https://langchain.com/)** - RAG framework foundation
- **[Groq](https://groq.com/)** - Fast LLM inference
- **[ChromaDB](https://www.trychroma.com/)** - Vector database
- **[Tailwind CSS](https://tailwindcss.com/)** - Beautiful styling
- **Medical Community** - For knowledge validation and guidance

## 📊 **Star History**

⭐ **Star this repository if you found it helpful!**

## 🚀 **Get Started Now**

```bash
git clone https://github.com/Sagar4276/Medical-AI-Agent.git
cd Medical-AI-Agent
pip install -r requirements.txt
python flask_app.py
# Visit: http://localhost:5000
```

**Experience the future of medical AI education!** 🏥✨

---

<div align="center">
  <p><strong>Built with ❤️ for medical education and research</strong></p>
  <p>
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-usage-examples">Examples</a> •
    <a href="#-contributing">Contributing</a> •
    <a href="#-support--community">Support</a>
  </p>
</div>
