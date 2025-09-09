"""
Glaucoma Corpus Builder Package
Comprehensive medical research data extraction and corpus building system
"""

__version__ = "1.0.0"
__author__ = "Medical AI Agent"
__description__ = "Comprehensive glaucoma research corpus builder with multi-API integration"

from .glaucoma_builder import GlaucomaCorpusBuilder, CorpusBuilderConfig
from .auth_manager import APIAuthManager
from .error_manager import CorpusBuilderError, APIError, DataProcessingError
from .data_processor import GlaucomaDataProcessor

__all__ = [
    "GlaucomaCorpusBuilder",
    "APIAuthManager", 
    "CorpusBuilderError",
    "APIError",
    "DataProcessingError",
    "GlaucomaDataProcessor"
]