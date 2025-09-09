"""
Comprehensive Glaucoma Corpus Builder
Main orchestrator for multi-API glaucoma research data extraction and corpus building
"""

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from .auth_manager import APIAuthManager
from .error_manager import ErrorManager, CorpusBuilderError, APIError, ErrorDetails, ErrorCategory, ErrorSeverity
from .data_processor import GlaucomaDataProcessor, ProcessedDocument, CorpusStatistics
from .api_clients import (
    PubMedClient, ClinicalTrialsClient, OpenFDAClient, 
    WHOClient, EuropePMCClient
)

logger = logging.getLogger(__name__)

@dataclass
class CorpusBuilderConfig:
    """Configuration for corpus building process"""
    # Data sources configuration
    enabled_sources: List[str]
    max_results_per_source: int
    search_years_back: int
    
    # Quality filters
    min_relevance_score: float
    min_quality_score: float
    
    # Processing options
    remove_duplicates: bool
    extract_entities: bool
    calculate_statistics: bool
    
    # Output configuration
    output_directory: str
    export_formats: List[str]
    include_metadata: bool
    
    # Performance settings
    max_concurrent_requests: int
    request_delay: float
    timeout_seconds: int

@dataclass
class CorpusBuilderResult:
    """Results from corpus building process"""
    success: bool
    total_documents: int
    processed_documents: int
    filtered_documents: int
    sources_used: List[str]
    processing_time: float
    statistics: Optional[CorpusStatistics]
    output_files: List[str]
    error_summary: Dict[str, Any]
    recommendations: List[str]

class GlaucomaCorpusBuilder:
    """
    Comprehensive glaucoma research corpus builder
    
    Features:
    - Multi-API integration (PubMed, ClinicalTrials.gov, OpenFDA, etc.)
    - Robust error handling and retry mechanisms
    - Quality assessment and filtering
    - Duplicate detection and removal
    - Comprehensive data processing and structuring
    - Statistical analysis and reporting
    - Multiple export formats
    - Performance monitoring and optimization
    """
    
    def __init__(self, config: Optional[CorpusBuilderConfig] = None):
        """
        Initialize the corpus builder
        
        Args:
            config: Configuration object, uses defaults if None
        """
        # Set default configuration
        self.config = config or self._get_default_config()
        
        # Initialize core components
        self.auth_manager = APIAuthManager()
        self.error_manager = ErrorManager()
        self.data_processor = GlaucomaDataProcessor()
        
        # Initialize API clients
        self.clients = {}
        self._initialize_clients()
        
        # Glaucoma-specific search terms
        self.glaucoma_search_terms = {
            "primary": [
                "glaucoma",
                "intraocular pressure", 
                "optic nerve glaucoma",
                "angle-closure glaucoma",
                "open-angle glaucoma"
            ],
            "secondary": [
                "visual field defects glaucoma",
                "retinal ganglion cell glaucoma", 
                "trabecular meshwork",
                "aqueous humor dynamics",
                "optic disc cupping"
            ],
            "treatments": [
                "glaucoma surgery",
                "glaucoma medications",
                "trabeculectomy",
                "glaucoma laser therapy"
            ],
            "diagnostics": [
                "glaucoma diagnosis",
                "tonometry",
                "perimetry glaucoma", 
                "OCT glaucoma",
                "gonioscopy"
            ]
        }
        
        logger.info("Glaucoma Corpus Builder initialized successfully")
    
    def _get_default_config(self) -> CorpusBuilderConfig:
        """Get default configuration"""
        return CorpusBuilderConfig(
            enabled_sources=["pubmed", "clinical_trials", "europepmc"],
            max_results_per_source=1000,
            search_years_back=5,
            min_relevance_score=15.0,
            min_quality_score=0.5,
            remove_duplicates=True,
            extract_entities=True,
            calculate_statistics=True,
            output_directory="./data/glaucoma_corpus",
            export_formats=["json"],
            include_metadata=True,
            max_concurrent_requests=5,
            request_delay=0.5,
            timeout_seconds=30
        )
    
    def _initialize_clients(self):
        """Initialize API clients based on available credentials"""
        client_classes = {
            "pubmed": PubMedClient,
            "clinical_trials": ClinicalTrialsClient,
            "openfda": OpenFDAClient,
            "who_gho": WHOClient,
            "europepmc": EuropePMCClient
        }
        
        initialized_clients = []
        
        for source_name in self.config.enabled_sources:
            if source_name in client_classes:
                try:
                    if self.auth_manager.get_credential(source_name):
                        client_class = client_classes[source_name]
                        self.clients[source_name] = client_class(self.auth_manager)
                        initialized_clients.append(source_name)
                        logger.info(f"Initialized {source_name} client")
                    else:
                        logger.warning(f"No credentials available for {source_name}")
                except Exception as e:
                    logger.error(f"Failed to initialize {source_name} client: {e}")
        
        if not initialized_clients:
            logger.warning("No API clients were successfully initialized")
        else:
            logger.info(f"Initialized clients: {', '.join(initialized_clients)}")
    
    def build_corpus(self, 
                    custom_search_terms: Optional[List[str]] = None,
                    subtopics: Optional[List[str]] = None) -> CorpusBuilderResult:
        """
        Build comprehensive glaucoma research corpus
        
        Args:
            custom_search_terms: Additional search terms to include
            subtopics: Specific glaucoma subtopics to focus on
            
        Returns:
            CorpusBuilderResult with complete results and statistics
        """
        start_time = datetime.now()
        logger.info("Starting glaucoma corpus building process...")
        
        try:
            # Prepare search terms
            search_terms = self._prepare_search_terms(custom_search_terms, subtopics)
            
            # Extract raw data from all sources
            raw_documents = self._extract_from_all_sources(search_terms)
            
            # Process the documents
            processed_documents = self._process_documents(raw_documents)
            
            # Apply filters
            filtered_documents = self._apply_filters(processed_documents)
            
            # Generate statistics
            statistics = None
            if self.config.calculate_statistics:
                statistics = self.data_processor.generate_statistics(filtered_documents)
            
            # Export results
            output_files = self._export_results(filtered_documents, statistics)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(filtered_documents, statistics)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = CorpusBuilderResult(
                success=True,
                total_documents=len(raw_documents),
                processed_documents=len(processed_documents),
                filtered_documents=len(filtered_documents),
                sources_used=list(self.clients.keys()),
                processing_time=processing_time,
                statistics=statistics,
                output_files=output_files,
                error_summary=self.error_manager.get_error_summary(),
                recommendations=recommendations
            )
            
            logger.info(f"Corpus building completed successfully in {processing_time:.1f}s")
            logger.info(f"Final corpus: {len(filtered_documents)} high-quality documents")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_summary = self.error_manager.get_error_summary()
            
            logger.error(f"Corpus building failed after {processing_time:.1f}s: {e}")
            
            return CorpusBuilderResult(
                success=False,
                total_documents=0,
                processed_documents=0,
                filtered_documents=0,
                sources_used=list(self.clients.keys()),
                processing_time=processing_time,
                statistics=None,
                output_files=[],
                error_summary=error_summary,
                recommendations=["Check error logs and API credentials"]
            )
    
    def _prepare_search_terms(self, 
                            custom_terms: Optional[List[str]] = None,
                            subtopics: Optional[List[str]] = None) -> List[str]:
        """Prepare comprehensive search terms"""
        search_terms = []
        
        # Add primary glaucoma terms
        search_terms.extend(self.glaucoma_search_terms["primary"])
        
        # Add subtopic-specific terms
        if subtopics:
            for subtopic in subtopics:
                if subtopic in self.glaucoma_search_terms:
                    search_terms.extend(self.glaucoma_search_terms[subtopic])
        else:
            # Add all categories if no specific subtopics
            for category_terms in self.glaucoma_search_terms.values():
                search_terms.extend(category_terms)
        
        # Add custom terms
        if custom_terms:
            search_terms.extend(custom_terms)
        
        # Remove duplicates and sort
        unique_terms = list(set(search_terms))
        unique_terms.sort()
        
        logger.info(f"Prepared {len(unique_terms)} search terms")
        return unique_terms
    
    def _extract_from_all_sources(self, search_terms: List[str]) -> List[Dict[str, Any]]:
        """Extract data from all available sources"""
        all_documents = []
        
        logger.info(f"Extracting data from {len(self.clients)} sources...")
        
        # Use ThreadPoolExecutor for concurrent extraction
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_requests) as executor:
            # Submit extraction tasks
            future_to_source = {}
            
            for source_name, client in self.clients.items():
                future = executor.submit(
                    self._extract_from_source, 
                    source_name, 
                    client, 
                    search_terms
                )
                future_to_source[future] = source_name
            
            # Collect results
            for future in as_completed(future_to_source):
                source_name = future_to_source[future]
                try:
                    documents = future.result()
                    all_documents.extend(documents)
                    logger.info(f"Extracted {len(documents)} documents from {source_name}")
                except Exception as e:
                    logger.error(f"Error extracting from {source_name}: {e}")
                    self.error_manager.log_error(
                        self.error_manager.categorize_error(e, source_name)
                    )
        
        logger.info(f"Total raw documents extracted: {len(all_documents)}")
        return all_documents
    
    def _extract_from_source(self, 
                           source_name: str,
                           client: Any,
                           search_terms: List[str]) -> List[Dict[str, Any]]:
        """Extract data from a specific source"""
        documents = []
        max_results = self.config.max_results_per_source
        
        try:
            if source_name == "pubmed" and hasattr(client, 'search_glaucoma_corpus'):
                # Use specialized glaucoma search for PubMed
                docs = client.search_glaucoma_corpus(
                    subtopics=None,
                    max_results=max_results,
                    years_back=self.config.search_years_back
                )
                documents.extend(docs)
                
            else:
                # Use general search for other sources
                results_per_term = max_results // len(search_terms)
                
                for term in search_terms[:10]:  # Limit to top 10 terms
                    try:
                        docs = client.search(term, max_results=results_per_term)
                        documents.extend(docs)
                        
                        # Rate limiting
                        if self.config.request_delay > 0:
                            import time
                            time.sleep(self.config.request_delay)
                            
                    except Exception as e:
                        logger.warning(f"Error searching '{term}' in {source_name}: {e}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error in source extraction for {source_name}: {e}")
            raise
        
        return documents
    
    def _process_documents(self, raw_documents: List[Dict[str, Any]]) -> List[ProcessedDocument]:
        """Process raw documents"""
        logger.info("Processing raw documents...")
        return self.data_processor.process_documents(raw_documents)
    
    def _apply_filters(self, documents: List[ProcessedDocument]) -> List[ProcessedDocument]:
        """Apply quality and relevance filters"""
        logger.info("Applying filters...")
        
        # Apply relevance filter
        filtered = self.data_processor.filter_by_relevance(
            documents, 
            self.config.min_relevance_score
        )
        
        # Apply quality filter
        if self.config.min_quality_score > 0:
            filtered = self.data_processor.filter_by_quality(
                filtered,
                self.config.min_quality_score
            )
        
        return filtered
    
    def _export_results(self, 
                       documents: List[ProcessedDocument],
                       statistics: Optional[CorpusStatistics] = None) -> List[str]:
        """Export results to files"""
        output_files = []
        
        # Ensure output directory exists
        os.makedirs(self.config.output_directory, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for format_type in self.config.export_formats:
            if format_type.lower() == "json":
                # Export main corpus
                corpus_file = os.path.join(
                    self.config.output_directory,
                    f"glaucoma_corpus_{timestamp}.json"
                )
                
                if self.data_processor.export_corpus(documents, corpus_file, "json"):
                    output_files.append(corpus_file)
                
                # Export statistics if available
                if statistics:
                    stats_file = os.path.join(
                        self.config.output_directory,
                        f"corpus_statistics_{timestamp}.json"
                    )
                    
                    try:
                        with open(stats_file, 'w', encoding='utf-8') as f:
                            json.dump(asdict(statistics), f, indent=2, default=str)
                        output_files.append(stats_file)
                    except Exception as e:
                        logger.error(f"Error exporting statistics: {e}")
        
        return output_files
    
    def _generate_recommendations(self, 
                                documents: List[ProcessedDocument],
                                statistics: Optional[CorpusStatistics] = None) -> List[str]:
        """Generate recommendations based on corpus analysis"""
        recommendations = []
        
        if not documents:
            recommendations.append("No documents in final corpus - review search terms and filters")
            return recommendations
        
        # Document count recommendations
        if len(documents) < 100:
            recommendations.append("Consider expanding search terms or reducing quality filters for more documents")
        elif len(documents) > 5000:
            recommendations.append("Large corpus detected - consider more restrictive filters for focused research")
        
        # Quality recommendations
        if statistics:
            avg_quality = sum(doc.quality_metrics['overall_quality'] for doc in documents) / len(documents)
            if avg_quality < 0.7:
                recommendations.append("Average quality is low - consider higher quality thresholds")
            
            # Relevance recommendations
            if statistics.average_relevance_score < 30:
                recommendations.append("Low average relevance - refine search terms to be more glaucoma-specific")
            
            # Source diversity
            source_count = len(statistics.documents_by_source)
            if source_count == 1:
                recommendations.append("Single source detected - enable additional APIs for comprehensive coverage")
            
            # Date range recommendations
            years = list(statistics.documents_by_year.keys())
            if years and years[-1] != str(datetime.now().year):
                recommendations.append("No recent publications - check if current year data is available")
        
        # Data completeness
        docs_with_abstracts = sum(1 for doc in documents if doc.abstract)
        abstract_coverage = docs_with_abstracts / len(documents)
        if abstract_coverage < 0.8:
            recommendations.append(f"Only {abstract_coverage:.1%} of documents have abstracts - may impact analysis quality")
        
        return recommendations
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of all API clients"""
        status = {
            "overall_health": True,
            "client_status": {},
            "credentials_status": {},
            "last_check": datetime.now().isoformat()
        }
        
        for source_name, client in self.clients.items():
            try:
                is_healthy = client.health_check()
                status["client_status"][source_name] = {
                    "healthy": is_healthy,
                    "last_request": getattr(client, 'last_request_time', 0),
                    "request_count": getattr(client, 'request_count', 0)
                }
                
                if not is_healthy:
                    status["overall_health"] = False
                    
            except Exception as e:
                status["client_status"][source_name] = {
                    "healthy": False,
                    "error": str(e)
                }
                status["overall_health"] = False
        
        # Check credentials
        available_apis = self.auth_manager.list_available_apis()
        for api_info in available_apis:
            status["credentials_status"][api_info["name"]] = {
                "available": api_info["available"],
                "description": api_info.get("description", "")
            }
        
        return status
    
    def get_progress_info(self) -> Dict[str, Any]:
        """Get current progress information"""
        return {
            "error_summary": self.error_manager.get_error_summary(),
            "processed_documents": len(self.data_processor.processed_documents),
            "active_clients": list(self.clients.keys()),
            "configuration": asdict(self.config)
        }
    
    def cleanup(self):
        """Clean up resources"""
        for client in self.clients.values():
            try:
                client.close()
            except Exception as e:
                logger.error(f"Error closing client: {e}")
        
        logger.info("Corpus builder cleanup completed")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()