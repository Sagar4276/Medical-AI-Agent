"""
Data Processor for Glaucoma Corpus Builder
Comprehensive data processing, cleaning, and structuring for research data
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime
from collections import Counter, defaultdict
import hashlib

from .error_manager import DataProcessingError, ErrorDetails, ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)

@dataclass
class ProcessedDocument:
    """Container for processed research document"""
    id: str
    title: str
    abstract: str
    full_text: str
    authors: List[str]
    publication_date: str
    journal: str
    doi: str
    keywords: List[str]
    mesh_terms: List[str]
    glaucoma_relevance_score: float
    article_type: str
    source: str
    processing_metadata: Dict[str, Any]
    extracted_entities: Dict[str, List[str]]
    quality_metrics: Dict[str, float]

@dataclass
class CorpusStatistics:
    """Statistics for the processed corpus"""
    total_documents: int
    documents_by_source: Dict[str, int]
    documents_by_type: Dict[str, int]
    documents_by_year: Dict[str, int]
    top_journals: List[Tuple[str, int]]
    top_authors: List[Tuple[str, int]]
    top_keywords: List[Tuple[str, int]]
    top_mesh_terms: List[Tuple[str, int]]
    average_relevance_score: float
    quality_distribution: Dict[str, int]
    language_distribution: Dict[str, int]

class GlaucomaDataProcessor:
    """
    Comprehensive data processor for glaucoma research corpus
    
    Features:
    - Text cleaning and normalization
    - Medical entity extraction
    - Quality assessment and scoring
    - Duplicate detection and removal
    - Content structuring and standardization
    - Statistical analysis and reporting
    """
    
    def __init__(self):
        # Medical terminology patterns
        self.glaucoma_entities = {
            "conditions": [
                "primary open-angle glaucoma", "angle-closure glaucoma", "normal-tension glaucoma",
                "secondary glaucoma", "congenital glaucoma", "neovascular glaucoma",
                "pigmentary glaucoma", "pseudoexfoliative glaucoma", "traumatic glaucoma"
            ],
            "measurements": [
                "intraocular pressure", "IOP", "cup-to-disc ratio", "CDR", "visual field defect",
                "retinal nerve fiber layer", "RNFL", "ganglion cell complex", "GCC",
                "central corneal thickness", "CCT", "anterior chamber depth"
            ],
            "procedures": [
                "tonometry", "perimetry", "gonioscopy", "ophthalmoscopy", "OCT", 
                "optical coherence tomography", "visual field testing", "fundus photography",
                "trabeculectomy", "tube shunt", "laser trabeculoplasty", "cyclophotocoagulation"
            ],
            "medications": [
                "prostaglandin analogs", "beta-blockers", "carbonic anhydrase inhibitors",
                "alpha-adrenergic agonists", "cholinergic agents", "latanoprost", "timolol",
                "brimonidine", "dorzolamide", "brinzolamide", "travoprost", "bimatoprost"
            ],
            "anatomy": [
                "optic disc", "optic nerve", "retinal ganglion cells", "trabecular meshwork",
                "aqueous humor", "anterior chamber", "posterior chamber", "ciliary body",
                "lamina cribrosa", "optic nerve head", "peripapillary region"
            ]
        }
        
        # Quality assessment criteria
        self.quality_indicators = {
            "high_quality": ["randomized", "controlled trial", "meta-analysis", "systematic review"],
            "medium_quality": ["cohort", "case-control", "observational"],
            "low_quality": ["case report", "editorial", "letter", "comment"]
        }
        
        # Text preprocessing patterns
        self.cleanup_patterns = [
            (r'\s+', ' '),  # Multiple spaces
            (r'\n+', '\n'),  # Multiple newlines
            (r'[^\w\s\-\.\,\;\:\!\?\(\)\[\]\/]', ''),  # Special characters
            (r'(?i)copyright.*?\d{4}.*?\.', ''),  # Copyright statements
            (r'(?i)all rights reserved.*?\.', ''),  # Rights statements
        ]
        
        self.processed_documents: List[ProcessedDocument] = []
        self.document_hashes: Set[str] = set()
        
    def process_documents(self, raw_documents: List[Dict[str, Any]]) -> List[ProcessedDocument]:
        """
        Process raw documents from various sources
        
        Args:
            raw_documents: List of raw document dictionaries
            
        Returns:
            List of processed documents
        """
        processed = []
        duplicates_removed = 0
        
        logger.info(f"Processing {len(raw_documents)} raw documents...")
        
        for i, doc in enumerate(raw_documents):
            try:
                processed_doc = self._process_single_document(doc)
                
                if processed_doc:
                    # Check for duplicates
                    if not self._is_duplicate(processed_doc):
                        processed.append(processed_doc)
                        self.processed_documents.append(processed_doc)
                        self.document_hashes.add(processed_doc.processing_metadata['content_hash'])
                    else:
                        duplicates_removed += 1
                
                # Log progress
                if (i + 1) % 100 == 0:
                    logger.info(f"Processed {i + 1}/{len(raw_documents)} documents")
                    
            except Exception as e:
                logger.error(f"Error processing document {i}: {e}")
                continue
        
        logger.info(f"Processing complete: {len(processed)} documents processed, {duplicates_removed} duplicates removed")
        return processed
    
    def _process_single_document(self, doc: Dict[str, Any]) -> Optional[ProcessedDocument]:
        """Process a single document"""
        try:
            # Extract basic information
            doc_id = doc.get('pmid') or doc.get('id') or doc.get('doi') or str(hash(doc.get('title', '')))
            title = self._clean_text(doc.get('title', ''))
            abstract = self._clean_text(doc.get('abstract', ''))
            
            # Skip documents without essential content
            if not title and not abstract:
                logger.warning(f"Skipping document {doc_id}: no title or abstract")
                return None
            
            # Full text construction
            full_text = self._construct_full_text(doc, title, abstract)
            
            # Extract and process metadata
            authors = doc.get('authors', [])
            if isinstance(authors, str):
                authors = [authors]
            
            publication_date = self._normalize_date(doc.get('publication_date', ''))
            journal = doc.get('journal', '')
            doi = doc.get('doi', '')
            
            # Extract keywords and MeSH terms
            keywords = doc.get('keywords', [])
            mesh_terms = doc.get('mesh_terms', [])
            
            # Calculate relevance score
            relevance_score = doc.get('metadata', {}).get('glaucoma_relevance_score', 0.0)
            if relevance_score == 0.0:
                relevance_score = self._calculate_relevance_score(title, abstract, keywords, mesh_terms)
            
            # Determine article type
            article_type = doc.get('metadata', {}).get('article_type', '')
            if not article_type:
                article_type = self._classify_article_type(title, abstract, full_text)
            
            # Extract medical entities
            extracted_entities = self._extract_medical_entities(full_text)
            
            # Assess quality
            quality_metrics = self._assess_quality(doc, title, abstract, full_text)
            
            # Generate content hash for duplicate detection
            content_hash = self._generate_content_hash(title, abstract)
            
            # Create processing metadata
            processing_metadata = {
                'processed_date': datetime.now().isoformat(),
                'content_hash': content_hash,
                'original_source': doc.get('source', 'unknown'),
                'text_length': len(full_text),
                'abstract_length': len(abstract),
                'title_length': len(title),
                'processing_version': '1.0'
            }
            
            return ProcessedDocument(
                id=doc_id,
                title=title,
                abstract=abstract,
                full_text=full_text,
                authors=authors,
                publication_date=publication_date,
                journal=journal,
                doi=doi,
                keywords=keywords,
                mesh_terms=mesh_terms,
                glaucoma_relevance_score=relevance_score,
                article_type=article_type,
                source=doc.get('source', 'unknown'),
                processing_metadata=processing_metadata,
                extracted_entities=extracted_entities,
                quality_metrics=quality_metrics
            )
            
        except Exception as e:
            logger.error(f"Error in _process_single_document: {e}")
            raise DataProcessingError(
                ErrorDetails(
                    category=ErrorCategory.DATA_PROCESSING,
                    severity=ErrorSeverity.MEDIUM,
                    message=f"Failed to process document: {str(e)}"
                )
            )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Apply cleanup patterns
        for pattern, replacement in self.cleanup_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Additional cleaning
        text = text.strip()
        
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        return text
    
    def _construct_full_text(self, doc: Dict[str, Any], title: str, abstract: str) -> str:
        """Construct full text from available content"""
        parts = []
        
        if title:
            parts.append(f"Title: {title}")
        
        if abstract:
            parts.append(f"Abstract: {abstract}")
        
        # Add other available content
        if doc.get('introduction'):
            parts.append(f"Introduction: {self._clean_text(doc['introduction'])}")
        
        if doc.get('methods'):
            parts.append(f"Methods: {self._clean_text(doc['methods'])}")
        
        if doc.get('results'):
            parts.append(f"Results: {self._clean_text(doc['results'])}")
        
        if doc.get('discussion'):
            parts.append(f"Discussion: {self._clean_text(doc['discussion'])}")
        
        if doc.get('conclusion'):
            parts.append(f"Conclusion: {self._clean_text(doc['conclusion'])}")
        
        # If we have full_text field, use it
        if doc.get('full_text') and len(doc['full_text']) > len(abstract):
            parts.append(f"Content: {self._clean_text(doc['full_text'])}")
        
        return "\n\n".join(parts)
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize publication date to ISO format"""
        if not date_str:
            return ""
        
        # Try to parse various date formats
        date_patterns = [
            r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD
            r'(\d{4})/(\d{1,2})/(\d{1,2})',  # YYYY/MM/DD
            r'(\d{4})\s+(\w+)\s+(\d{1,2})',  # YYYY Month DD
            r'(\d{4})\s+(\w+)',              # YYYY Month
            r'(\d{4})',                      # YYYY only
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, date_str)
            if match:
                try:
                    year = int(match.group(1))
                    if year < 1900 or year > 2030:
                        continue
                    
                    if len(match.groups()) == 3:
                        month = match.group(2)
                        day = int(match.group(3))
                        
                        # Convert month name to number if needed
                        if month.isalpha():
                            month_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                         'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
                            month_num = next((i+1 for i, m in enumerate(month_names) 
                                            if m in month.lower()[:3]), 1)
                        else:
                            month_num = int(month)
                        
                        return f"{year}-{month_num:02d}-{day:02d}"
                    
                    elif len(match.groups()) == 2:
                        month = match.group(2)
                        if month.isalpha():
                            month_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                                         'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
                            month_num = next((i+1 for i, m in enumerate(month_names) 
                                            if m in month.lower()[:3]), 1)
                        else:
                            month_num = int(month)
                        return f"{year}-{month_num:02d}-01"
                    
                    else:
                        return f"{year}-01-01"
                        
                except (ValueError, IndexError):
                    continue
        
        return date_str  # Return original if no pattern matches
    
    def _calculate_relevance_score(self, title: str, abstract: str, 
                                 keywords: List[str], mesh_terms: List[str]) -> float:
        """Calculate glaucoma relevance score"""
        score = 0.0
        text_content = f"{title} {abstract}".lower()
        
        # Entity-based scoring
        for category, entities in self.glaucoma_entities.items():
            category_score = 0
            for entity in entities:
                if entity.lower() in text_content:
                    category_score += 1
            
            # Weight different categories
            weights = {
                "conditions": 10.0,
                "measurements": 5.0,
                "procedures": 3.0,
                "medications": 3.0,
                "anatomy": 2.0
            }
            score += category_score * weights.get(category, 1.0)
        
        # MeSH terms (high weight)
        glaucoma_mesh = ["glaucoma", "intraocular pressure", "optic nerve"]
        for mesh_term in mesh_terms:
            if any(term.lower() in mesh_term.lower() for term in glaucoma_mesh):
                score += 15.0
        
        # Keywords
        for keyword in keywords:
            if any(term.lower() in keyword.lower() for term in glaucoma_mesh):
                score += 5.0
        
        # Title emphasis
        title_lower = title.lower()
        if "glaucoma" in title_lower:
            score += 20.0
        
        return min(score, 100.0)  # Cap at 100
    
    def _classify_article_type(self, title: str, abstract: str, full_text: str) -> str:
        """Classify article type based on content"""
        text_content = f"{title} {abstract} {full_text}".lower()
        
        # Check for specific indicators
        type_indicators = {
            "systematic_review": ["systematic review", "meta-analysis", "meta analysis"],
            "clinical_trial": ["randomized", "clinical trial", "rct", "controlled trial"],
            "case_report": ["case report", "case study", "case series"],
            "cohort_study": ["cohort study", "longitudinal", "follow-up", "prospective"],
            "cross_sectional": ["cross-sectional", "cross sectional", "prevalence study"],
            "laboratory_study": ["in vitro", "cell culture", "experimental study"],
            "review": ["review", "overview", "narrative review"]
        }
        
        for article_type, indicators in type_indicators.items():
            if any(indicator in text_content for indicator in indicators):
                return article_type
        
        return "research_article"
    
    def _extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract medical entities from text"""
        text_lower = text.lower()
        extracted = defaultdict(list)
        
        for category, entities in self.glaucoma_entities.items():
            for entity in entities:
                if entity.lower() in text_lower:
                    extracted[category].append(entity)
        
        # Convert to regular dict and remove duplicates
        return {k: list(set(v)) for k, v in extracted.items()}
    
    def _assess_quality(self, doc: Dict[str, Any], title: str, 
                       abstract: str, full_text: str) -> Dict[str, float]:
        """Assess document quality metrics"""
        metrics = {}
        
        # Completeness score
        completeness = 0.0
        if title:
            completeness += 0.2
        if abstract and len(abstract) > 100:
            completeness += 0.3
        if doc.get('authors'):
            completeness += 0.1
        if doc.get('journal'):
            completeness += 0.1
        if doc.get('doi'):
            completeness += 0.1
        if len(full_text) > 500:
            completeness += 0.2
        
        metrics['completeness'] = completeness
        
        # Content quality score
        content_quality = 0.0
        text_content = f"{title} {abstract}".lower()
        
        # Check for quality indicators
        for quality_level, indicators in self.quality_indicators.items():
            if any(indicator in text_content for indicator in indicators):
                if quality_level == "high_quality":
                    content_quality = 0.9
                elif quality_level == "medium_quality":
                    content_quality = 0.7
                elif quality_level == "low_quality":
                    content_quality = 0.3
                break
        
        if content_quality == 0.0:
            content_quality = 0.5  # Default for unknown quality
        
        metrics['content_quality'] = content_quality
        
        # Length-based metrics
        abstract_length = len(abstract) if abstract else 0
        metrics['abstract_adequacy'] = min(abstract_length / 200, 1.0)  # Normalize to 200 chars
        
        # Overall quality score
        metrics['overall_quality'] = (
            metrics['completeness'] * 0.4 +
            metrics['content_quality'] * 0.4 +
            metrics['abstract_adequacy'] * 0.2
        )
        
        return metrics
    
    def _generate_content_hash(self, title: str, abstract: str) -> str:
        """Generate hash for duplicate detection"""
        content = f"{title.lower().strip()}{abstract.lower().strip()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_duplicate(self, doc: ProcessedDocument) -> bool:
        """Check if document is duplicate"""
        return doc.processing_metadata['content_hash'] in self.document_hashes
    
    def filter_by_quality(self, documents: List[ProcessedDocument], 
                         min_quality: float = 0.6) -> List[ProcessedDocument]:
        """Filter documents by quality threshold"""
        filtered = [doc for doc in documents 
                   if doc.quality_metrics['overall_quality'] >= min_quality]
        
        logger.info(f"Quality filtering: {len(filtered)}/{len(documents)} documents retained "
                   f"(threshold: {min_quality})")
        return filtered
    
    def filter_by_relevance(self, documents: List[ProcessedDocument],
                           min_relevance: float = 20.0) -> List[ProcessedDocument]:
        """Filter documents by relevance score"""
        filtered = [doc for doc in documents if doc.glaucoma_relevance_score >= min_relevance]
        
        logger.info(f"Relevance filtering: {len(filtered)}/{len(documents)} documents retained "
                   f"(threshold: {min_relevance})")
        return filtered
    
    def generate_statistics(self, documents: List[ProcessedDocument]) -> CorpusStatistics:
        """Generate comprehensive corpus statistics"""
        if not documents:
            return CorpusStatistics(
                total_documents=0,
                documents_by_source={},
                documents_by_type={},
                documents_by_year={},
                top_journals=[],
                top_authors=[],
                top_keywords=[],
                top_mesh_terms=[],
                average_relevance_score=0.0,
                quality_distribution={},
                language_distribution={}
            )
        
        # Count by source
        by_source = Counter(doc.source for doc in documents)
        
        # Count by type
        by_type = Counter(doc.article_type for doc in documents)
        
        # Count by year
        by_year = Counter()
        for doc in documents:
            year = doc.publication_date[:4] if doc.publication_date else "Unknown"
            by_year[year] += 1
        
        # Top journals
        journal_counter = Counter(doc.journal for doc in documents if doc.journal)
        top_journals = journal_counter.most_common(10)
        
        # Top authors
        author_counter = Counter()
        for doc in documents:
            for author in doc.authors:
                author_counter[author] += 1
        top_authors = author_counter.most_common(10)
        
        # Top keywords
        keyword_counter = Counter()
        for doc in documents:
            for keyword in doc.keywords:
                keyword_counter[keyword.lower()] += 1
        top_keywords = keyword_counter.most_common(20)
        
        # Top MeSH terms
        mesh_counter = Counter()
        for doc in documents:
            for mesh_term in doc.mesh_terms:
                mesh_counter[mesh_term] += 1
        top_mesh_terms = mesh_counter.most_common(20)
        
        # Average relevance score
        avg_relevance = sum(doc.glaucoma_relevance_score for doc in documents) / len(documents)
        
        # Quality distribution
        quality_ranges = {"high": 0, "medium": 0, "low": 0}
        for doc in documents:
            quality = doc.quality_metrics['overall_quality']
            if quality >= 0.8:
                quality_ranges["high"] += 1
            elif quality >= 0.6:
                quality_ranges["medium"] += 1
            else:
                quality_ranges["low"] += 1
        
        # Language distribution (simplified - assume English for now)
        language_dist = {"English": len(documents)}
        
        return CorpusStatistics(
            total_documents=len(documents),
            documents_by_source=dict(by_source),
            documents_by_type=dict(by_type),
            documents_by_year=dict(sorted(by_year.items())),
            top_journals=top_journals,
            top_authors=top_authors,
            top_keywords=top_keywords,
            top_mesh_terms=top_mesh_terms,
            average_relevance_score=avg_relevance,
            quality_distribution=quality_ranges,
            language_distribution=language_dist
        )
    
    def export_corpus(self, documents: List[ProcessedDocument], 
                     output_file: str, format: str = "json") -> bool:
        """Export processed corpus to file"""
        try:
            if format.lower() == "json":
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump([asdict(doc) for doc in documents], f, 
                             indent=2, ensure_ascii=False, default=str)
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            logger.info(f"Exported {len(documents)} documents to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting corpus: {e}")
            return False