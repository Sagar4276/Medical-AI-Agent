"""
PubMed API Client for Glaucoma Research
NCBI E-utilities API client for biomedical literature retrieval
"""

import re
import xml.etree.ElementTree as ET
from typing import Dict, Any, List, Optional
from urllib.parse import quote
from datetime import datetime

from .base_client import BaseAPIClient
from ..auth_manager import APIAuthManager
from ..error_manager import APIError, DataProcessingError, ErrorDetails, ErrorCategory, ErrorSeverity

import logging
logger = logging.getLogger(__name__)

class PubMedClient(BaseAPIClient):
    """
    PubMed API client for accessing NCBI biomedical literature database
    
    Features:
    - Advanced search queries with MeSH terms
    - Bulk document retrieval
    - Citation metadata extraction
    - Full-text link discovery
    - Glaucoma-specific search optimization
    """
    
    def __init__(self, auth_manager: APIAuthManager):
        super().__init__(auth_manager, "pubmed", max_requests_per_second=3)
        
        # PubMed-specific configurations
        self.databases = ["pubmed", "pmc"]  # PubMed and PMC
        self.retmax_default = 100  # Default max results per request
        self.retmax_limit = 10000   # API limit
        
        # Glaucoma-specific search terms and MeSH terms
        self.glaucoma_mesh_terms = [
            "Glaucoma[MeSH]",
            "Glaucoma, Open-Angle[MeSH]", 
            "Glaucoma, Angle-Closure[MeSH]",
            "Glaucoma, Neovascular[MeSH]",
            "Intraocular Pressure[MeSH]",
            "Optic Nerve Diseases[MeSH]",
            "Visual Field Defects[MeSH]",
            "Ophthalmology[MeSH]"
        ]
        
        self.glaucoma_keywords = [
            "glaucoma", "intraocular pressure", "IOP", "optic nerve",
            "visual field", "retinal ganglion", "aqueous humor",
            "trabecular meshwork", "optic disc", "cup-to-disc ratio",
            "perimetry", "tonometry", "gonioscopy"
        ]
        
    def search(self, 
               query: str,
               max_results: int = 100,
               database: str = "pubmed",
               date_range: Optional[tuple] = None,
               article_types: Optional[List[str]] = None,
               **kwargs) -> List[Dict[str, Any]]:
        """
        Search PubMed for articles
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            database: Database to search (pubmed, pmc)
            date_range: Tuple of (start_date, end_date) in YYYY/MM/DD format
            article_types: List of article types to filter
            
        Returns:
            List of article metadata dictionaries
        """
        try:
            # Enhance query with glaucoma-specific terms if not present
            enhanced_query = self._enhance_glaucoma_query(query)
            
            # Add date range if specified
            if date_range:
                date_filter = f" AND ({date_range[0]}[PDAT]:{date_range[1]}[PDAT])"
                enhanced_query += date_filter
                
            # Add article type filters
            if article_types:
                type_filter = " AND (" + " OR ".join([f"{t}[PT]" for t in article_types]) + ")"
                enhanced_query += type_filter
            
            logger.info(f"Searching PubMed with query: {enhanced_query}")
            
            # Step 1: Search to get PMIDs
            pmids = self._search_pmids(enhanced_query, max_results, database)
            
            if not pmids:
                logger.warning("No results found for query")
                return []
            
            logger.info(f"Found {len(pmids)} articles, fetching metadata...")
            
            # Step 2: Fetch detailed metadata for PMIDs
            articles = self._fetch_article_details(pmids, database)
            
            return articles
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            raise APIError(
                ErrorDetails(
                    category=ErrorCategory.API,
                    severity=ErrorSeverity.HIGH,
                    message=f"PubMed search failed: {str(e)}",
                    api_endpoint="esearch.fcgi"
                )
            )
    
    def _enhance_glaucoma_query(self, query: str) -> str:
        """Enhance search query with glaucoma-specific terms"""
        query_lower = query.lower()
        
        # If query doesn't contain glaucoma terms, add them
        has_glaucoma_terms = any(term.lower().replace("[mesh]", "") in query_lower 
                                for term in self.glaucoma_keywords)
        
        if not has_glaucoma_terms and "glaucoma" not in query_lower:
            # Add glaucoma as a required term
            query = f"({query}) AND (glaucoma[MeSH] OR glaucoma[TIAB])"
        
        return query
    
    def _search_pmids(self, query: str, max_results: int, database: str) -> List[str]:
        """Search for PMIDs using esearch"""
        params = {
            "db": database,
            "term": query,
            "retmax": min(max_results, self.retmax_limit),
            "retmode": "json",
            "sort": "relevance",
            "tool": "medical_ai_agent",
            "email": "research@medical-ai-agent.com"
        }
        
        # Add API key if available
        if self.credential.api_key:
            params["api_key"] = self.credential.api_key
        
        response = self.get("esearch.fcgi", params=params)
        
        if not response.success:
            raise APIError(
                ErrorDetails(
                    category=ErrorCategory.API,
                    severity=ErrorSeverity.HIGH,
                    message="Failed to search PubMed",
                    status_code=response.status_code
                )
            )
        
        # Extract PMIDs from response
        search_results = response.data.get("esearchresult", {})
        pmids = search_results.get("idlist", [])
        
        logger.debug(f"Found {len(pmids)} PMIDs")
        return pmids
    
    def _fetch_article_details(self, pmids: List[str], database: str) -> List[Dict[str, Any]]:
        """Fetch detailed article metadata using efetch"""
        if not pmids:
            return []
        
        # PubMed API allows max 200 IDs per request
        batch_size = 200
        all_articles = []
        
        for i in range(0, len(pmids), batch_size):
            batch_pmids = pmids[i:i + batch_size]
            pmid_string = ",".join(batch_pmids)
            
            params = {
                "db": database,
                "id": pmid_string,
                "retmode": "xml",
                "rettype": "abstract" if database == "pubmed" else "full",
                "tool": "medical_ai_agent",
                "email": "research@medical-ai-agent.com"
            }
            
            # Add API key if available
            if self.credential.api_key:
                params["api_key"] = self.credential.api_key
            
            response = self.get("efetch.fcgi", params=params)
            
            if response.success:
                articles = self._parse_pubmed_xml(response.data.get("raw_content", ""))
                all_articles.extend(articles)
            else:
                logger.error(f"Failed to fetch details for batch {i//batch_size + 1}")
        
        logger.info(f"Successfully parsed {len(all_articles)} articles")
        return all_articles
    
    def _parse_pubmed_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """Parse PubMed XML response to extract article metadata"""
        articles = []
        
        try:
            root = ET.fromstring(xml_content)
            
            # Handle PubmedArticleSet
            for article_elem in root.findall(".//PubmedArticle"):
                article = self._extract_article_data(article_elem)
                if article:
                    articles.append(article)
                    
            # Handle PMC articles if present
            for article_elem in root.findall(".//Article"):
                article = self._extract_pmc_data(article_elem)
                if article:
                    articles.append(article)
                    
        except ET.ParseError as e:
            logger.error(f"Error parsing PubMed XML: {e}")
            raise DataProcessingError(
                ErrorDetails(
                    category=ErrorCategory.DATA_PROCESSING,
                    severity=ErrorSeverity.MEDIUM,
                    message=f"Failed to parse PubMed XML response: {e}"
                )
            )
        
        return articles
    
    def _extract_article_data(self, article_elem: ET.Element) -> Optional[Dict[str, Any]]:
        """Extract data from PubmedArticle element"""
        try:
            medline_citation = article_elem.find("MedlineCitation")
            if medline_citation is None:
                return None
                
            article_elem_inner = medline_citation.find("Article")
            if article_elem_inner is None:
                return None
            
            # Basic article information
            pmid_elem = medline_citation.find("PMID")
            pmid = pmid_elem.text if pmid_elem is not None else None
            
            # Title
            title_elem = article_elem_inner.find(".//ArticleTitle")
            title = self._clean_text(title_elem.text if title_elem is not None else "")
            
            # Abstract
            abstract_elem = article_elem_inner.find(".//AbstractText")
            abstract = self._clean_text(abstract_elem.text if abstract_elem is not None else "")
            
            # Handle structured abstracts
            if not abstract:
                abstract_parts = []
                for abs_elem in article_elem_inner.findall(".//AbstractText"):
                    label = abs_elem.get("Label", "")
                    text = abs_elem.text or ""
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
                abstract = " ".join(abstract_parts)
            
            # Authors
            authors = []
            for author_elem in article_elem_inner.findall(".//Author"):
                lastname_elem = author_elem.find("LastName")
                forename_elem = author_elem.find("ForeName")
                
                if lastname_elem is not None and forename_elem is not None:
                    author_name = f"{forename_elem.text} {lastname_elem.text}"
                    authors.append(author_name.strip())
            
            # Journal information
            journal_elem = article_elem_inner.find(".//Journal")
            journal_title = ""
            publication_date = ""
            volume = ""
            issue = ""
            
            if journal_elem is not None:
                title_elem = journal_elem.find(".//Title")
                journal_title = title_elem.text if title_elem is not None else ""
                
                # Publication date
                pub_date_elem = journal_elem.find(".//PubDate")
                if pub_date_elem is not None:
                    year_elem = pub_date_elem.find("Year")
                    month_elem = pub_date_elem.find("Month")
                    day_elem = pub_date_elem.find("Day")
                    
                    date_parts = []
                    if year_elem is not None:
                        date_parts.append(year_elem.text)
                    if month_elem is not None:
                        date_parts.append(month_elem.text)
                    if day_elem is not None:
                        date_parts.append(day_elem.text)
                    
                    publication_date = " ".join(date_parts)
                
                # Volume and Issue
                issue_elem = journal_elem.find(".//JournalIssue")
                if issue_elem is not None:
                    vol_elem = issue_elem.find("Volume")
                    iss_elem = issue_elem.find("Issue")
                    volume = vol_elem.text if vol_elem is not None else ""
                    issue = iss_elem.text if iss_elem is not None else ""
            
            # MeSH terms
            mesh_terms = []
            for mesh_elem in medline_citation.findall(".//MeshHeading/DescriptorName"):
                if mesh_elem.text:
                    mesh_terms.append(mesh_elem.text)
            
            # Keywords
            keywords = []
            for keyword_elem in medline_citation.findall(".//Keyword"):
                if keyword_elem.text:
                    keywords.append(keyword_elem.text)
            
            # DOI
            doi = ""
            for article_id_elem in medline_citation.findall(".//ArticleId"):
                if article_id_elem.get("IdType") == "doi":
                    doi = article_id_elem.text or ""
                    break
            
            # PMC ID
            pmc_id = ""
            for article_id_elem in medline_citation.findall(".//ArticleId"):
                if article_id_elem.get("IdType") == "pmc":
                    pmc_id = article_id_elem.text or ""
                    break
            
            return {
                "pmid": pmid,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "journal": journal_title,
                "publication_date": publication_date,
                "volume": volume,
                "issue": issue,
                "doi": doi,
                "pmc_id": pmc_id,
                "mesh_terms": mesh_terms,
                "keywords": keywords,
                "source": "PubMed",
                "full_text": f"{title}\n\n{abstract}",
                "metadata": {
                    "glaucoma_relevance_score": self._calculate_glaucoma_relevance(title, abstract, mesh_terms, keywords),
                    "article_type": self._determine_article_type(title, abstract),
                    "extracted_date": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting article data: {e}")
            return None
    
    def _extract_pmc_data(self, article_elem: ET.Element) -> Optional[Dict[str, Any]]:
        """Extract data from PMC Article element (simplified)"""
        # This would be similar to _extract_article_data but for PMC format
        # For now, return None as PMC has a different XML structure
        return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        if not text:
            return ""
        
        # Remove XML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def _calculate_glaucoma_relevance(self, title: str, abstract: str, 
                                    mesh_terms: List[str], keywords: List[str]) -> float:
        """Calculate relevance score for glaucoma research"""
        score = 0.0
        text_content = f"{title} {abstract}".lower()
        
        # MeSH term scoring (highest weight)
        for mesh_term in mesh_terms:
            if any(gterm.lower().replace("[mesh]", "") in mesh_term.lower() 
                   for gterm in self.glaucoma_mesh_terms):
                score += 10.0
        
        # Keyword scoring
        for keyword in keywords:
            if any(gterm.lower() in keyword.lower() for gterm in self.glaucoma_keywords):
                score += 5.0
        
        # Title scoring (high weight)
        title_lower = title.lower()
        for term in self.glaucoma_keywords:
            if term.lower() in title_lower:
                score += 3.0
        
        # Abstract scoring
        for term in self.glaucoma_keywords:
            if term.lower() in text_content:
                score += 1.0
                # Boost for multiple occurrences
                count = text_content.count(term.lower())
                score += min(count - 1, 2) * 0.5
        
        return min(score, 100.0)  # Cap at 100
    
    def _determine_article_type(self, title: str, abstract: str) -> str:
        """Determine the type of research article"""
        text_content = f"{title} {abstract}".lower()
        
        if any(term in text_content for term in ["systematic review", "meta-analysis", "meta analysis"]):
            return "systematic_review"
        elif any(term in text_content for term in ["randomized", "clinical trial", "rct"]):
            return "clinical_trial"
        elif any(term in text_content for term in ["case report", "case study"]):
            return "case_report"
        elif any(term in text_content for term in ["cohort", "longitudinal", "follow-up"]):
            return "cohort_study"
        elif any(term in text_content for term in ["cross-sectional", "prevalence"]):
            return "cross_sectional"
        elif any(term in text_content for term in ["in vitro", "cell culture"]):
            return "laboratory_study"
        elif any(term in text_content for term in ["review", "overview"]):
            return "review"
        else:
            return "research_article"
    
    def get_document(self, document_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get specific document by PMID"""
        try:
            articles = self._fetch_article_details([document_id], "pubmed")
            return articles[0] if articles else None
        except Exception as e:
            logger.error(f"Error fetching document {document_id}: {e}")
            return None
    
    def search_glaucoma_corpus(self, 
                              subtopics: Optional[List[str]] = None,
                              max_results: int = 1000,
                              years_back: int = 10) -> List[Dict[str, Any]]:
        """
        Comprehensive glaucoma corpus search
        
        Args:
            subtopics: Specific glaucoma subtopics to focus on
            max_results: Maximum number of articles to retrieve
            years_back: Number of years back to search
            
        Returns:
            List of glaucoma research articles
        """
        # Build comprehensive glaucoma query
        base_query = "glaucoma[MeSH] OR (glaucoma[TIAB] AND (intraocular pressure[TIAB] OR optic nerve[TIAB]))"
        
        if subtopics:
            subtopic_terms = " OR ".join([f"{topic}[TIAB]" for topic in subtopics])
            base_query += f" AND ({subtopic_terms})"
        
        # Add date range
        current_year = datetime.now().year
        start_year = current_year - years_back
        date_range = (f"{start_year}/01/01", f"{current_year}/12/31")
        
        # Search with high-quality article types
        article_types = ["Journal Article", "Review", "Clinical Trial", "Meta-Analysis"]
        
        return self.search(
            query=base_query,
            max_results=max_results,
            date_range=date_range,
            article_types=article_types
        )