"""
Europe PMC API Client for Glaucoma Research  
"""

from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_client import BaseAPIClient
from ..auth_manager import APIAuthManager
from ..error_manager import APIError

import logging
logger = logging.getLogger(__name__)

class EuropePMCClient(BaseAPIClient):
    """Europe PMC API client for life sciences literature"""
    
    def __init__(self, auth_manager: APIAuthManager):
        super().__init__(auth_manager, "europepmc", max_requests_per_second=3)
    
    def search(self, query: str, max_results: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """Search Europe PMC"""
        try:
            params = {
                "query": f"({query}) AND (glaucoma)",
                "pageSize": min(max_results, 1000),
                "format": "json",
                "resultType": "core"
            }
            
            response = self.get("search", params=params)
            
            if response.success:
                results = response.data.get("resultList", {}).get("result", [])
                return [self._convert_europepmc_result(result) for result in results]
            else:
                return []
                
        except Exception as e:
            logger.error(f"Europe PMC search error: {e}")
            return []
    
    def _convert_europepmc_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Convert Europe PMC result to standard format"""
        return {
            "id": result.get("pmid", result.get("id", "")),
            "title": result.get("title", ""),
            "abstract": result.get("abstractText", ""),
            "full_text": f"{result.get('title', '')}\n\n{result.get('abstractText', '')}",
            "authors": [f"{author.get('firstName', '')} {author.get('lastName', '')}" 
                       for author in result.get("authorList", {}).get("author", [])],
            "publication_date": result.get("firstPublicationDate", ""),
            "journal": result.get("journalTitle", ""),
            "doi": result.get("doi", ""),
            "source": "Europe PMC",
            "keywords": result.get("keywordList", {}).get("keyword", []),
            "mesh_terms": [],
            "metadata": {
                "glaucoma_relevance_score": 50.0,  # Default score
                "article_type": "research_article",
                "extracted_date": datetime.now().isoformat()
            }
        }
    
    def get_document(self, document_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get Europe PMC document by ID"""
        return None