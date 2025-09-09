"""
Clinical Trials API Client for Glaucoma Research
ClinicalTrials.gov API client for accessing clinical trial data
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_client import BaseAPIClient
from ..auth_manager import APIAuthManager
from ..error_manager import APIError, DataProcessingError, ErrorDetails, ErrorCategory, ErrorSeverity

import logging
logger = logging.getLogger(__name__)

class ClinicalTrialsClient(BaseAPIClient):
    """
    ClinicalTrials.gov API client for accessing clinical trial information
    
    Features:
    - Search clinical trials related to glaucoma
    - Extract trial metadata and status
    - Filter by trial phase, status, and location
    - Process trial results and outcomes
    """
    
    def __init__(self, auth_manager: APIAuthManager):
        super().__init__(auth_manager, "clinical_trials", max_requests_per_second=2)
        
        # Clinical trials specific configuration
        self.base_api_url = "https://clinicaltrials.gov/api/query/"
        
        # Glaucoma-related condition terms
        self.glaucoma_conditions = [
            "Glaucoma",
            "Primary Open Angle Glaucoma",
            "Angle Closure Glaucoma", 
            "Normal Tension Glaucoma",
            "Secondary Glaucoma",
            "Intraocular Pressure",
            "Optic Nerve Disease"
        ]
        
    def search(self, 
               query: str,
               max_results: int = 100,
               status: Optional[List[str]] = None,
               phase: Optional[List[str]] = None,
               **kwargs) -> List[Dict[str, Any]]:
        """
        Search clinical trials
        
        Args:
            query: Search query string
            max_results: Maximum number of results
            status: Trial status filter (e.g., ['Recruiting', 'Completed'])
            phase: Trial phase filter (e.g., ['Phase 2', 'Phase 3'])
            
        Returns:
            List of clinical trial data
        """
        try:
            # Enhance query for glaucoma research
            enhanced_query = self._enhance_glaucoma_query(query)
            
            # Build search parameters
            params = {
                "cond": enhanced_query,
                "fmt": "json",
                "min_rnk": 1,
                "max_rnk": min(max_results, 1000)  # API limit
            }
            
            # Add status filter
            if status:
                params["recrs"] = ",".join(status)
                
            # Add phase filter  
            if phase:
                params["phase"] = ",".join(phase)
                
            logger.info(f"Searching clinical trials with query: {enhanced_query}")
            
            response = self.get("study_fields", params=params)
            
            if not response.success:
                raise APIError(
                    ErrorDetails(
                        category=ErrorCategory.API,
                        severity=ErrorSeverity.HIGH,
                        message="Failed to search clinical trials",
                        status_code=response.status_code
                    )
                )
            
            # Parse results
            trials = self._parse_trials_response(response.data)
            
            logger.info(f"Found {len(trials)} clinical trials")
            return trials
            
        except Exception as e:
            logger.error(f"Error searching clinical trials: {e}")
            raise APIError(
                ErrorDetails(
                    category=ErrorCategory.API,
                    severity=ErrorSeverity.HIGH,
                    message=f"Clinical trials search failed: {str(e)}",
                    api_endpoint="study_fields"
                )
            )
    
    def _enhance_glaucoma_query(self, query: str) -> str:
        """Enhance search query for glaucoma relevance"""
        query_lower = query.lower()
        
        # If query doesn't contain glaucoma terms, add them
        has_glaucoma_terms = any(condition.lower() in query_lower 
                                for condition in self.glaucoma_conditions)
        
        if not has_glaucoma_terms and "glaucoma" not in query_lower:
            query = f"glaucoma OR {query}"
            
        return query
    
    def _parse_trials_response(self, response_data: Any) -> List[Dict[str, Any]]:
        """Parse clinical trials API response"""
        trials = []
        
        try:
            if isinstance(response_data, dict):
                studies = response_data.get("StudyFieldsResponse", {}).get("StudyFields", [])
            else:
                studies = []
            
            for study in studies:
                trial = self._extract_trial_data(study)
                if trial:
                    trials.append(trial)
                    
        except Exception as e:
            logger.error(f"Error parsing trials response: {e}")
            raise DataProcessingError(
                ErrorDetails(
                    category=ErrorCategory.DATA_PROCESSING,
                    severity=ErrorSeverity.MEDIUM,
                    message=f"Failed to parse clinical trials response: {e}"
                )
            )
        
        return trials
    
    def _extract_trial_data(self, study: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract structured data from a single trial"""
        try:
            # Helper function to safely extract field
            def get_field(field_name: str, default="") -> str:
                field_data = study.get(field_name, [])
                if isinstance(field_data, list) and field_data:
                    return field_data[0] if field_data[0] else default
                return str(field_data) if field_data else default
            
            # Basic information
            nct_id = get_field("NCTId")
            title = get_field("BriefTitle")
            summary = get_field("BriefSummary")
            detailed_description = get_field("DetailedDescription")
            
            # Status and phase
            status = get_field("OverallStatus")
            phase = get_field("Phase")
            study_type = get_field("StudyType")
            
            # Dates
            start_date = get_field("StartDate")
            completion_date = get_field("CompletionDate")
            
            # Sponsor and investigators
            sponsor = get_field("LeadSponsorName")
            investigators = study.get("OverallOfficialName", [])
            
            # Locations
            locations = study.get("LocationFacility", [])
            countries = study.get("LocationCountry", [])
            
            # Conditions and interventions
            conditions = study.get("Condition", [])
            interventions = study.get("InterventionName", [])
            
            # Eligibility
            eligibility_criteria = get_field("EligibilityCriteria")
            min_age = get_field("MinimumAge")
            max_age = get_field("MaximumAge")
            gender = get_field("Gender")
            
            # Outcomes
            primary_outcomes = study.get("PrimaryOutcomeMeasure", [])
            secondary_outcomes = study.get("SecondaryOutcomeMeasure", [])
            
            # Create full text
            full_text_parts = [
                f"Title: {title}",
                f"Summary: {summary}",
                f"Description: {detailed_description}",
                f"Conditions: {', '.join(conditions)}",
                f"Interventions: {', '.join(interventions)}",
                f"Eligibility: {eligibility_criteria}"
            ]
            full_text = "\n\n".join([part for part in full_text_parts if part.split(": ", 1)[1]])
            
            return {
                "nct_id": nct_id,
                "title": title,
                "abstract": summary,
                "full_text": full_text,
                "description": detailed_description,
                "status": status,
                "phase": phase,
                "study_type": study_type,
                "start_date": start_date,
                "completion_date": completion_date,
                "sponsor": sponsor,
                "investigators": investigators,
                "locations": locations,
                "countries": countries,
                "conditions": conditions,
                "interventions": interventions,
                "eligibility_criteria": eligibility_criteria,
                "min_age": min_age,
                "max_age": max_age,
                "gender": gender,
                "primary_outcomes": primary_outcomes,
                "secondary_outcomes": secondary_outcomes,
                "source": "ClinicalTrials.gov",
                "publication_date": start_date,
                "authors": investigators,
                "journal": "ClinicalTrials.gov Registry",
                "doi": f"https://clinicaltrials.gov/ct2/show/{nct_id}",
                "keywords": conditions + interventions,
                "mesh_terms": conditions,
                "metadata": {
                    "glaucoma_relevance_score": self._calculate_trial_relevance(title, summary, conditions),
                    "article_type": "clinical_trial",
                    "extracted_date": datetime.now().isoformat(),
                    "trial_phase": phase,
                    "trial_status": status
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting trial data: {e}")
            return None
    
    def _calculate_trial_relevance(self, title: str, summary: str, conditions: List[str]) -> float:
        """Calculate relevance score for glaucoma research"""
        score = 0.0
        text_content = f"{title} {summary}".lower()
        
        # Condition-based scoring (highest weight)
        for condition in conditions:
            condition_lower = condition.lower()
            if "glaucoma" in condition_lower:
                if "primary" in condition_lower and "open" in condition_lower:
                    score += 25.0  # Primary open-angle glaucoma
                elif "angle" in condition_lower and "closure" in condition_lower:
                    score += 25.0  # Angle-closure glaucoma
                elif "normal tension" in condition_lower:
                    score += 20.0  # Normal tension glaucoma
                else:
                    score += 15.0  # General glaucoma
            elif "intraocular pressure" in condition_lower:
                score += 10.0
            elif "optic nerve" in condition_lower:
                score += 8.0
        
        # Title scoring
        title_lower = title.lower()
        if "glaucoma" in title_lower:
            score += 15.0
        if "intraocular pressure" in title_lower or "iop" in title_lower:
            score += 10.0
        
        # Summary scoring
        glaucoma_terms = [
            "glaucoma", "intraocular pressure", "iop", "optic nerve",
            "visual field", "trabecular", "aqueous humor"
        ]
        
        for term in glaucoma_terms:
            if term in text_content:
                score += 3.0
                # Count occurrences
                count = text_content.count(term)
                score += min(count - 1, 3) * 1.0
        
        return min(score, 100.0)
    
    def get_document(self, document_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get specific clinical trial by NCT ID"""
        try:
            params = {
                "expr": document_id,
                "fmt": "json",
                "min_rnk": 1,
                "max_rnk": 1
            }
            
            response = self.get("study_fields", params=params)
            
            if response.success:
                trials = self._parse_trials_response(response.data)
                return trials[0] if trials else None
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error fetching trial {document_id}: {e}")
            return None
    
    def search_glaucoma_trials(self, 
                              trial_types: Optional[List[str]] = None,
                              phases: Optional[List[str]] = None,
                              statuses: Optional[List[str]] = None,
                              max_results: int = 500) -> List[Dict[str, Any]]:
        """
        Comprehensive glaucoma clinical trials search
        
        Args:
            trial_types: Types of trials to include
            phases: Trial phases to include
            statuses: Trial statuses to include
            max_results: Maximum results to return
            
        Returns:
            List of glaucoma clinical trials
        """
        # Default parameters for comprehensive glaucoma search
        if not trial_types:
            trial_types = ["Interventional", "Observational"]
        
        if not phases:
            phases = ["Phase 1", "Phase 2", "Phase 3", "Phase 4"]
            
        if not statuses:
            statuses = ["Recruiting", "Active, not recruiting", "Completed", "Enrolling by invitation"]
        
        # Search with glaucoma conditions
        all_trials = []
        
        for condition in self.glaucoma_conditions:
            try:
                trials = self.search(
                    query=condition,
                    max_results=max_results // len(self.glaucoma_conditions),
                    status=statuses,
                    phase=phases
                )
                all_trials.extend(trials)
                
            except Exception as e:
                logger.warning(f"Error searching for condition '{condition}': {e}")
                continue
        
        # Remove duplicates based on NCT ID
        seen_ids = set()
        unique_trials = []
        for trial in all_trials:
            nct_id = trial.get("nct_id")
            if nct_id and nct_id not in seen_ids:
                seen_ids.add(nct_id)
                unique_trials.append(trial)
        
        logger.info(f"Found {len(unique_trials)} unique glaucoma clinical trials")
        return unique_trials