"""
OpenFDA API Client for Glaucoma Research
FDA openFDA API client for drug and device information
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json

from .base_client import BaseAPIClient
from ..auth_manager import APIAuthManager
from ..error_manager import APIError, DataProcessingError, ErrorDetails, ErrorCategory, ErrorSeverity

import logging
logger = logging.getLogger(__name__)

class OpenFDAClient(BaseAPIClient):
    """
    OpenFDA API client for accessing FDA drug and device data
    
    Features:
    - Search for glaucoma-related drugs and adverse events
    - Extract drug labeling information
    - Process adverse event reports
    - Analyze device safety data
    """
    
    def __init__(self, auth_manager: APIAuthManager):
        super().__init__(auth_manager, "openfda", max_requests_per_second=4)
        
        # OpenFDA endpoints
        self.endpoints = {
            "drug_events": "drug/event.json",
            "drug_labels": "drug/label.json", 
            "device_events": "device/event.json",
            "device_510k": "device/510k.json",
            "device_pma": "device/pma.json"
        }
        
        # Glaucoma-related drug terms
        self.glaucoma_drugs = [
            "latanoprost", "timolol", "brimonidine", "dorzolamide", "brinzolamide",
            "travoprost", "bimatoprost", "tafluprost", "betaxolol", "levobunolol",
            "acetazolamide", "methazolamide", "pilocarpine", "carbachol"
        ]
        
        # Glaucoma-related device terms
        self.glaucoma_devices = [
            "tonometer", "perimeter", "gonioscope", "pachymeter",
            "glaucoma drainage device", "tube shunt", "glaucoma implant",
            "oct", "optical coherence tomography"
        ]
        
    def search(self, 
               query: str,
               endpoint: str = "drug_events",
               max_results: int = 100,
               **kwargs) -> List[Dict[str, Any]]:
        """
        Search OpenFDA data
        
        Args:
            query: Search query string
            endpoint: OpenFDA endpoint to search
            max_results: Maximum number of results
            
        Returns:
            List of FDA data records
        """
        try:
            if endpoint not in self.endpoints:
                raise ValueError(f"Unknown endpoint: {endpoint}")
            
            # Build search parameters
            params = {
                "search": self._build_search_query(query, endpoint),
                "limit": min(max_results, 1000)  # API limit
            }
            
            # Add API key if available (as query parameter for OpenFDA)
            if self.credential.api_key:
                params["api_key"] = self.credential.api_key
            
            logger.info(f"Searching OpenFDA {endpoint} with query: {query}")
            
            response = self.get(self.endpoints[endpoint], params=params)
            
            if not response.success:
                raise APIError(
                    ErrorDetails(
                        category=ErrorCategory.API,
                        severity=ErrorSeverity.HIGH,
                        message="Failed to search OpenFDA",
                        status_code=response.status_code
                    )
                )
            
            # Parse results
            records = self._parse_fda_response(response.data, endpoint)
            
            logger.info(f"Found {len(records)} FDA records")
            return records
            
        except Exception as e:
            logger.error(f"Error searching OpenFDA: {e}")
            raise APIError(
                ErrorDetails(
                    category=ErrorCategory.API,
                    severity=ErrorSeverity.HIGH,
                    message=f"OpenFDA search failed: {str(e)}",
                    api_endpoint=endpoint
                )
            )
    
    def _build_search_query(self, query: str, endpoint: str) -> str:
        """Build appropriate search query for endpoint"""
        query_lower = query.lower()
        
        if endpoint == "drug_events":
            # Search in patient drug reactions and indications
            if any(drug in query_lower for drug in self.glaucoma_drugs):
                return f"patient.drug.medicinalproduct:\"{query}\" OR patient.drug.openfda.generic_name:\"{query}\""
            else:
                return f"patient.reaction.reactionmeddrapt:\"{query}\" OR patient.drug.drugindication:\"{query}\""
                
        elif endpoint == "drug_labels":
            # Search in drug labeling
            return f"openfda.generic_name:\"{query}\" OR indications_and_usage:\"{query}\" OR description:\"{query}\""
            
        elif endpoint == "device_events":
            # Search in device adverse events
            return f"device.generic_name:\"{query}\" OR device.brand_name:\"{query}\" OR mdr_report_key:\"{query}\""
            
        else:
            # Generic search
            return f"\"{query}\""
    
    def _parse_fda_response(self, response_data: Any, endpoint: str) -> List[Dict[str, Any]]:
        """Parse OpenFDA API response"""
        records = []
        
        try:
            results = response_data.get("results", [])
            
            for result in results:
                if endpoint == "drug_events":
                    record = self._extract_drug_event_data(result)
                elif endpoint == "drug_labels":
                    record = self._extract_drug_label_data(result)
                elif endpoint == "device_events":
                    record = self._extract_device_event_data(result)
                else:
                    record = self._extract_generic_data(result, endpoint)
                
                if record:
                    records.append(record)
                    
        except Exception as e:
            logger.error(f"Error parsing FDA response: {e}")
            raise DataProcessingError(
                ErrorDetails(
                    category=ErrorCategory.DATA_PROCESSING,
                    severity=ErrorSeverity.MEDIUM,
                    message=f"Failed to parse OpenFDA response: {e}"
                )
            )
        
        return records
    
    def _extract_drug_event_data(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract data from drug adverse event"""
        try:
            # Event information
            report_id = event.get("safetyreportid", "")
            report_date = event.get("receiptdate", "")
            
            # Patient information
            patient = event.get("patient", {})
            reactions = patient.get("reaction", [])
            drugs = patient.get("drug", [])
            
            # Extract reaction information
            reaction_terms = []
            for reaction in reactions:
                term = reaction.get("reactionmeddrapt", "")
                if term:
                    reaction_terms.append(term)
            
            # Extract drug information
            drug_names = []
            drug_indications = []
            for drug in drugs:
                # Generic name
                openfda = drug.get("openfda", {})
                generic_names = openfda.get("generic_name", [])
                drug_names.extend(generic_names)
                
                # Brand name
                brand_names = openfda.get("brand_name", [])
                drug_names.extend(brand_names)
                
                # Medicinal product
                medicinal = drug.get("medicinalproduct", "")
                if medicinal:
                    drug_names.append(medicinal)
                
                # Indication
                indication = drug.get("drugindication", "")
                if indication:
                    drug_indications.append(indication)
            
            # Create content
            title = f"Drug Adverse Event: {', '.join(reaction_terms[:3])}"
            abstract = f"Adverse event involving drugs: {', '.join(set(drug_names[:5]))}. Reactions: {', '.join(reaction_terms[:5])}."
            
            full_text_parts = [
                f"Title: {title}",
                f"Report ID: {report_id}",
                f"Drugs: {', '.join(set(drug_names))}",
                f"Reactions: {', '.join(reaction_terms)}",
                f"Indications: {', '.join(drug_indications)}"
            ]
            full_text = "\n\n".join(full_text_parts)
            
            return {
                "id": report_id,
                "title": title,
                "abstract": abstract,
                "full_text": full_text,
                "publication_date": report_date,
                "source": "FDA Adverse Event Reports",
                "authors": ["FDA"],
                "journal": "FDA FAERS Database",
                "doi": "",
                "keywords": drug_names + reaction_terms,
                "mesh_terms": drug_indications,
                "metadata": {
                    "glaucoma_relevance_score": self._calculate_drug_relevance(drug_names, drug_indications, reaction_terms),
                    "article_type": "adverse_event_report",
                    "extracted_date": datetime.now().isoformat(),
                    "report_type": "drug_adverse_event"
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting drug event data: {e}")
            return None
    
    def _extract_drug_label_data(self, label: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract data from drug labeling"""
        try:
            # OpenFDA information
            openfda = label.get("openfda", {})
            generic_names = openfda.get("generic_name", [])
            brand_names = openfda.get("brand_name", [])
            
            # Labeling information
            indications = label.get("indications_and_usage", [""])
            description = label.get("description", [""])
            warnings = label.get("warnings", [""])
            dosage = label.get("dosage_and_administration", [""])
            
            # Create title and content
            drug_name = generic_names[0] if generic_names else (brand_names[0] if brand_names else "Unknown Drug")
            title = f"Drug Label: {drug_name}"
            
            # Join list fields
            indications_text = " ".join(indications) if isinstance(indications, list) else str(indications)
            description_text = " ".join(description) if isinstance(description, list) else str(description)
            
            abstract = f"Drug labeling for {drug_name}. {indications_text[:200]}..."
            
            full_text_parts = [
                f"Title: {title}",
                f"Generic Names: {', '.join(generic_names)}",
                f"Brand Names: {', '.join(brand_names)}",
                f"Indications: {indications_text}",
                f"Description: {description_text}"
            ]
            full_text = "\n\n".join(full_text_parts)
            
            return {
                "id": f"label_{drug_name.replace(' ', '_')}",
                "title": title,
                "abstract": abstract,
                "full_text": full_text,
                "publication_date": "",
                "source": "FDA Drug Labels",
                "authors": ["FDA"],
                "journal": "FDA Drug Labeling Database",
                "doi": "",
                "keywords": generic_names + brand_names,
                "mesh_terms": [indications_text] if indications_text else [],
                "metadata": {
                    "glaucoma_relevance_score": self._calculate_drug_relevance(generic_names + brand_names, [indications_text], []),
                    "article_type": "drug_label",
                    "extracted_date": datetime.now().isoformat(),
                    "report_type": "drug_labeling"
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting drug label data: {e}")
            return None
    
    def _extract_device_event_data(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract data from device adverse event"""
        try:
            # Report information
            report_key = event.get("mdr_report_key", "")
            report_date = event.get("date_received", "")
            
            # Device information
            device = event.get("device", [])
            if device:
                device_info = device[0] if isinstance(device, list) else device
                device_name = device_info.get("generic_name", "")
                brand_name = device_info.get("brand_name", "")
                device_class = device_info.get("device_class", "")
            else:
                device_name = ""
                brand_name = ""
                device_class = ""
            
            # Event description
            event_description = event.get("mdr_text", [])
            if isinstance(event_description, list):
                event_text = " ".join([text.get("text", "") for text in event_description])
            else:
                event_text = str(event_description)
            
            title = f"Device Event: {device_name or brand_name}"
            abstract = f"Adverse event report for medical device. {event_text[:200]}..."
            
            full_text_parts = [
                f"Title: {title}",
                f"Report Key: {report_key}",
                f"Device: {device_name}",
                f"Brand: {brand_name}",
                f"Class: {device_class}",
                f"Description: {event_text}"
            ]
            full_text = "\n\n".join(full_text_parts)
            
            return {
                "id": report_key,
                "title": title,
                "abstract": abstract,
                "full_text": full_text,
                "publication_date": report_date,
                "source": "FDA Device Adverse Events",
                "authors": ["FDA"],
                "journal": "FDA MAUDE Database",
                "doi": "",
                "keywords": [device_name, brand_name] if device_name or brand_name else [],
                "mesh_terms": [],
                "metadata": {
                    "glaucoma_relevance_score": self._calculate_device_relevance(device_name, brand_name, event_text),
                    "article_type": "device_adverse_event",
                    "extracted_date": datetime.now().isoformat(),
                    "report_type": "device_adverse_event"
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting device event data: {e}")
            return None
    
    def _extract_generic_data(self, data: Dict[str, Any], endpoint: str) -> Optional[Dict[str, Any]]:
        """Extract generic data for unknown endpoints"""
        try:
            # Try to extract basic information
            title = "FDA Data Record"
            abstract = json.dumps(data)[:200] + "..."
            
            return {
                "id": str(hash(str(data))),
                "title": title,
                "abstract": abstract,
                "full_text": json.dumps(data, indent=2),
                "publication_date": "",
                "source": f"FDA {endpoint}",
                "authors": ["FDA"],
                "journal": f"FDA {endpoint} Database",
                "doi": "",
                "keywords": [],
                "mesh_terms": [],
                "metadata": {
                    "glaucoma_relevance_score": 0.0,
                    "article_type": "regulatory_data",
                    "extracted_date": datetime.now().isoformat(),
                    "report_type": endpoint
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting generic data: {e}")
            return None
    
    def _calculate_drug_relevance(self, drug_names: List[str], 
                                indications: List[str], reactions: List[str]) -> float:
        """Calculate relevance score for glaucoma drug data"""
        score = 0.0
        
        # Check for glaucoma drugs
        for drug_name in drug_names:
            if any(gd in drug_name.lower() for gd in self.glaucoma_drugs):
                score += 30.0
        
        # Check indications
        indications_text = " ".join(indications).lower()
        if "glaucoma" in indications_text:
            score += 25.0
        if "intraocular pressure" in indications_text or "iop" in indications_text:
            score += 15.0
        
        # Check reactions  
        reactions_text = " ".join(reactions).lower()
        if "glaucoma" in reactions_text:
            score += 20.0
        if "vision" in reactions_text or "eye" in reactions_text:
            score += 10.0
            
        return min(score, 100.0)
    
    def _calculate_device_relevance(self, device_name: str, brand_name: str, description: str) -> float:
        """Calculate relevance score for glaucoma device data"""
        score = 0.0
        
        # Check device names
        device_text = f"{device_name} {brand_name}".lower()
        for device in self.glaucoma_devices:
            if device in device_text:
                score += 25.0
        
        # Check description
        description_lower = description.lower()
        if "glaucoma" in description_lower:
            score += 30.0
        if "eye" in description_lower or "optic" in description_lower:
            score += 10.0
        if "pressure" in description_lower:
            score += 15.0
            
        return min(score, 100.0)
    
    def get_document(self, document_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Get specific FDA record by ID"""
        # OpenFDA doesn't support direct ID lookups easily
        # This would need to be implemented based on specific endpoint
        return None
    
    def search_glaucoma_drugs(self, max_results: int = 200) -> List[Dict[str, Any]]:
        """Search for glaucoma-related drugs"""
        all_records = []
        
        # Search drug events for glaucoma drugs
        for drug in self.glaucoma_drugs[:5]:  # Limit to avoid rate limits
            try:
                events = self.search(drug, "drug_events", max_results // len(self.glaucoma_drugs))
                all_records.extend(events)
            except Exception as e:
                logger.warning(f"Error searching drug events for {drug}: {e}")
        
        # Search drug labels
        try:
            labels = self.search("glaucoma", "drug_labels", max_results // 4)
            all_records.extend(labels)
        except Exception as e:
            logger.warning(f"Error searching drug labels: {e}")
        
        return all_records
    
    def search_glaucoma_devices(self, max_results: int = 100) -> List[Dict[str, Any]]:
        """Search for glaucoma-related medical devices"""
        all_records = []
        
        # Search device events
        for device in self.glaucoma_devices[:3]:  # Limit search terms
            try:
                events = self.search(device, "device_events", max_results // len(self.glaucoma_devices))
                all_records.extend(events)
            except Exception as e:
                logger.warning(f"Error searching device events for {device}: {e}")
        
        return all_records