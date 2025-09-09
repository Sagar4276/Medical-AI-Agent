"""
Authentication Manager for Glaucoma Corpus Builder
Secure handling of API keys and authentication for multiple medical research APIs
"""

import os
import json
import logging
from typing import Dict, Optional, Any, List
from dataclasses import dataclass, asdict
from cryptography.fernet import Fernet
from dotenv import load_dotenv
from .error_manager import AuthenticationError, ErrorDetails, ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)

@dataclass
class APICredential:
    """Container for API credential information"""
    name: str
    api_key: str
    base_url: str
    rate_limit: Optional[int] = None
    additional_headers: Optional[Dict[str, str]] = None
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary excluding sensitive data"""
        data = asdict(self)
        # Mask the API key for logging/display
        if data['api_key']:
            data['api_key'] = f"{data['api_key'][:8]}***"
        return data


class APIAuthManager:
    """
    Comprehensive authentication manager for medical research APIs
    
    Supports:
    - Multiple API providers (PubMed, NIH, FDA, etc.)
    - Secure credential storage with encryption
    - Environment variable management
    - Rate limiting awareness
    - Credential validation
    """
    
    # Default API configurations for major medical research APIs
    DEFAULT_API_CONFIGS = {
        "pubmed": {
            "base_url": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/",
            "rate_limit": 10,  # requests per second
            "description": "NCBI PubMed E-utilities API for biomedical literature",
            "env_key": "PUBMED_API_KEY"
        },
        "clinical_trials": {
            "base_url": "https://clinicaltrials.gov/api/",
            "rate_limit": 20,
            "description": "ClinicalTrials.gov API for clinical trial data",
            "env_key": "CLINICAL_TRIALS_API_KEY"
        },
        "openfda": {
            "base_url": "https://api.fda.gov/",
            "rate_limit": 40,
            "description": "FDA openFDA API for drug and device information",
            "env_key": "OPENFDA_API_KEY"
        },
        "who_gho": {
            "base_url": "https://ghoapi.azureedge.net/api/",
            "rate_limit": 10,
            "description": "WHO Global Health Observatory API",
            "env_key": "WHO_API_KEY"
        },
        "umls": {
            "base_url": "https://uts-ws.nlm.nih.gov/rest/",
            "rate_limit": 5,
            "description": "UMLS Terminology Services API",
            "env_key": "UMLS_API_KEY"
        },
        "europepmc": {
            "base_url": "https://www.ebi.ac.uk/europepmc/webservices/rest/",
            "rate_limit": 10,
            "description": "Europe PMC API for life science literature",
            "env_key": "EUROPEPMC_API_KEY"
        },
        "crossref": {
            "base_url": "https://api.crossref.org/",
            "rate_limit": 50,
            "description": "Crossref API for scholarly metadata",
            "env_key": "CROSSREF_API_KEY"
        }
    }
    
    def __init__(self, 
                 credentials_file: Optional[str] = None,
                 encryption_key: Optional[bytes] = None):
        """
        Initialize the authentication manager
        
        Args:
            credentials_file: Path to encrypted credentials file
            encryption_key: Encryption key for credential security
        """
        load_dotenv()  # Load environment variables
        
        self.credentials: Dict[str, APICredential] = {}
        self.credentials_file = credentials_file or os.path.join(
            os.path.dirname(__file__), "..", "data", "credentials.enc"
        )
        
        # Set up encryption
        if encryption_key:
            self.cipher = Fernet(encryption_key)
        else:
            # Generate or load encryption key
            key_file = os.path.join(os.path.dirname(self.credentials_file), "auth.key")
            self.cipher = self._get_or_create_cipher(key_file)
        
        self._load_credentials()
        
    def _get_or_create_cipher(self, key_file: str) -> Fernet:
        """Get or create encryption cipher"""
        try:
            if os.path.exists(key_file):
                with open(key_file, 'rb') as f:
                    key = f.read()
            else:
                key = Fernet.generate_key()
                os.makedirs(os.path.dirname(key_file), exist_ok=True)
                with open(key_file, 'wb') as f:
                    f.write(key)
                logger.info(f"Generated new encryption key: {key_file}")
                
            return Fernet(key)
        except Exception as e:
            logger.error(f"Failed to initialize encryption: {e}")
            raise AuthenticationError(
                ErrorDetails(
                    category=ErrorCategory.AUTHENTICATION,
                    severity=ErrorSeverity.CRITICAL,
                    message=f"Failed to initialize credential encryption: {e}"
                )
            )
    
    def _load_credentials(self):
        """Load credentials from environment and encrypted file"""
        # Load from environment variables
        self._load_from_environment()
        
        # Load from encrypted file
        self._load_from_file()
        
        logger.info(f"Loaded credentials for {len(self.credentials)} APIs")
        
    def _load_from_environment(self):
        """Load API credentials from environment variables"""
        for api_name, config in self.DEFAULT_API_CONFIGS.items():
            env_key = config["env_key"]
            api_key = os.getenv(env_key)
            
            if api_key:
                credential = APICredential(
                    name=api_name,
                    api_key=api_key,
                    base_url=config["base_url"],
                    rate_limit=config["rate_limit"],
                    description=config["description"]
                )
                self.credentials[api_name] = credential
                logger.debug(f"Loaded {api_name} credentials from environment")
                
    def _load_from_file(self):
        """Load credentials from encrypted file"""
        if not os.path.exists(self.credentials_file):
            return
            
        try:
            with open(self.credentials_file, 'rb') as f:
                encrypted_data = f.read()
                
            decrypted_data = self.cipher.decrypt(encrypted_data)
            credentials_data = json.loads(decrypted_data.decode())
            
            for api_name, cred_data in credentials_data.items():
                if api_name not in self.credentials:  # Don't override env vars
                    credential = APICredential(**cred_data)
                    self.credentials[api_name] = credential
                    logger.debug(f"Loaded {api_name} credentials from file")
                    
        except Exception as e:
            logger.warning(f"Failed to load credentials from file: {e}")
    
    def add_credential(self, 
                      api_name: str,
                      api_key: str,
                      base_url: Optional[str] = None,
                      rate_limit: Optional[int] = None,
                      additional_headers: Optional[Dict[str, str]] = None,
                      description: Optional[str] = None) -> bool:
        """
        Add or update API credential
        
        Args:
            api_name: Name of the API
            api_key: API key or token
            base_url: Base URL for the API
            rate_limit: Rate limit (requests per second)
            additional_headers: Additional headers needed
            description: Description of the API
            
        Returns:
            bool: True if credential was added successfully
        """
        try:
            # Use default config if available
            if api_name in self.DEFAULT_API_CONFIGS and base_url is None:
                config = self.DEFAULT_API_CONFIGS[api_name]
                base_url = config["base_url"]
                rate_limit = rate_limit or config["rate_limit"]
                description = description or config["description"]
            
            if not base_url:
                raise ValueError(f"Base URL is required for API '{api_name}'")
                
            credential = APICredential(
                name=api_name,
                api_key=api_key,
                base_url=base_url,
                rate_limit=rate_limit,
                additional_headers=additional_headers,
                description=description
            )
            
            # Validate the credential
            if self.validate_credential(credential):
                self.credentials[api_name] = credential
                self._save_credentials()
                logger.info(f"Added credential for {api_name}")
                return True
            else:
                raise ValueError(f"Invalid credential for {api_name}")
                
        except Exception as e:
            logger.error(f"Failed to add credential for {api_name}: {e}")
            raise AuthenticationError(
                ErrorDetails(
                    category=ErrorCategory.AUTHENTICATION,
                    severity=ErrorSeverity.HIGH,
                    message=f"Failed to add credential for {api_name}: {e}"
                )
            )
    
    def get_credential(self, api_name: str) -> Optional[APICredential]:
        """Get API credential by name"""
        credential = self.credentials.get(api_name)
        if not credential:
            logger.warning(f"No credential found for API '{api_name}'")
            return None
        return credential
    
    def get_api_key(self, api_name: str) -> Optional[str]:
        """Get API key for specific API"""
        credential = self.get_credential(api_name)
        return credential.api_key if credential else None
    
    def get_headers(self, api_name: str) -> Dict[str, str]:
        """Get headers for API requests"""
        credential = self.get_credential(api_name)
        if not credential:
            return {}
            
        headers = {"User-Agent": "Medical-AI-Agent-Glaucoma-Corpus-Builder/1.0"}
        
        # Add API key to headers (common patterns)
        if credential.api_key:
            # Different APIs use different header patterns
            if api_name in ["pubmed", "umls"]:
                headers["api_key"] = credential.api_key
            elif api_name == "openfda":
                # OpenFDA uses query parameter, not header
                pass
            else:
                headers["Authorization"] = f"Bearer {credential.api_key}"
        
        # Add any additional headers
        if credential.additional_headers:
            headers.update(credential.additional_headers)
            
        return headers
    
    def validate_credential(self, credential: APICredential) -> bool:
        """
        Validate API credential
        
        Args:
            credential: Credential to validate
            
        Returns:
            bool: True if credential is valid
        """
        try:
            # Basic validation
            if not credential.api_key or not credential.base_url:
                return False
                
            # API-specific validation could be added here
            # For now, just check format
            if len(credential.api_key) < 8:
                logger.warning(f"API key for {credential.name} seems too short")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating credential for {credential.name}: {e}")
            return False
    
    def _save_credentials(self):
        """Save credentials to encrypted file"""
        try:
            # Prepare data for encryption (exclude env-loaded credentials)
            credentials_data = {}
            for api_name, credential in self.credentials.items():
                # Only save non-environment credentials
                env_key = self.DEFAULT_API_CONFIGS.get(api_name, {}).get("env_key")
                if not env_key or not os.getenv(env_key):
                    credentials_data[api_name] = asdict(credential)
            
            if not credentials_data:
                return  # No credentials to save
                
            # Encrypt and save
            json_data = json.dumps(credentials_data)
            encrypted_data = self.cipher.encrypt(json_data.encode())
            
            os.makedirs(os.path.dirname(self.credentials_file), exist_ok=True)
            with open(self.credentials_file, 'wb') as f:
                f.write(encrypted_data)
                
            logger.debug(f"Saved {len(credentials_data)} credentials to file")
            
        except Exception as e:
            logger.error(f"Failed to save credentials: {e}")
    
    def list_available_apis(self) -> List[Dict[str, Any]]:
        """List all available API configurations"""
        apis = []
        for api_name, credential in self.credentials.items():
            api_info = credential.to_dict()
            api_info["available"] = True
            apis.append(api_info)
            
        # Add unconfigured APIs
        for api_name, config in self.DEFAULT_API_CONFIGS.items():
            if api_name not in self.credentials:
                apis.append({
                    "name": api_name,
                    "base_url": config["base_url"],
                    "description": config["description"],
                    "available": False,
                    "env_key": config["env_key"]
                })
        
        return sorted(apis, key=lambda x: x["name"])
    
    def get_rate_limit(self, api_name: str) -> Optional[int]:
        """Get rate limit for specific API"""
        credential = self.get_credential(api_name)
        return credential.rate_limit if credential else None
    
    def remove_credential(self, api_name: str) -> bool:
        """Remove API credential"""
        if api_name in self.credentials:
            del self.credentials[api_name]
            self._save_credentials()
            logger.info(f"Removed credential for {api_name}")
            return True
        return False
    
    def test_credential(self, api_name: str) -> bool:
        """
        Test API credential by making a simple request
        
        Args:
            api_name: Name of the API to test
            
        Returns:
            bool: True if credential is working
        """
        credential = self.get_credential(api_name)
        if not credential:
            return False
            
        try:
            import requests
            
            # Simple test endpoints for each API
            test_endpoints = {
                "pubmed": "einfo.fcgi",
                "clinical_trials": "query/study_fields",
                "openfda": "drug/event.json?limit=1",
                "who_gho": "",  # Root endpoint
                "umls": "version",
                "europepmc": "search?query=diabetes&pageSize=1",
                "crossref": "works?rows=1"
            }
            
            if api_name not in test_endpoints:
                logger.warning(f"No test endpoint defined for {api_name}")
                return True  # Assume valid if we can't test
                
            test_url = credential.base_url + test_endpoints[api_name]
            headers = self.get_headers(api_name)
            
            response = requests.get(test_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"Credential test for {api_name}: SUCCESS")
                return True
            else:
                logger.warning(f"Credential test for {api_name}: FAILED (Status: {response.status_code})")
                return False
                
        except Exception as e:
            logger.error(f"Error testing credential for {api_name}: {e}")
            return False