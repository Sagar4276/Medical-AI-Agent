"""
Base API Client for Medical Research APIs
Common functionality and interface for all API clients
"""

import time
import requests
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass
from datetime import datetime, timedelta

from ..auth_manager import APIAuthManager, APICredential
from ..error_manager import APIError, NetworkError, ErrorDetails, ErrorCategory, ErrorSeverity, api_retry

logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    """Standardized API response container"""
    success: bool
    data: Any
    status_code: int
    headers: Dict[str, str]
    response_time: float
    api_name: str
    endpoint: str
    error_message: Optional[str] = None

@dataclass
class RateLimit:
    """Rate limiting information"""
    requests_per_second: int
    requests_per_minute: int
    requests_per_hour: int
    requests_per_day: int
    current_window_start: datetime
    current_requests: int

class BaseAPIClient(ABC):
    """
    Abstract base class for all medical research API clients
    
    Provides:
    - Authentication management
    - Rate limiting
    - Error handling with retries
    - Request logging and monitoring
    - Response standardization
    """
    
    def __init__(self, 
                 auth_manager: APIAuthManager,
                 api_name: str,
                 default_timeout: int = 30,
                 max_requests_per_second: Optional[int] = None):
        """
        Initialize base API client
        
        Args:
            auth_manager: Authentication manager instance
            api_name: Name of the API
            default_timeout: Default request timeout in seconds
            max_requests_per_second: Override rate limit
        """
        self.auth_manager = auth_manager
        self.api_name = api_name
        self.default_timeout = default_timeout
        
        # Get API credential
        self.credential = auth_manager.get_credential(api_name)
        if not self.credential:
            raise APIError(
                ErrorDetails(
                    category=ErrorCategory.AUTHENTICATION,
                    severity=ErrorSeverity.HIGH,
                    message=f"No credentials found for API '{api_name}'"
                )
            )
        
        # Set up rate limiting
        self.rate_limit = self._setup_rate_limiting(max_requests_per_second)
        self.last_request_time = 0.0
        self.request_count = 0
        
        # Request session for connection pooling
        self.session = requests.Session()
        self.session.headers.update(auth_manager.get_headers(api_name))
        
        logger.info(f"Initialized {api_name} API client")
    
    def _setup_rate_limiting(self, override_rate: Optional[int]) -> RateLimit:
        """Set up rate limiting based on API specifications"""
        if override_rate:
            rps = override_rate
        elif self.credential.rate_limit:
            rps = self.credential.rate_limit
        else:
            rps = 10  # Conservative default
            
        return RateLimit(
            requests_per_second=rps,
            requests_per_minute=rps * 60,
            requests_per_hour=rps * 3600,
            requests_per_day=rps * 86400,
            current_window_start=datetime.now(),
            current_requests=0
        )
    
    def _enforce_rate_limit(self):
        """Enforce rate limiting by sleeping if necessary"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < (1.0 / self.rate_limit.requests_per_second):
            sleep_time = (1.0 / self.rate_limit.requests_per_second) - time_since_last
            logger.debug(f"Rate limiting: sleeping {sleep_time:.3f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
        self.request_count += 1
    
    @api_retry
    def _make_request(self, 
                     method: str,
                     endpoint: str, 
                     params: Optional[Dict[str, Any]] = None,
                     data: Optional[Dict[str, Any]] = None,
                     headers: Optional[Dict[str, str]] = None,
                     timeout: Optional[int] = None) -> APIResponse:
        """
        Make HTTP request with comprehensive error handling
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (relative to base URL)
            params: Query parameters
            data: Request body data
            headers: Additional headers
            timeout: Request timeout
            
        Returns:
            APIResponse: Standardized response object
        """
        self._enforce_rate_limit()
        
        url = f"{self.credential.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        timeout = timeout or self.default_timeout
        request_headers = self.session.headers.copy()
        if headers:
            request_headers.update(headers)
        
        start_time = time.time()
        
        try:
            logger.debug(f"Making {method} request to {url}")
            
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data if method in ['POST', 'PUT', 'PATCH'] else None,
                headers=request_headers,
                timeout=timeout
            )
            
            response_time = time.time() - start_time
            
            # Handle HTTP errors
            if not response.ok:
                error_msg = f"HTTP {response.status_code}: {response.reason}"
                try:
                    error_detail = response.json().get('error', error_msg)
                except:
                    error_detail = response.text[:200] if response.text else error_msg
                
                raise APIError(
                    ErrorDetails(
                        category=ErrorCategory.API,
                        severity=ErrorSeverity.HIGH if response.status_code >= 500 else ErrorSeverity.MEDIUM,
                        message=f"API request failed: {error_detail}",
                        api_endpoint=url,
                        status_code=response.status_code
                    )
                )
            
            # Parse response data
            try:
                response_data = response.json()
            except ValueError:
                # Not JSON response
                response_data = {"raw_content": response.text}
            
            return APIResponse(
                success=True,
                data=response_data,
                status_code=response.status_code,
                headers=dict(response.headers),
                response_time=response_time,
                api_name=self.api_name,
                endpoint=endpoint
            )
            
        except requests.exceptions.Timeout:
            raise NetworkError(
                ErrorDetails(
                    category=ErrorCategory.NETWORK,
                    severity=ErrorSeverity.MEDIUM,
                    message=f"Request timeout after {timeout}s",
                    api_endpoint=url
                )
            )
            
        except requests.exceptions.ConnectionError as e:
            raise NetworkError(
                ErrorDetails(
                    category=ErrorCategory.NETWORK,
                    severity=ErrorSeverity.HIGH,
                    message=f"Connection error: {str(e)}",
                    api_endpoint=url
                )
            )
            
        except Exception as e:
            raise APIError(
                ErrorDetails(
                    category=ErrorCategory.API,
                    severity=ErrorSeverity.HIGH,
                    message=f"Unexpected error: {str(e)}",
                    api_endpoint=url
                )
            )
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None, **kwargs) -> APIResponse:
        """Make GET request"""
        return self._make_request("GET", endpoint, params=params, **kwargs)
    
    def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> APIResponse:
        """Make POST request"""
        return self._make_request("POST", endpoint, data=data, **kwargs)
    
    def put(self, endpoint: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> APIResponse:
        """Make PUT request"""
        return self._make_request("PUT", endpoint, data=data, **kwargs)
    
    def delete(self, endpoint: str, **kwargs) -> APIResponse:
        """Make DELETE request"""
        return self._make_request("DELETE", endpoint, **kwargs)
    
    @abstractmethod
    def search(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Search for documents/data using the API
        
        Args:
            query: Search query string
            **kwargs: Additional search parameters
            
        Returns:
            List of search results
        """
        pass
    
    @abstractmethod
    def get_document(self, document_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Retrieve specific document by ID
        
        Args:
            document_id: Unique document identifier
            **kwargs: Additional parameters
            
        Returns:
            Document data or None if not found
        """
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics"""
        return {
            "api_name": self.api_name,
            "request_count": self.request_count,
            "rate_limit": {
                "requests_per_second": self.rate_limit.requests_per_second,
                "current_requests": self.rate_limit.current_requests
            },
            "base_url": self.credential.base_url,
            "last_request_time": self.last_request_time
        }
    
    def close(self):
        """Close the session and clean up resources"""
        if hasattr(self, 'session'):
            self.session.close()
            logger.debug(f"Closed {self.api_name} API client session")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
    
    def batch_search(self, 
                    queries: List[str], 
                    batch_size: int = 10,
                    delay_between_batches: float = 1.0) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Perform batch searches with rate limiting
        
        Args:
            queries: List of search queries
            batch_size: Number of queries per batch
            delay_between_batches: Delay between batches in seconds
            
        Yields:
            List of results for each batch
        """
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            batch_results = []
            
            for query in batch:
                try:
                    results = self.search(query)
                    batch_results.extend(results)
                except Exception as e:
                    logger.error(f"Error in batch search for query '{query}': {e}")
                    continue
            
            yield batch_results
            
            # Delay between batches
            if i + batch_size < len(queries):
                time.sleep(delay_between_batches)
    
    def health_check(self) -> bool:
        """
        Perform health check on the API
        
        Returns:
            bool: True if API is healthy
        """
        try:
            # Use the auth manager's test function
            return self.auth_manager.test_credential(self.api_name)
        except Exception as e:
            logger.error(f"Health check failed for {self.api_name}: {e}")
            return False