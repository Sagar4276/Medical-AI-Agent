"""
Error Management System for Glaucoma Corpus Builder
Comprehensive error handling with retry mechanisms and logging
"""

import logging
import time
import functools
from typing import Any, Callable, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    API = "api"
    AUTHENTICATION = "authentication"
    DATA_PROCESSING = "data_processing"
    NETWORK = "network"
    VALIDATION = "validation"
    SYSTEM = "system"


@dataclass
class ErrorDetails:
    """Detailed error information for tracking and debugging"""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    api_endpoint: Optional[str] = None
    status_code: Optional[int] = None
    retry_count: int = 0
    timestamp: Optional[float] = None
    additional_context: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class CorpusBuilderError(Exception):
    """Base exception class for corpus builder errors"""
    
    def __init__(self, error_details: ErrorDetails):
        self.error_details = error_details
        super().__init__(error_details.message)
        
    def __str__(self):
        return f"[{self.error_details.category.value.upper()}] {self.error_details.message}"


class APIError(CorpusBuilderError):
    """API-specific error with detailed context"""
    pass


class AuthenticationError(CorpusBuilderError):
    """Authentication-related errors"""
    pass


class DataProcessingError(CorpusBuilderError):
    """Data processing and validation errors"""
    pass


class NetworkError(CorpusBuilderError):
    """Network connectivity errors"""
    pass


class ErrorManager:
    """
    Comprehensive error management system with:
    - Automatic retry mechanisms
    - Rate limiting awareness
    - Error categorization and severity assessment
    - Detailed logging and monitoring
    """
    
    def __init__(self, 
                 max_retries: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_multiplier: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_multiplier = backoff_multiplier
        self.error_history: List[ErrorDetails] = []
        
    def calculate_delay(self, retry_count: int) -> float:
        """Calculate exponential backoff delay"""
        delay = self.base_delay * (self.backoff_multiplier ** retry_count)
        return min(delay, self.max_delay)
    
    def is_retryable_error(self, error: Exception, status_code: Optional[int] = None) -> bool:
        """Determine if an error is retryable"""
        # Network errors are generally retryable
        if isinstance(error, (ConnectionError, TimeoutError)):
            return True
            
        # HTTP status codes that are retryable
        if status_code:
            retryable_codes = {429, 500, 502, 503, 504, 408}
            return status_code in retryable_codes
            
        # Authentication errors are generally not retryable
        if isinstance(error, AuthenticationError):
            return False
            
        return True
    
    def categorize_error(self, error: Exception, api_endpoint: Optional[str] = None, 
                        status_code: Optional[int] = None) -> ErrorDetails:
        """Categorize and assess error severity"""
        
        # Determine category
        if isinstance(error, AuthenticationError):
            category = ErrorCategory.AUTHENTICATION
            severity = ErrorSeverity.HIGH
        elif isinstance(error, APIError):
            category = ErrorCategory.API
            severity = ErrorSeverity.MEDIUM if status_code and 400 <= status_code < 500 else ErrorSeverity.HIGH
        elif isinstance(error, DataProcessingError):
            category = ErrorCategory.DATA_PROCESSING
            severity = ErrorSeverity.LOW
        elif isinstance(error, (ConnectionError, TimeoutError)):
            category = ErrorCategory.NETWORK
            severity = ErrorSeverity.MEDIUM
        else:
            category = ErrorCategory.SYSTEM
            severity = ErrorSeverity.HIGH
            
        # Adjust severity based on status code
        if status_code:
            if status_code == 429:  # Rate limit
                severity = ErrorSeverity.LOW
            elif status_code == 401:  # Unauthorized
                severity = ErrorSeverity.HIGH
            elif status_code >= 500:  # Server errors
                severity = ErrorSeverity.HIGH
                
        return ErrorDetails(
            category=category,
            severity=severity,
            message=str(error),
            api_endpoint=api_endpoint,
            status_code=status_code,
            additional_context={"error_type": type(error).__name__}
        )
    
    def log_error(self, error_details: ErrorDetails):
        """Log error with appropriate severity level"""
        self.error_history.append(error_details)
        
        context = f"API: {error_details.api_endpoint}" if error_details.api_endpoint else ""
        status = f" (Status: {error_details.status_code})" if error_details.status_code else ""
        retry_info = f" (Retry: {error_details.retry_count})" if error_details.retry_count > 0 else ""
        
        log_message = f"{error_details.message} {context}{status}{retry_info}"
        
        if error_details.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message)
        elif error_details.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_details.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def handle_api_error(self, 
                        error: Exception, 
                        api_endpoint: str, 
                        status_code: Optional[int] = None) -> ErrorDetails:
        """Handle API-specific errors with context"""
        error_details = self.categorize_error(error, api_endpoint, status_code)
        
        # Convert to appropriate exception type
        if error_details.category == ErrorCategory.AUTHENTICATION:
            raise AuthenticationError(error_details)
        elif error_details.category == ErrorCategory.API:
            raise APIError(error_details)
        elif error_details.category == ErrorCategory.NETWORK:
            raise NetworkError(error_details)
        else:
            raise CorpusBuilderError(error_details)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors encountered"""
        if not self.error_history:
            return {"total_errors": 0, "by_category": {}, "by_severity": {}}
            
        by_category = {}
        by_severity = {}
        
        for error_detail in self.error_history:
            # Count by category
            category = error_detail.category.value
            by_category[category] = by_category.get(category, 0) + 1
            
            # Count by severity
            severity = error_detail.severity.value
            by_severity[severity] = by_severity.get(severity, 0) + 1
        
        return {
            "total_errors": len(self.error_history),
            "by_category": by_category,
            "by_severity": by_severity,
            "most_recent": self.error_history[-1] if self.error_history else None
        }


def with_retry(max_retries: int = 3, 
               base_delay: float = 1.0,
               exceptions: tuple = (Exception,)):
    """
    Decorator for automatic retry with exponential backoff
    
    Args:
        max_retries: Maximum number of retries
        base_delay: Initial delay between retries
        exceptions: Tuple of exceptions to catch and retry
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            error_manager = ErrorManager(max_retries=max_retries, base_delay=base_delay)
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_retries:
                        error_details = error_manager.categorize_error(e)
                        error_details.retry_count = attempt
                        error_manager.log_error(error_details)
                        raise e
                    
                    if not error_manager.is_retryable_error(e):
                        error_details = error_manager.categorize_error(e)
                        error_manager.log_error(error_details)
                        raise e
                    
                    delay = error_manager.calculate_delay(attempt)
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
            
        return wrapper
    return decorator


# Convenience decorators for common retry patterns
def api_retry(func):
    return with_retry(max_retries=3, base_delay=1.0, 
                     exceptions=(APIError, NetworkError, ConnectionError))(func)

def network_retry(func):
    return with_retry(max_retries=5, base_delay=0.5,
                     exceptions=(ConnectionError, TimeoutError))(func)