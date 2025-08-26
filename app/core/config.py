import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Setup logger for configuration-related logging
logger = logging.getLogger(__name__)

class Config:
    """
    Configuration management class for the RAG (Retrieval-Augmented Generation) System.
    
    This class centralizes all configuration settings loaded from environment variables,
    providing a single source of truth for application settings. It handles loading,
    validation, and access to configuration values with appropriate defaults.
    
    The configuration includes:
    - Application settings (name, debug mode)
    - Vector database settings
    - Embedding model configuration
    - LLM (Language Model) provider settings
    - Document processing parameters
    
    Attributes:
        APP_NAME (str): Name of the application
        DEBUG (bool): Debug mode flag
        VECTOR_DB_PATH (str): Path to store vector database files
        EMBEDDING_MODEL (str): Name of the sentence embedding model
        LLM_PROVIDER (str): LLM service provider (google)
        LLM_MODEL (str): Specific model name for the LLM provider
        LLM_API_KEY (str): API key for accessing LLM services
        CHUNK_SIZE (int): Maximum size of text chunks for processing
        CHUNK_OVERLAP (int): Overlap size between consecutive chunks
    """
    
    def __init__(self):
        """
        Initialize the configuration by loading and validating all settings.
        
        This method performs the following steps:
        1. Load all configuration values from environment variables
        2. Validate critical configuration parameters
        3. Log the initialization status
        
        Raises:
            ValueError: If any critical configuration is invalid or missing
        """
        self._load_config()
        self._validate_config()
        logger.info("Configuration initialized successfully")
    
    def _load_config(self) -> None:
        """
        Load all configuration values from environment variables with defaults.
        
        This method reads environment variables and sets configuration attributes.
        If a variable is not set, it uses a sensible default value.
        
        Configuration categories loaded:
        - Application settings
        - Vector database settings
        - Embedding model settings
        - LLM provider settings
        - Text processing parameters
        
        Note:
            All string values are stripped of leading/trailing whitespace.
        """
        # Application Configuration
        self.APP_NAME: str = os.getenv("APP_NAME", "RAG System").strip()
        self.DEBUG: bool = os.getenv("DEBUG", "False").lower().strip() == "true"
        
        # Vector Database Configuration
        self.VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "vector_db").strip()
        
        # Embedding Model Configuration
        self.EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2").strip()
        
        # LLM Configuration
        self.LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "google").lower().strip()
        self.LLM_MODEL: str = os.getenv("LLM_MODEL", "gemma-3-12b-it").strip()
        self.LLM_API_KEY: str = os.getenv("LLM_API_KEY", "").strip()
        
        # Document Processing Configuration
        self.CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
        self.CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    def _validate_config(self) -> None:
        """
        Validate all critical configuration parameters.
        
        This method performs validation checks to ensure that:
        - LLM provider is supported
        - Chunking parameters are logical and within bounds
        - Required API keys are provided
        
        Raises:
            ValueError: If any validation check fails, containing a descriptive error message
        """
        errors = []
        
        # Validate LLM Provider
        if self.LLM_PROVIDER not in ["google"]:
            errors.append(f"Unsupported LLM provider: '{self.LLM_PROVIDER}'. Must be 'google'")
        
        # Validate Chunking Parameters
        if self.CHUNK_SIZE <= 0:
            errors.append("CHUNK_SIZE must be a positive integer")
        
        if self.CHUNK_OVERLAP < 0:
            errors.append("CHUNK_OVERLAP must be a non-negative integer")
        
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            errors.append("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        
        # Validate API Key
        if not self.LLM_API_KEY:
            errors.append("LLM_API_KEY is required but not provided")
        
        # Report validation errors
        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(f"  • {error}" for error in errors)
            logger.error(error_message)
            raise ValueError(error_message)
        
        logger.debug("All configuration parameters validated successfully")
    
# Create a global configuration instance for application-wide access
config = Config()