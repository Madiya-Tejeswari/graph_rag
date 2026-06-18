"""
Configuration management for PG&E GraphRAG backend.
Uses environment variables with sensible defaults.
"""

import os
from typing import List
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Get the root directory
ROOT_DIR = Path(__file__).parent

# ============================================================================
# API Configuration
# ============================================================================
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://localhost:5000",
    os.getenv("FRONTEND_URL", "http://localhost:3000"),
]

# ============================================================================
# Neo4j Configuration
# ============================================================================
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# ============================================================================
# Storage Configuration
# ============================================================================
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", ROOT_DIR / "storage"))
PDF_DIR = Path(os.getenv("PDF_DIR", STORAGE_DIR / "pdfs"))
IMAGE_DIR = Path(os.getenv("IMAGE_DIR", STORAGE_DIR / "images"))
IMAGE_METADATA_DIR = Path(os.getenv("IMAGE_METADATA_DIR", STORAGE_DIR / "image_metadata"))
IMAGE_EMBEDDINGS_DIR = Path(os.getenv("IMAGE_EMBEDDINGS_DIR", STORAGE_DIR / "image_embeddings"))

# Create directories if they don't exist
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_METADATA_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
PDF_DIR.mkdir(parents=True, exist_ok=True)

# PDF paths for ingestion
PDF_PATHS = [
    str(PDF_DIR / "greenbook.pdf"),
    str(PDF_DIR / "tariffs.pdf"),
]

# ============================================================================
# Retrieval Configuration
# ============================================================================
GRAPH_TOP_K = int(os.getenv("GRAPH_TOP_K", "10"))
IMAGE_TOP_K = int(os.getenv("IMAGE_TOP_K", "5"))
IMAGE_SIMILARITY_THRESHOLD = float(os.getenv("IMAGE_SIMILARITY_THRESHOLD", "0.7"))
EXPANSION_HOPS = int(os.getenv("EXPANSION_HOPS", "3"))

# ============================================================================
# LLM Configuration
# ============================================================================
# Do NOT hardcode any model provider
# The frontend specifies which model to use
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4")

# API Keys for various providers (optional)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Bedrock configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# ============================================================================
# Embedding Configuration
# ============================================================================
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# Image index configuration
IMAGE_INDEX_TYPE = os.getenv("IMAGE_INDEX_TYPE", "faiss")  # faiss or hnswlib
IMAGE_INDEX_PATH = Path(os.getenv("IMAGE_INDEX_PATH", STORAGE_DIR / "image_index.pkl"))

# ============================================================================
# Ingestion Configuration
# ============================================================================
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Image processing
ENABLE_IMAGE_CAPTIONING = os.getenv("ENABLE_IMAGE_CAPTIONING", "true").lower() == "true"
IMAGE_CAPTION_MODEL = os.getenv("IMAGE_CAPTION_MODEL", "llava")  # Using local model

# Entity extraction
ENABLE_ENTITY_EXTRACTION = os.getenv("ENABLE_ENTITY_EXTRACTION", "true").lower() == "true"

# Table extraction
ENABLE_TABLE_EXTRACTION = os.getenv("ENABLE_TABLE_EXTRACTION", "true").lower() == "true"

# ============================================================================
# Logging Configuration
# ============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_DIR = Path(os.getenv("LOG_DIR", ROOT_DIR / "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Feature Flags
# ============================================================================
ENABLE_GRAPH_EXPANSION = os.getenv("ENABLE_GRAPH_EXPANSION", "true").lower() == "true"
ENABLE_IMAGE_RETRIEVAL = os.getenv("ENABLE_IMAGE_RETRIEVAL", "true").lower() == "true"
ENABLE_CACHING = os.getenv("ENABLE_CACHING", "true").lower() == "true"

# ============================================================================
# Cache Configuration
# ============================================================================
CACHE_DIR = Path(os.getenv("CACHE_DIR", STORAGE_DIR / "cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1 hour

# ============================================================================
# Batch Processing
# ============================================================================
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))


# ============================================================================
# Summary
# ============================================================================
class Settings:
    """Settings container for easy access."""
    
    # API
    api_host = API_HOST
    api_port = API_PORT
    debug = DEBUG
    cors_origins = CORS_ORIGINS
    
    # Neo4j
    neo4j_uri = NEO4J_URI
    neo4j_username = NEO4J_USERNAME
    neo4j_password = NEO4J_PASSWORD
    neo4j_database = NEO4J_DATABASE
    
    # Storage
    storage_dir = STORAGE_DIR
    pdf_dir = PDF_DIR
    pdf_paths = PDF_PATHS
    image_dir = IMAGE_DIR
    image_metadata_dir = IMAGE_METADATA_DIR
    image_embeddings_dir = IMAGE_EMBEDDINGS_DIR
    
    # Retrieval
    graph_top_k = GRAPH_TOP_K
    image_top_k = IMAGE_TOP_K
    image_similarity_threshold = IMAGE_SIMILARITY_THRESHOLD
    expansion_hops = EXPANSION_HOPS
    
    # LLM
    default_model = DEFAULT_MODEL
    openai_api_key = OPENAI_API_KEY
    anthropic_api_key = ANTHROPIC_API_KEY
    groq_api_key = GROQ_API_KEY
    google_api_key = GOOGLE_API_KEY
    ollama_base_url = OLLAMA_BASE_URL
    aws_region = AWS_REGION
    
    # Embeddings
    embedding_model = EMBEDDING_MODEL
    embedding_dimension = EMBEDDING_DIMENSION
    image_index_type = IMAGE_INDEX_TYPE
    image_index_path = IMAGE_INDEX_PATH
    
    # Ingestion
    max_chunk_size = MAX_CHUNK_SIZE
    chunk_overlap = CHUNK_OVERLAP
    enable_image_captioning = ENABLE_IMAGE_CAPTIONING
    image_caption_model = IMAGE_CAPTION_MODEL
    enable_entity_extraction = ENABLE_ENTITY_EXTRACTION
    enable_table_extraction = ENABLE_TABLE_EXTRACTION
    
    # Logging
    log_level = LOG_LEVEL
    log_dir = LOG_DIR
    
    # Features
    enable_graph_expansion = ENABLE_GRAPH_EXPANSION
    enable_image_retrieval = ENABLE_IMAGE_RETRIEVAL
    enable_caching = ENABLE_CACHING
    
    # Cache
    cache_dir = CACHE_DIR
    cache_ttl_seconds = CACHE_TTL_SECONDS
    
    # Batch
    batch_size = BATCH_SIZE
    max_workers = MAX_WORKERS

    # Uppercase aliases (for code that references settings.UPPERCASE_NAME)
    CORS_ORIGINS = CORS_ORIGINS
    API_HOST = API_HOST
    API_PORT = API_PORT
    DEBUG = DEBUG
    NEO4J_URI = NEO4J_URI
    NEO4J_USERNAME = NEO4J_USERNAME
    NEO4J_PASSWORD = NEO4J_PASSWORD
    NEO4J_DATABASE = NEO4J_DATABASE
    STORAGE_DIR = STORAGE_DIR
    PDF_DIR = PDF_DIR
    PDF_PATHS = PDF_PATHS
    IMAGE_DIR = IMAGE_DIR
    IMAGE_METADATA_DIR = IMAGE_METADATA_DIR
    IMAGE_EMBEDDINGS_DIR = IMAGE_EMBEDDINGS_DIR
    GRAPH_TOP_K = GRAPH_TOP_K
    IMAGE_TOP_K = IMAGE_TOP_K
    IMAGE_SIMILARITY_THRESHOLD = IMAGE_SIMILARITY_THRESHOLD
    EXPANSION_HOPS = EXPANSION_HOPS
    DEFAULT_MODEL = DEFAULT_MODEL
    OPENAI_API_KEY = OPENAI_API_KEY
    ANTHROPIC_API_KEY = ANTHROPIC_API_KEY
    GROQ_API_KEY = GROQ_API_KEY
    GOOGLE_API_KEY = GOOGLE_API_KEY
    OLLAMA_BASE_URL = OLLAMA_BASE_URL
    AWS_REGION = AWS_REGION
    EMBEDDING_MODEL = EMBEDDING_MODEL
    EMBEDDING_DIMENSION = EMBEDDING_DIMENSION
    IMAGE_INDEX_TYPE = IMAGE_INDEX_TYPE
    IMAGE_INDEX_PATH = IMAGE_INDEX_PATH
    MAX_CHUNK_SIZE = MAX_CHUNK_SIZE
    CHUNK_OVERLAP = CHUNK_OVERLAP
    ENABLE_IMAGE_CAPTIONING = ENABLE_IMAGE_CAPTIONING
    IMAGE_CAPTION_MODEL = IMAGE_CAPTION_MODEL
    ENABLE_ENTITY_EXTRACTION = ENABLE_ENTITY_EXTRACTION
    ENABLE_TABLE_EXTRACTION = ENABLE_TABLE_EXTRACTION
    LOG_LEVEL = LOG_LEVEL
    LOG_DIR = LOG_DIR
    ENABLE_GRAPH_EXPANSION = ENABLE_GRAPH_EXPANSION
    ENABLE_IMAGE_RETRIEVAL = ENABLE_IMAGE_RETRIEVAL
    ENABLE_CACHING = ENABLE_CACHING
    CACHE_DIR = CACHE_DIR
    CACHE_TTL_SECONDS = CACHE_TTL_SECONDS
    BATCH_SIZE = BATCH_SIZE
    MAX_WORKERS = MAX_WORKERS


# Create settings instance
settings = Settings()
