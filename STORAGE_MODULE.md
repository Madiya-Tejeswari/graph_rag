# Storage Module Documentation

## Overview
The storage module provides two main components for the PG&E GraphRAG backend:
1. **Neo4jClient** - Graph database operations
2. **ImageStore** - Image storage and management

---

## Neo4jClient

### Purpose
Manages all Neo4j graph database operations for storing and querying the knowledge graph.

### Key Features
- Connection management with error handling
- Node creation and deletion
- Relationship creation between nodes
- Cypher query execution
- Index creation for performance optimization
- Graph statistics and analytics
- Safe database clearing with confirmation

### Main Methods

#### Connection Methods
```python
# Connect to database
client.connect() -> bool

# Close connection
client.close() -> None
```

#### Node Operations
```python
# Create a node
node_id = client.create_node(label: str, properties: Dict) -> int

# Get a node
node = client.get_node(node_id: int) -> Dict

# Delete a node
client.delete_node(node_id: int) -> bool
```

#### Relationship Operations
```python
# Create relationship between nodes
client.create_relationship(source_id, target_id, relationship_type, properties) -> bool

# Find relationships of specific type
rels = client.find_by_relationship(relationship_type: str) -> List[Dict]
```

#### Search Operations
```python
# Search for entities by keyword
entities = client.search_entities(keyword: str, top_k: int) -> List[Dict]

# Execute custom Cypher query
results = client.execute_query(cypher_query: str, parameters: Dict) -> List[Dict]
```

#### Database Management
```python
# Create performance indexes
client.create_indexes() -> bool

# Get database statistics
stats = client.get_graph_stats() -> Dict

# Clear all data (requires confirm=True)
client.clear_database(confirm: bool) -> bool
```

### Usage Example
```python
from storage.neo4j_client import Neo4jClient

# Initialize client
client = Neo4jClient(
    uri="bolt://localhost:7687",
    username="neo4j",
    password="password",
    database="pge_rag"
)

# Connect
client.connect()

# Create a document node
doc_id = client.create_node("Document", {"name": "greenbook.pdf", "type": "document"})

# Create an entity node
entity_id = client.create_node("Entity", {"name": "transformer", "type": "Equipment"})

# Create relationship
client.create_relationship(doc_id, entity_id, "MENTIONS")

# Search
results = client.search_entities("transformer", top_k=10)

# Cleanup
client.close()
```

---

## ImageStore

### Purpose
Manages image storage, metadata, and embeddings for the visual retrieval system.

### Key Features
- Image persistence with PNG format
- JSON metadata storage
- Numpy embedding vector storage
- Image search by page and document
- Embedding management
- Image validation
- Statistics and analytics
- Safe storage clearing with confirmation

### Main Methods

#### Image Operations
```python
# Save image with metadata
store.save_image(image_id: str, image_data: bytes, metadata: Dict) -> bool

# Get image binary data
image_bytes = store.get_image(image_id: str) -> Optional[bytes]

# Delete image and its files
store.delete_image(image_id: str) -> bool

# Validate image integrity
store.validate_image(image_id: str) -> bool
```

#### Metadata Operations
```python
# Get metadata for an image
metadata = store.get_metadata(image_id: str) -> Dict

# Get image with metadata
full_data = store.get_image_with_metadata(image_id: str) -> Dict

# Update metadata
store.update_metadata(image_id: str, updates: Dict) -> bool
```

#### Embedding Operations
```python
# Save embedding vector
store.save_embedding(image_id: str, embedding: np.ndarray) -> bool

# Get embedding vector
embedding = store.get_embedding(image_id: str) -> Optional[np.ndarray]
```

#### Search Operations
```python
# Get all image IDs
ids = store.get_all_image_ids() -> List[str]

# Search by page number
page_images = store.search_by_page(page_number: int) -> List[str]

# Search by document
doc_images = store.search_by_document(document_name: str) -> List[str]
```

#### Storage Management
```python
# Get total image count
count = store.get_image_count() -> int

# Get detailed statistics
stats = store.get_image_stats() -> Dict

# Clear all storage (requires confirm=True)
store.clear_storage(confirm: bool) -> bool
```

### Usage Example
```python
import numpy as np
from storage.image_store import ImageStore
from pathlib import Path

# Initialize store
store = ImageStore(
    image_dir=Path("storage/images"),
    metadata_dir=Path("storage/image_metadata"),
    embeddings_dir=Path("storage/image_embeddings")
)

# Save image
with open("diagram.png", "rb") as f:
    image_data = f.read()

metadata = {
    "page_number": 5,
    "document_name": "greenbook.pdf",
    "caption": "Transformer installation diagram",
    "source_type": "image"
}

store.save_image("img_001", image_data, metadata)

# Save embedding
embedding = np.random.randn(384)  # Example embedding
store.save_embedding("img_001", embedding)

# Retrieve
retrieved = store.get_image_with_metadata("img_001")

# Search
images_on_page_5 = store.search_by_page(5)
images_in_greenbook = store.search_by_document("greenbook.pdf")

# Stats
stats = store.get_image_stats()
print(f"Total images: {stats['total_images']}")
```

---

## Integration with the System

### In IngestionPipeline
```python
from storage.neo4j_client import Neo4jClient
from storage.image_store import ImageStore
from ingestion.pipeline import IngestionPipeline

# Both clients are passed to the pipeline
pipeline = IngestionPipeline(neo4j_client, image_store)
pipeline.run(pdf_paths=[...])
```

### In GraphBuilder
The Neo4jClient is used to build the knowledge graph from extracted PDF content.

### In ImageProcessor
The ImageStore is used to persist processed images and their metadata/embeddings.

### In Retrieval Engines
Both clients are used by:
- **GraphSearchEngine** - Queries the Neo4j database for facts and entities
- **ImageSearchEngine** - Uses embeddings for visual similarity search

---

## Error Handling

Both components include comprehensive error handling:
- Connection failures are logged and handled gracefully
- Missing files/data return None or empty collections
- Database operations are wrapped in try-catch blocks
- All methods include detailed logging for debugging

---

## Performance Considerations

### Neo4j Optimization
- Indexes are automatically created on commonly searched properties
- Queries use proper Cypher patterns for efficiency
- Node IDs are used for fast relationship lookups

### Image Storage Optimization
- Images stored as PNG (compressed format)
- Metadata stored as JSON (human-readable and efficient)
- Embeddings stored as numpy binary (fast loading)
- Separate directories for different file types

---

## Safety Features

### Database Protection
- `clear_database()` requires `confirm=True` parameter
- No automatic data deletion
- All deletion operations are logged

### Storage Protection  
- `clear_storage()` requires `confirm=True` parameter
- Image validation before operations
- Metadata consistency checks

---

## Configuration

Both components read from `config.py`:

```python
# Neo4j settings
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
NEO4J_DATABASE = "pge_rag"

# Storage paths
IMAGE_DIR = Path("storage/images")
IMAGE_METADATA_DIR = Path("storage/image_metadata")
IMAGE_EMBEDDINGS_DIR = Path("storage/image_embeddings")
```

---

## Testing

### Neo4jClient Tests
```python
# Unit tests should verify:
- Connection/disconnection
- Node creation/retrieval/deletion
- Relationship creation
- Query execution
- Index creation
- Statistics collection
```

### ImageStore Tests
```python
# Unit tests should verify:
- Image save/retrieve
- Metadata save/retrieve
- Embedding save/retrieve
- Search operations
- Image validation
- Storage statistics
```

---

## Logging

Both components use Python's standard logging module:
- Set `LOG_LEVEL` in config.py to control verbosity
- All operations are logged at appropriate levels
- Errors are logged with exception info for debugging
