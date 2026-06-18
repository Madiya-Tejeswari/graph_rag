# PG&E GraphRAG Backend

Production-ready hybrid multimodal GraphRAG system for PG&E Greenbook and Tariff documents.

## Architecture

This backend implements a production-grade GraphRAG system with:

- **GraphRAG as primary retrieval mechanism** for text, tables, and figure facts
- **Image similarity search** for visual questions
- **Neo4j graph database** for persistent knowledge storage
- **Model-agnostic LLM** support (OpenAI, Anthropic, Groq, Google, Ollama, AWS Bedrock)
- **Docling-powered PDF parsing** for multimodal content extraction
- **Sentence transformers** for image embedding and similarity search

## System Design

### Ingestion Pipeline (Offline, Run Once)

```
PDF → Docling → Text/Tables/Figures/Images → Knowledge Graph → Neo4j
                                              ↓
                                         Image Index (FAISS)
                                              ↓
                                         Persistent Storage
```

### Query Pipeline (Runtime)

```
User Query
    ↓
    ├→ Graph Search (Neo4j/Cypher)
    ├→ Image Search (Similarity)
    ↓
Context Builder
    ↓
Prompt Builder
    ↓
LLM Generation
    ↓
Response (with citations)
```

## Installation

### Prerequisites

- Python 3.9+
- Neo4j 5.0+ (running instance)
- (Optional) Ollama for local LLM models

### Setup

1. **Clone and install dependencies:**

```bash
cd backend
pip install -r requirements.txt
```

2. **Environment configuration:**

Create `.env` file:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=False

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
NEO4J_DATABASE=pge_rag

# Storage
STORAGE_DIR=./storage
PDF_DIR=./storage/pdfs

# LLM (choose one or more)
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GROQ_API_KEY=your_key
OLLAMA_BASE_URL=http://localhost:11434

# Retrieval
GRAPH_TOP_K=10
IMAGE_TOP_K=5
IMAGE_SIMILARITY_THRESHOLD=0.7
EXPANSION_HOPS=3

# Features
ENABLE_IMAGE_CAPTIONING=true
ENABLE_ENTITY_EXTRACTION=true
ENABLE_TABLE_EXTRACTION=true
```

3. **Prepare PDFs:**

Place your PDFs in `./storage/pdfs/`:
- `greenbook.pdf`
- `tariffs.pdf`

4. **Start services:**

```bash
# Start Neo4j (if using Docker)
docker run --name neo4j -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest

# Start Ollama (optional, for local models)
ollama serve
```

## Running the System

### 1. Offline Ingestion (One-time)

```bash
python run_ingestion.py
```

This will:
- Parse all PDFs using Docling
- Extract text, tables, figures, and images
- Build Neo4j knowledge graph
- Generate image captions
- Create image embeddings
- Persist everything to disk

Monitor progress in logs. This is a one-time operation.

### 2. Start API Server

```bash
python -m api.main
```

Server runs at `http://localhost:8000`

**API Documentation:** `http://localhost:8000/docs`

## API Endpoints

### Health & Status

```http
GET /health
GET /graph/status
```

### Ingestion

```http
POST /ingest
```

Triggers background ingestion. Returns 202 Accepted.

### RAG Query

```http
POST /rag
Content-Type: application/json

{
  "query": "What is the maximum transformer size for single phase service?",
  "model": "gpt-4",
  "rag_approach": "graph_rag"
}
```

**Response:**

```json
{
  "status": "success",
  "query": "What is the maximum transformer size for single phase service?",
  "answer": "The maximum transformer size for single phase service is 100 kVA, as specified in Table 4-1.",
  "sources": [
    {
      "title": "p.52",
      "url": "page://52"
    }
  ],
  "metadata": {
    "retrieval_time_ms": 245,
    "generation_time_ms": 1523,
    "total_time_ms": 1768,
    "model_used": "gpt-4",
    "retrieval_method": "graph_rag",
    "input_tokens": 1200,
    "output_tokens": 450,
    "total_tokens": 1650,
    "generated_at": "2024-01-15T10:30:00"
  }
}
```

## Architecture Details

### Neo4j Graph Schema

**Node Types:**
- `Document` - Source PDF document
- `Page` - Document page
- `Entity` - Extracted entities (equipment, services, etc.)
- `Table` - Table node
- `Figure` - Figure node
- `Component` - Figure components
- `Fact` - Extracted facts

**Relationships:**
- `HAS_PAGE` - Document to page
- `HAS_TABLE` - Document to table
- `HAS_FIGURE` - Document to figure
- `MENTIONS` - References to entities
- `CONTAINS` - Container relationships
- `CONNECTED_TO` - Equipment connections
- `PART_OF` - Component relationships

**Provenance on every node:**
- `page_number` - PDF page number
- `document_name` - Source document
- `source_type` - Type (table, figure, text, entity)
- `source_name` - Original identifier

### Image Storage

Images stored locally with structure:
```
storage/
├── images/           # Image files (.png)
├── image_metadata/   # Metadata JSON files
├── image_embeddings/ # Embedding vectors (.npy)
└── image_index.pkl   # FAISS search index
```

### Retrieval Pipeline

1. **Graph Search:**
   - Entity extraction from query
   - Neo4j entity matching
   - Relationship expansion (1-3 hops)
   - Evidence collection with provenance

2. **Image Search:**
   - Query embedding via sentence-transformers
   - Cosine similarity against image embeddings
   - Threshold filtering (default 0.7)
   - Top-K results

3. **Context Building:**
   - Merge graph and image results
   - Organize by source type
   - Truncate to token limit
   - Deduplicate results

### Generation

- GraphRAG-style prompt with evidence
- Model-agnostic LLM call
- Token counting and metrics
- Source citation extraction

## Configuration

### Tuning Parameters

**Retrieval:**
```python
GRAPH_TOP_K=10           # Number of graph results
IMAGE_TOP_K=5            # Number of image results
IMAGE_SIMILARITY_THRESHOLD=0.7  # Minimum similarity score
EXPANSION_HOPS=3         # Relationship expansion depth
```

**Generation:**
```python
MAX_CHUNK_SIZE=512       # Text chunk size
CHUNK_OVERLAP=100        # Chunk overlap
ENABLE_IMAGE_CAPTIONING=true  # Caption generation
ENABLE_ENTITY_EXTRACTION=true  # Entity extraction
ENABLE_TABLE_EXTRACTION=true   # Table extraction
```

**Cache:**
```python
ENABLE_CACHING=true
CACHE_TTL_SECONDS=3600   # 1 hour cache TTL
```

## Model Selection

The system is model-agnostic. Frontend specifies model in request:

```python
# OpenAI
"model": "gpt-4"
"model": "gpt-3.5-turbo"

# Anthropic
"model": "claude-3-sonnet-20240229"
"model": "claude-3-haiku-20240307"

# Groq
"model": "groq/mixtral-8x7b"

# Google
"model": "gemini-pro"

# Ollama (local)
"model": "ollama/mistral"
"model": "ollama/neural-chat"

# AWS Bedrock
"model": "bedrock/anthropic.claude-3-sonnet"
"model": "bedrock/meta.llama2-13b"
```

Set corresponding API keys in `.env`.

## Logging

Logs written to `./logs/graphrag.log`

Configuration in `config.py`:
```python
LOG_LEVEL = "INFO"
LOG_DIR = "./logs"
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

```bash
# Format
black backend/

# Lint
flake8 backend/

# Type check
mypy backend/
```

## Performance Considerations

- **Ingestion:** ~5-30 minutes per 100-page PDF (depends on content complexity)
- **Query:** ~0.5-2 seconds (excluding LLM generation)
- **Graph queries:** <100ms for typical queries
- **Image search:** <50ms with cached embeddings

## Troubleshooting

### Neo4j Connection Issues

```bash
# Check Neo4j is running
docker ps | grep neo4j

# Verify credentials in .env
# Default: neo4j / password
```

### Out of Memory

Adjust batch sizes:
```python
BATCH_SIZE=16  # Reduce from 32
MAX_WORKERS=2  # Reduce parallelism
```

### Slow Queries

Enable graph indexing:
```bash
python -c "from storage.neo4j_client import Neo4jClient; Neo4jClient(...).create_indexes()"
```

### Image Caption Generation Fails

Ensure Ollama is running:
```bash
ollama serve
ollama pull llava
```

Or disable image captioning:
```env
ENABLE_IMAGE_CAPTIONING=false
```

## Production Deployment

### Docker

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY backend/ .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["python", "-m", "api.main"]
```

Build and run:
```bash
docker build -t pge-graphrag .
docker run -p 8000:8000 -e NEO4J_URI=bolt://neo4j:7687 pge-graphrag
```

### Environment Variables

Set in container or .env:
- `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
- API keys for chosen LLM providers
- `LOG_LEVEL=INFO` for production
- `DEBUG=False`

### Monitoring

- API health: `/health`
- Graph status: `/graph/status`
- Logs: `./logs/graphrag.log`
- Neo4j metrics: Neo4j browser at http://localhost:7474

## API Compatibility

**Frontend Contract Maintained:**

Request schema:
```python
class RAGRequest(BaseModel):
    query: str
    model: str
    rag_approach: str
```

Response schema:
```python
class RAGResponse(BaseModel):
    status: str
    query: str
    answer: str
    sources: List[Source]
    metadata: Metadata
```

No breaking changes to existing frontend.

## License

Proprietary - PG&E

## Support

For issues or questions, contact the development team.
