# Quick Start Guide - PG&E GraphRAG Backend

## Option 1: Local Development Setup

### 1. Prerequisites

```bash
# Python 3.9+
python --version

# Install Neo4j (using Docker)
docker run --name neo4j -p 7687:7687 -p 7474:7474 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

### 2. Install Backend

```bash
cd backend
pip install -r requirements.txt
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys (OpenAI, Anthropic, etc.)
```

### 4. Add Your PDFs

```bash
mkdir -p storage/pdfs
# Copy greenbook.pdf and tariffs.pdf to storage/pdfs/
```

### 5. Run Ingestion

```bash
python run_ingestion.py
```

Monitor the output. This creates the knowledge graph and image indexes.

### 6. Start API Server

```bash
python -m api.main
```

Server starts at `http://localhost:8000`

### 7. Test the API

```bash
# Health check
curl http://localhost:8000/health

# Graph status
curl http://localhost:8000/graph/status

# API docs
open http://localhost:8000/docs

# Query example
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the maximum transformer size for single phase service?",
    "model": "gpt-4"
  }'
```

---

## Option 2: Docker Compose (Recommended)

### 1. Prerequisites

```bash
docker --version
docker-compose --version
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 3. Add PDFs

```bash
mkdir -p storage/pdfs
# Copy greenbook.pdf and tariffs.pdf to storage/pdfs/
```

### 4. Start Services

```bash
# Start Neo4j and API
docker-compose up -d

# Monitor logs
docker-compose logs -f api

# Wait for API to be healthy
docker-compose ps
```

### 5. Run Ingestion

```bash
# Access API container shell
docker-compose exec api bash

# Run ingestion
python run_ingestion.py

# Exit
exit
```

### 6. Test the API

```bash
curl http://localhost:8000/health
curl http://localhost:8000/docs
```

---

## Option 3: Docker with Ollama (Local Models)

For running without cloud LLM providers:

```bash
# Start all services including Ollama
docker-compose --profile with-ollama up -d

# Pull LLM models
docker-compose exec ollama ollama pull mistral
docker-compose exec ollama ollama pull llava

# In .env, set:
ENABLE_IMAGE_CAPTIONING=true
# No API keys needed
```

---

## Verification Checklist

```bash
# ✓ API is running
curl http://localhost:8000/health
# Response: {"status": "healthy", ...}

# ✓ Graph database connected
curl http://localhost:8000/graph/status
# Response: {"status": "ready", "documents": 1, ...}

# ✓ Neo4j browser (optional)
open http://localhost:7474
# Login: neo4j / password

# ✓ API documentation
open http://localhost:8000/docs
```

---

## Common Issues

### Neo4j Connection Failed

```bash
# Check if Neo4j is running
docker ps | grep neo4j

# Verify credentials in .env match docker-compose or your Neo4j instance
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
NEO4J_URI=bolt://localhost:7687

# Restart Neo4j if needed
docker restart neo4j
```

### Import Errors

```bash
# Verify all dependencies installed
pip install -r requirements.txt

# Check Python version
python --version  # Should be 3.9+
```

### PDFs Not Found

```bash
# Create directory
mkdir -p storage/pdfs

# Copy PDFs
cp /path/to/greenbook.pdf storage/pdfs/
cp /path/to/tariffs.pdf storage/pdfs/

# Verify
ls -la storage/pdfs/
```

### Ingestion Hangs

```bash
# Check logs
tail -f logs/graphrag.log

# If Docling is slow, it's normal for large PDFs
# Be patient - typically 5-30 minutes per 100 pages
```

---

## Next Steps

1. **Test with sample queries:**

```bash
# Terminal 1: Start server
python -m api.main

# Terminal 2: Test queries
curl -X POST http://localhost:8000/rag \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is single phase service?",
    "model": "gpt-4"
  }'
```

2. **Connect your frontend:**

Configure frontend to call `http://localhost:8000/rag` with same schema.

3. **Monitor performance:**

- Check `/graph/status` to see what's indexed
- Monitor logs in `logs/graphrag.log`
- Adjust `GRAPH_TOP_K` and `IMAGE_TOP_K` if needed

4. **Deploy to production:**

See README.md for production deployment guide.

---

## Useful Commands

```bash
# Logs
docker-compose logs -f api
docker-compose logs -f neo4j

# Shell access
docker-compose exec api bash
docker-compose exec neo4j cypher-shell

# Restart
docker-compose restart api

# Full reset (clears database!)
docker-compose down -v
docker-compose up -d

# Status
docker-compose ps

# Stop
docker-compose stop

# Remove containers and volumes
docker-compose down -v
```

---

## Support

For issues:
1. Check logs: `docker-compose logs api`
2. Verify Neo4j: `docker-compose ps neo4j`
3. Test connectivity: `curl http://localhost:8000/health`
4. Check `.env` configuration

Questions? Review README.md for detailed documentation.
