"""
FastAPI main application for PG&E GraphRAG system.
"""

import logging
import time
import re
import base64
from datetime import datetime
from typing import Optional, List
import asyncio

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from api.schemas import RAGRequest, RAGResponse, Source, ImageResult, Metadata
from config import settings
from retrieval.graph_search import GraphSearchEngine
from retrieval.image_search import ImageSearchEngine
from retrieval.context_builder import ContextBuilder
from generation.prompt_builder import PromptBuilder
from generation.llm_client import LLMClient
from storage.neo4j_client import Neo4jClient
from storage.image_store import ImageStore
from ingestion.pipeline import IngestionPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="PG&E GraphRAG Backend",
    description="Production-ready GraphRAG system for PG&E Greenbook and Tariff documents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
neo4j_client: Optional[Neo4jClient] = None
image_store: Optional[ImageStore] = None
graph_search: Optional[GraphSearchEngine] = None
image_search: Optional[ImageSearchEngine] = None
context_builder: Optional[ContextBuilder] = None
prompt_builder: Optional[PromptBuilder] = None
llm_client: Optional[LLMClient] = None


@app.on_event("startup")
async def startup_event():
    """Initialize connections and load indexes on startup."""
    global neo4j_client, image_store, graph_search, image_search, context_builder, prompt_builder, llm_client
    
    try:
        logger.info("Initializing GraphRAG system...")
        
        # Initialize Neo4j connection with retry
        neo4j_client = Neo4jClient(
            uri=settings.NEO4J_URI,
            username=settings.NEO4J_USERNAME,
            password=settings.NEO4J_PASSWORD,
            database=settings.NEO4J_DATABASE
        )
        
        connected = False
        for attempt in range(3):
            if neo4j_client.connect():
                connected = True
                break
            logger.warning(f"Neo4j connection attempt {attempt + 1}/3 failed, retrying in 2s...")
            await asyncio.sleep(2)
        
        if connected:
            logger.info("Neo4j connected")
        else:
            logger.warning("Neo4j not available at startup — will retry on first query")
        
        # Initialize image store
        image_store = ImageStore(
            image_dir=settings.IMAGE_DIR,
            metadata_dir=settings.IMAGE_METADATA_DIR,
            embeddings_dir=settings.IMAGE_EMBEDDINGS_DIR
        )
        logger.info("Image store initialized")
        
        # Initialize retrieval engines
        graph_search = GraphSearchEngine(neo4j_client)
        image_search = ImageSearchEngine(image_store)
        context_builder = ContextBuilder()
        
        # Initialize generation components
        prompt_builder = PromptBuilder()
        llm_client = LLMClient()
        
        logger.info("GraphRAG system initialized successfully")
        
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Close connections on shutdown."""
    try:
        if neo4j_client:
            neo4j_client.close()
            logger.info("Neo4j connection closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return JSONResponse(
        status_code=200,
        content={"status": "healthy", "service": "PG&E GraphRAG Backend"}
    )


@app.get("/graph/status", tags=["Graph"])
async def graph_status():
    """Get graph database status and statistics."""
    try:
        if not neo4j_client:
            return JSONResponse(
                status_code=503,
                content={"status": "unavailable", "error": "Neo4j not initialized"}
            )
        
        stats = neo4j_client.get_graph_stats()
        return JSONResponse(
            status_code=200,
            content={
                "status": "ready" if stats.get("documents", 0) > 0 else "empty",
                **stats
            }
        )
    except Exception as e:
        logger.error(f"Error getting graph status: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )


@app.post("/ingest", tags=["Ingestion"])
async def ingest(background_tasks: BackgroundTasks):
    """
    Trigger offline ingestion pipeline.
    This is a long-running process that should run once.
    """
    try:
        if not neo4j_client or not image_store:
            raise HTTPException(
                status_code=503,
                detail="System not properly initialized"
            )
        
        # Run ingestion in background
        background_tasks.add_task(
            run_ingestion_task,
            neo4j_client,
            image_store
        )
        
        return JSONResponse(
            status_code=202,
            content={
                "status": "ingestion_started",
                "message": "Ingestion pipeline started in background"
            }
        )
    except Exception as e:
        logger.error(f"Error starting ingestion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def run_ingestion_task(neo4j_client: Neo4jClient, image_store: ImageStore):
    """Run the ingestion pipeline in the background."""
    try:
        logger.info("Starting ingestion pipeline...")
        pipeline = IngestionPipeline(neo4j_client, image_store)
        pipeline.run(pdf_paths=settings.PDF_PATHS)
        logger.info("Ingestion pipeline completed successfully")
    except Exception as e:
        logger.error(f"Error in ingestion pipeline: {str(e)}")


@app.post("/rag", response_model=RAGResponse, tags=["RAG"])
async def rag_query(request: RAGRequest):
    """
    Main RAG endpoint.
    Processes user queries using graph retrieval and LLM generation.
    """
    start_time = time.time()
    retrieval_start = time.time()
    
    try:
        # Validate inputs
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        if not request.model:
            raise HTTPException(status_code=400, detail="Model must be specified")
        
        if not neo4j_client or not graph_search or not image_search:
            raise HTTPException(status_code=503, detail="System not initialized")
        
        # Try to reconnect Neo4j if it wasn't connected at startup
        if neo4j_client.driver is None:
            logger.info("Neo4j driver not connected, attempting reconnect...")
            neo4j_client.connect()
        
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Classify query to decide what content to include
        query_lower = request.query.lower()
        
        # Determine if images are needed
        image_keywords = [
            "figure", "diagram", "drawing", "layout", "look like", "looks like",
            "show", "illustration", "schematic", "picture", "image", "visual",
            "connection", "installation", "underground", "overhead", "wiring",
            "meter", "panel", "enclosure", "configuration"
        ]
        needs_images = any(kw in query_lower for kw in image_keywords)
        
        # Determine if tables are likely needed
        table_keywords = [
            "table", "list", "compare", "comparison", "vs", "versus",
            "color code", "colour code", "specifications", "rates", "schedule",
            "requirements", "sizes", "dimensions", "voltage", "load",
            "clearance", "minimum", "maximum", "limit"
        ]
        needs_tables = any(kw in query_lower for kw in table_keywords)
        
        # Graph retrieval (top_k=5 for focused results)
        graph_results = graph_search.search(
            query=request.query,
            top_k=5,
            expansion_hops=settings.EXPANSION_HOPS,
            search_tables=needs_tables,
            search_figures=needs_images
        )
        logger.info(f"Retrieved {len(graph_results)} graph results")
        
        # Image retrieval — only when relevant
        image_results = []
        if needs_images:
            image_results = image_search.search(
                query=request.query,
                top_k=3,
                threshold=settings.IMAGE_SIMILARITY_THRESHOLD
            )
        logger.info(f"Retrieved {len(image_results)} image results (needs_images={needs_images})")
        
        # Build context
        context = context_builder.build(
            graph_results=graph_results,
            image_results=image_results
        )
        
        # Generate prompt
        prompt = prompt_builder.build(
            query=request.query,
            context=context
        )
        
        retrieval_time = int((time.time() - retrieval_start) * 1000)
        
        # Generate answer with LLM
        generation_start = time.time()
        answer, token_usage = llm_client.generate(
            prompt=prompt,
            model=request.model
        )
        generation_time = int((time.time() - generation_start) * 1000)
        
        # Extract sources from graph results
        sources = extract_sources(graph_results)
        
        # Collect images — only when query needs visuals OR the answer references figures
        response_images = []
        answer_lower = answer.lower()
        answer_mentions_figure = any(w in answer_lower for w in ["figure", "diagram", "illustration", "fig."])
        
        if needs_images or answer_mentions_figure:
            if not image_results and answer_mentions_figure:
                # Perform a quick search to find relevant images if we didn't already
                image_results = image_search.search(
                    query=request.query,
                    top_k=3,
                    threshold=0.5
                )
            
            seen_image_ids = set()
            for img_data in image_results:
                img_id = img_data.get("image_id")
                if img_id and img_id not in seen_image_ids:
                    img_path = image_store.image_dir / f"{img_id}.png"
                    if img_path.exists() and img_path.stat().st_size > 1000:
                        # Read image and encode as base64 data URI
                        with open(img_path, "rb") as f:
                            img_bytes = f.read()
                        img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                        response_images.append(ImageResult(
                            image_base64=f"data:image/png;base64,{img_b64}"
                        ))
                        seen_image_ids.add(img_id)
                
                if len(response_images) >= 1:
                    break
            
            # If no images found (e.g. sentence transformers missing), query Neo4j directly!
            if not response_images:
                fig_refs = re.findall(r'(?:figure|fig\.?|diagram|drawing)\s*([\d-]+|[a-z0-9]+)', answer_lower)
                
                stop_words = {"what", "is", "the", "of", "a", "an", "for", "and", "or", "to", "in", "how", "why", "show", "me", "image", "figure", "diagram", "looks", "like"}
                words = request.query.split()
                keywords = [w.strip("?.,;:").lower() for w in words if w.strip("?.,;:").lower() not in stop_words and len(w.strip("?.,;:")) > 2]
                
                cypher_queries = []
                if fig_refs:
                    for ref in fig_refs[:1]:
                        cypher_queries.append((
                            """
                            MATCH (i:Image)
                            WHERE toLower(i.figure_name) CONTAINS $ref OR toLower(i.caption) CONTAINS $ref
                            RETURN i.image_base64 AS image_base64, i.image_id AS image_id
                            LIMIT 1
                            """,
                            {"ref": ref}
                        ))
                
                if not fig_refs and keywords:
                    or_conditions = " OR ".join([f"toLower(i.caption) CONTAINS '{kw}' OR toLower(i.figure_name) CONTAINS '{kw}'" for kw in keywords[:6]])
                    if or_conditions:
                        cypher_queries.append((
                            f"""
                            MATCH (i:Image)
                            WHERE {or_conditions}
                            RETURN i.image_base64 AS image_base64, i.image_id AS image_id
                            LIMIT 1
                            """,
                            {}
                        ))
                
                for query, params in cypher_queries:
                    records = neo4j_client.execute_query(query, params)
                    for r in records:
                        img_id = r.get("image_id")
                        if img_id and img_id not in seen_image_ids:
                            response_images.append(ImageResult(image_base64=r.get("image_base64")))
                            seen_image_ids.add(img_id)
                        if len(response_images) >= 1:
                            break
                    if len(response_images) >= 1:
                        break
        
        logger.info(f"Including {len(response_images)} images in response")
        
        # Build response
        total_time = int((time.time() - start_time) * 1000)
        
        metadata = Metadata(
            retrievaltimems=retrieval_time,
            generationtimems=generation_time,
            totaltimems=total_time,
            generatedat=datetime.now().isoformat(),
            inputtokens=token_usage.get('input_tokens', 0),
            outputtokens=token_usage.get('output_tokens', 0),
            totaltokens=token_usage.get('total_tokens', 0),
            modelused=request.model,
            retrievalmethod="vector_search"
        )
        
        response = RAGResponse(
            status="success",
            query=request.query,
            answer=answer,
            sources=sources,
            images=response_images,
            metadata=metadata
        )
        
        logger.info(f"Query processed successfully in {total_time}ms")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing RAG query: {error_msg}", exc_info=True)
        
        # Check for rate limit errors
        if "429" in error_msg or "rate_limit" in error_msg.lower():
            raise HTTPException(
                status_code=429, 
                detail="⏳ API rate limit reached. Please try a different model from the sidebar or wait a few minutes."
            )
        
        # Check for decommissioned models
        if "decommissioned" in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail="⚠️ This model has been decommissioned. Please select a different model from the sidebar."
            )
        
        # Check for invalid API key
        if "invalid_api_key" in error_msg.lower() or "401" in error_msg:
            raise HTTPException(
                status_code=401,
                detail="🔑 Invalid API key. Please check your GROQ_API_KEY in the .env file."
            )
        
        raise HTTPException(status_code=500, detail=f"Error processing query: {error_msg}")


def extract_sources(graph_results: list) -> list:
    """
    Extract unique page sources from graph retrieval results.
    
    Returns sources with proper title, URL, and page number dynamically based on the document.
    """
    sources_dict = {}
    
    for result in graph_results:
        provenance = result.get('provenance', {})
        page_number = provenance.get('page_number')
        # Default to greenbook.pdf if not found
        doc_name = provenance.get('document_name', 'greenbook-manual-full.pdf')
        
        # Map internal 'greenbook.pdf' to the frontend's expected filename
        if doc_name == "greenbook.pdf":
            doc_name = "greenbook-manual-full.pdf"
            
        if page_number:
            key = f"{doc_name}_{page_number}"
            if key not in sources_dict:
                sources_dict[key] = {
                    "page_number": page_number,
                    "document_name": doc_name
                }
    
    # Convert to Source objects matching frontend contract
    sources = []
    
    # Sort keys to ensure stable output order
    for key in sorted(sources_dict.keys()):
        val = sources_dict[key]
        page_num = val["page_number"]
        doc_name = val["document_name"]
        
        # Format the title (e.g. Greenbook Manual Full - Page 29)
        clean_title = doc_name.replace(".pdf", "").replace("-", " ").title()
        if clean_title.lower() == "greenbook manual full":
            clean_title = "PG&E Greenbook Manual Full"
            
        sources.append(
            Source(
                title=f"{clean_title} - Page {page_num}",
                url=f"/docs/{doc_name}#page={page_num}",
                pageno=str(page_num)
            )
        )
    
    return sources


def parse_table_content(content: str) -> tuple:
    """
    Parse pipe-separated table content from Neo4j into headers and rows.
    
    Handles complex tables by:
    - Filtering out rows with too many empty cells (sub-headers)
    - Skipping footnote/instruction rows
    - Normalizing column counts to the most common width
    
    Returns:
        Tuple of (headers: List[str], rows: List[List[str]])
    """
    if not content or not content.strip():
        return [], []
    
    # Split by newlines to get rows
    raw_rows = [line.strip() for line in content.strip().split('\n') if line.strip()]
    
    if not raw_rows:
        return [], []
    
    parsed_rows = []
    for raw_row in raw_rows:
        # Split by pipe, clean up
        cells = [cell.strip() for cell in raw_row.split('|')]
        # Remove empty leading/trailing cells from pipe format
        cells = [c for c in cells if c]
        if cells:
            parsed_rows.append(cells)
    
    if not parsed_rows:
        return [], []
    
    # Find the most common column count (to identify the "real" structure)
    col_counts = {}
    for row in parsed_rows:
        n = len(row)
        col_counts[n] = col_counts.get(n, 0) + 1
    
    # Use the most common column count as the expected width
    expected_cols = max(col_counts, key=col_counts.get)
    
    # Filter to rows that match the expected column count (or close to it)
    valid_rows = []
    for row in parsed_rows:
        # Skip rows with very different column count (sub-headers, footnotes)
        if abs(len(row) - expected_cols) > 1:
            continue
        
        # Skip rows where most cells are empty
        non_empty = sum(1 for c in row if c.strip())
        if non_empty < max(1, len(row) // 2):
            continue
        
        # Skip footnote/instruction rows (single long cell)
        if len(row) == 1 and len(row[0]) > 30:
            continue
        
        # Normalize column count
        if len(row) < expected_cols:
            row = row + [''] * (expected_cols - len(row))
        elif len(row) > expected_cols:
            row = row[:expected_cols]
        
        valid_rows.append(row)
    
    if not valid_rows:
        return [], []
    
    # First valid row is headers, rest are data rows
    headers = valid_rows[0]
    rows = valid_rows[1:]
    
    # Skip if we only have headers and no data
    if not rows:
        return [], []
    
    return headers, rows


@app.get("/images/{image_id}", tags=["Images"])
async def get_image(image_id: str):
    """Serve an image by its ID."""
    try:
        if not image_store:
            raise HTTPException(status_code=503, detail="Image store not initialized")
        
        image_data = image_store.get_image(image_id)
        if image_data is None:
            raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")
        
        return Response(content=image_data, media_type="image/png")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving image {image_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/images/page/{page_number}", tags=["Images"])
async def get_page_images(page_number: int):
    """Get all images on a specific page."""
    try:
        if not image_store:
            raise HTTPException(status_code=503, detail="Image store not initialized")
        
        page_image_ids = image_store.search_by_page(page_number)
        images = []
        
        for img_id in page_image_ids:
            # Skip tiny placeholder images
            img_path = image_store.image_dir / f"{img_id}.png"
            if img_path.exists() and img_path.stat().st_size > 1000:
                meta = image_store.get_metadata(img_id)
                if meta:
                    images.append({
                        "image_id": img_id,
                        "page_number": meta.get("page_number", 0),
                        "caption": meta.get("caption", ""),
                    })
        
        return JSONResponse(
            status_code=200,
            content={"page_number": page_number, "images": images}
        )
    except Exception as e:
        logger.error(f"Error getting page images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "PG&E GraphRAG Backend",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "graph_status": "/graph/status",
            "ingest": "/ingest",
            "rag": "/rag",
            "images": "/images/{image_id}",
            "page_images": "/images/page/{page_number}"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG
    )
