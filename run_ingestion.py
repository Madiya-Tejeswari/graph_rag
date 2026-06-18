"""
Offline ingestion script.
Run once to build knowledge graph and image indexes.

Usage:
    python run_ingestion.py
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run ingestion pipeline."""
    try:
        logger.info("=" * 80)
        logger.info("PG&E GraphRAG Ingestion Pipeline")
        logger.info("=" * 80)
        
        # Import after logging setup
        from config import settings
        from storage.neo4j_client import Neo4jClient
        from storage.image_store import ImageStore
        from ingestion.pipeline import IngestionPipeline
        
        # Check Neo4j connection
        logger.info("Connecting to Neo4j...")
        neo4j_client = Neo4jClient(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=settings.neo4j_database
        )
        neo4j_client.connect()
        logger.info("✓ Neo4j connected")
        
        # Check PDF files
        logger.info(f"\nChecking PDF files in {settings.pdf_dir}...")
        pdf_paths = [p for p in settings.pdf_paths if Path(p).exists()]
        
        if not pdf_paths:
            logger.error(f"No PDFs found in {settings.pdf_dir}")
            logger.error("Place your PDFs here:")
            for pdf_path in settings.pdf_paths:
                logger.error(f"  - {pdf_path}")
            return 1
        
        logger.info(f"Found {len(pdf_paths)} PDF(s):")
        for pdf_path in pdf_paths:
            file_size = Path(pdf_path).stat().st_size / (1024*1024)  # MB
            logger.info(f"  - {Path(pdf_path).name} ({file_size:.1f} MB)")
        
        # Initialize image store
        logger.info("\nInitializing image storage...")
        image_store = ImageStore(
            image_dir=settings.image_dir,
            metadata_dir=settings.image_metadata_dir,
            embeddings_dir=settings.image_embeddings_dir
        )
        logger.info("✓ Image store initialized")
        
        # Clear existing graph if needed
        response = input("\nClear existing graph database? (y/n): ").lower().strip()
        if response == 'y':
            logger.info("Clearing Neo4j database...")
            neo4j_client.clear_database(confirm=True)
            logger.info("✓ Database cleared")
        
        # Run ingestion pipeline
        logger.info("\n" + "=" * 80)
        logger.info("Starting Ingestion Pipeline")
        logger.info("=" * 80)
        
        pipeline = IngestionPipeline(neo4j_client, image_store)
        pipeline.run(pdf_paths=pdf_paths)
        
        # Print statistics
        logger.info("\n" + "=" * 80)
        logger.info("Ingestion Complete - Final Statistics")
        logger.info("=" * 80)
        
        stats = pipeline.get_statistics()
        logger.info(f"Documents:     {stats.get('documents', 0)}")
        logger.info(f"Pages:         {stats.get('pages', 0)}")
        logger.info(f"Entities:      {stats.get('entities', 0)}")
        logger.info(f"Relationships: {stats.get('relationships', 0)}")
        logger.info(f"Tables:        {stats.get('tables', 0)}")
        logger.info(f"Figures:       {stats.get('figures', 0)}")
        logger.info(f"Images:        {image_store.get_image_count()}")
        
        logger.info("\n✓ Ingestion pipeline completed successfully!")
        logger.info("\nNext steps:")
        logger.info("1. Start the API server: python -m api.main")
        logger.info("2. Test health: curl http://localhost:8000/health")
        logger.info("3. Check graph status: curl http://localhost:8000/graph/status")
        logger.info("4. Visit documentation: http://localhost:8000/docs")
        
        neo4j_client.close()
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n\nIngestion interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"\n\n✗ Ingestion failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
