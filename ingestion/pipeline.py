"""
Main ingestion pipeline orchestrator.
Coordinates PDF parsing, graph building, and image processing.
"""

import logging
from pathlib import Path
from typing import List, Optional
from ingestion.pdf_parser import PDFParser
from ingestion.graph_builder import GraphBuilder
from ingestion.image_processor import ImageProcessor
from storage.neo4j_client import Neo4jClient
from storage.image_store import ImageStore
from config import settings

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """Main ingestion pipeline."""
    
    def __init__(self, neo4j_client: Neo4jClient, image_store: ImageStore):
        """
        Initialize ingestion pipeline.
        
        Args:
            neo4j_client: Neo4j database client
            image_store: Image storage manager
        """
        self.neo4j_client = neo4j_client
        self.image_store = image_store
        self.pdf_parser = PDFParser()
        self.graph_builder = GraphBuilder(neo4j_client)
        self.image_processor = ImageProcessor(image_store, neo4j_client)
        
        logger.info("Ingestion pipeline initialized")
    
    def run(self, pdf_paths: List[str]):
        """
        Run complete ingestion pipeline.
        
        Args:
            pdf_paths: List of PDF file paths to ingest
        """
        try:
            logger.info(f"Starting ingestion of {len(pdf_paths)} PDFs")
            
            # Create indexes for performance
            self.neo4j_client.create_indexes()
            
            # Process each PDF
            for pdf_path in pdf_paths:
                if Path(pdf_path).exists():
                    logger.info(f"Processing: {pdf_path}")
                    self._process_pdf(pdf_path)
                else:
                    logger.warning(f"PDF not found: {pdf_path}")
            
            logger.info("Ingestion pipeline completed successfully")
            
        except Exception as e:
            logger.error(f"Error in ingestion pipeline: {str(e)}", exc_info=True)
            raise
    
    def _process_pdf(self, pdf_path: str):
        """
        Process a single PDF file through the pipeline.
        
        Args:
            pdf_path: Path to PDF file
        """
        try:
            # Step 1: Parse PDF
            logger.info(f"Step 1: Parsing PDF")
            extracted_data = self.pdf_parser.parse(pdf_path)
            
            # Step 2: Build knowledge graph
            logger.info(f"Step 2: Building knowledge graph")
            self.graph_builder.build(
                extracted_data=extracted_data,
                document_name=Path(pdf_path).name
            )
            
            # Step 3: Process images
            if settings.enable_image_captioning:
                logger.info(f"Step 3: Processing images")
                self.image_processor.process_batch(
                    images=extracted_data.get("images", []),
                    document_name=Path(pdf_path).name
                )
            
            logger.info(f"Completed processing: {pdf_path}")
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}", exc_info=True)
            raise
    
    def get_statistics(self) -> dict:
        """Get ingestion statistics."""
        return self.neo4j_client.get_graph_stats()
