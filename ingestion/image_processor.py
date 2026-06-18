"""
Image processor for caption generation and embedding.
Prepares images for similarity search.
"""

import logging
from typing import Dict, Any, List
from storage.image_store import ImageStore
from config import settings

logger = logging.getLogger(__name__)


class ImageProcessor:
    """Processes images for search and retrieval."""
    
    def __init__(self, image_store: ImageStore, neo4j_client=None):
        """
        Initialize image processor.
        
        Args:
            image_store: Image storage manager
            neo4j_client: Neo4j database client
        """
        self.image_store = image_store
        self.neo4j_client = neo4j_client
        self.caption_generator = self._load_caption_model()
    
    def _load_caption_model(self):
        """Load image captioning model."""
        try:
            if settings.IMAGE_CAPTION_MODEL == "llava":
                # Using local LLaVA model
                from ollama import Client
                client = Client(base_url=settings.OLLAMA_BASE_URL)
                logger.info("Loaded LLaVA for image captioning")
                return client
            else:
                logger.warning(f"Unknown caption model: {settings.IMAGE_CAPTION_MODEL}")
                return None
        except Exception as e:
            logger.warning(f"Error loading caption model: {str(e)}")
            return None
    
    def process_batch(self, images: List[Dict], document_name: str):
        """
        Process batch of images.
        
        Args:
            images: List of image dictionaries
            document_name: Name of source document
        """
        try:
            logger.info(f"Processing {len(images)} images from {document_name}")
            
            for image in images:
                self.process_image(image, document_name)
            
            logger.info(f"Completed processing {len(images)} images")
            
        except Exception as e:
            logger.error(f"Error in batch processing: {str(e)}")
    
    def process_image(self, image: Dict, document_name: str) -> bool:
        """
        Process single image.
        
        Args:
            image: Image dictionary with 'data' and metadata
            document_name: Source document name
            
        Returns:
            True if successful
        """
        try:
            image_id = image.get("image_id", "")
            image_data = image.get("data")
            page_number = image.get("page_number")
            
            if not image_id or not image_data:
                logger.warning("Missing image ID or data")
                return False
            
            # Use caption from PDF extraction first, fall back to AI captioning
            caption = image.get("caption", "")
            figure_name = image.get("figure_name", "")
            
            if not caption:
                # No caption from PDF text, try generating one
                caption = self._generate_caption(image_data)
            
            # Prepare metadata
            metadata = {
                "image_id": image_id,
                "page_number": page_number,
                "document_name": document_name,
                "caption": caption,
                "figure_name": figure_name,
                "width": image.get("width", 0),
                "height": image.get("height", 0),
                "source_type": "image"
            }
            
            # Save image
            success = self.image_store.save_image(image_id, image_data, metadata)
            
            # Store in Neo4j
            if success and self.neo4j_client:
                import base64
                image_b64 = base64.b64encode(image_data).decode('utf-8')
                data_uri = f"data:image/png;base64,{image_b64}"
                
                query = """
                MATCH (d:Document {name: $document_name})
                MATCH (p:Page {document_name: $document_name, page_number: $page_number})
                MERGE (i:Image {image_id: $image_id})
                SET i.image_base64 = $data_uri,
                    i.page_number = $page_number,
                    i.document_name = $document_name,
                    i.caption = $caption,
                    i.figure_name = $figure_name,
                    i.width = $width,
                    i.height = $height
                
                MERGE (d)-[:HAS_IMAGE]->(i)
                MERGE (p)-[:HAS_IMAGE]->(i)
                
                WITH i
                MATCH (f:Figure {document_name: $document_name, page_number: $page_number})
                WHERE f.name = $figure_name OR i.caption CONTAINS f.description
                MERGE (i)-[:DEPICTS_FIGURE]->(f)
                
                WITH i
                OPTIONAL MATCH (tc_near:TextChunk {document_name: $document_name, page_number: $page_number})
                WITH i, tc_near
                WHERE tc_near IS NOT NULL
                MERGE (i)-[:NEAR_TEXT]->(tc_near)
                
                WITH i
                OPTIONAL MATCH (tc_ref:TextChunk {document_name: $document_name})
                WHERE $figure_name <> "" AND toLower(tc_ref.content) CONTAINS toLower($figure_name)
                WITH i, tc_ref
                WHERE tc_ref IS NOT NULL
                MERGE (tc_ref)-[:REFERENCES_IMAGE]->(i)
                """
                
                self.neo4j_client.execute_query(query, {
                    "image_id": image_id,
                    "data_uri": data_uri,
                    "page_number": page_number,
                    "document_name": document_name,
                    "caption": caption,
                    "figure_name": figure_name,
                    "width": image.get("width", 0),
                    "height": image.get("height", 0)
                })
            
            if success:
                logger.debug(f"Processed image: {image_id}"
                           f"{' (' + figure_name + ')' if figure_name else ''}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            return False
    
    def _generate_caption(self, image_data: bytes) -> str:
        """
        Generate caption for image.
        
        Args:
            image_data: Image binary data
            
        Returns:
            Generated caption
        """
        try:
            if not self.caption_generator:
                return "Image (caption generation unavailable)"
            
            # Convert image data to base64 for Ollama
            import base64
            image_b64 = base64.b64encode(image_data).decode('utf-8')
            
            # Generate caption using LLaVA
            response = self.caption_generator.generate(
                model=settings.IMAGE_CAPTION_MODEL,
                prompt="Describe what you see in this image concisely.",
                images=[image_b64],
                stream=False
            )
            
            caption = response.get("response", "").strip()
            return caption if caption else "Image"
            
        except Exception as e:
            logger.debug(f"Error generating caption: {str(e)}")
            return "Image (caption generation failed)"
    
    def regenerate_captions(self, image_ids: List[str] = None) -> int:
        """
        Regenerate captions for images.
        
        Args:
            image_ids: Specific image IDs to regenerate, or None for all
            
        Returns:
            Number of successfully regenerated captions
        """
        try:
            if image_ids is None:
                image_ids = self.image_store.get_all_image_ids()
            
            updated = 0
            for image_id in image_ids:
                image_data = self.image_store.get_image(image_id)
                if image_data:
                    caption = self._generate_caption(image_data)
                    metadata = self.image_store.get_metadata(image_id)
                    if metadata:
                        metadata["caption"] = caption
                        if self.image_store.save_image(image_id, image_data, metadata):
                            updated += 1
            
            logger.info(f"Regenerated {updated} captions")
            return updated
            
        except Exception as e:
            logger.error(f"Error regenerating captions: {str(e)}")
            return 0
