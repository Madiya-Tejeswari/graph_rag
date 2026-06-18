"""
Image search engine using embedding-based similarity.
Used for visual questions about diagrams and drawings.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional
from storage.image_store import ImageStore
from config import settings

logger = logging.getLogger(__name__)

# Lazy import — sentence_transformers depends on torch which may not be available
SentenceTransformer = None
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    logger.warning(f"sentence_transformers not available: {e}. Image search will be disabled.")


class ImageSearchEngine:
    """Search engine for image retrieval using embeddings."""
    
    def __init__(self, image_store: ImageStore):
        """
        Initialize image search engine.
        
        Args:
            image_store: Image storage manager
        """
        self.image_store = image_store
        self.embedding_model = self._load_embedding_model()
        self.index = None
        self.image_ids = []
        self.embeddings = []
        
        # Load existing embeddings
        self._load_embeddings()
    
    def _load_embedding_model(self):
        """Load sentence transformer model for embeddings."""
        if SentenceTransformer is None:
            logger.warning("SentenceTransformer not available — image embedding search disabled")
            return None
        try:
            model = SentenceTransformer(settings.embedding_model)
            logger.info(f"Loaded embedding model: {settings.embedding_model}")
            return model
        except Exception as e:
            logger.error(f"Error loading embedding model: {str(e)}")
            return None
    
    def _load_embeddings(self):
        """Load all stored embeddings into memory."""
        try:
            image_ids = self.image_store.get_all_image_ids()
            embeddings = []
            valid_ids = []
            
            for image_id in image_ids:
                embedding = self.image_store.get_embedding(image_id)
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_ids.append(image_id)
            
            if embeddings:
                self.embeddings = np.array(embeddings)
                self.image_ids = valid_ids
                logger.info(f"Loaded {len(valid_ids)} image embeddings")
            
        except Exception as e:
            logger.error(f"Error loading embeddings: {str(e)}")
    
    def search(self, query: str, top_k: int = 5, threshold: float = 0.7) -> List[Dict]:
        """
        Search for images similar to the query.
        
        Pipeline:
        1. Embed query text
        2. Compute similarity with image embeddings
        3. Filter by threshold
        4. Return top results
        
        Args:
            query: Search query
            top_k: Number of top results
            threshold: Similarity threshold (0-1)
            
        Returns:
            List of similar images with metadata
        """
        try:
            # Check if query is visual in nature
            if not self._is_visual_query(query):
                logger.debug("Query is not visual in nature, skipping image search")
                return []
            
            # Check if we have embeddings
            if len(self.embeddings) == 0:
                logger.debug("No image embeddings available")
                return []
            
            logger.info(f"Searching images for: {query}")
            
            # Embed query
            query_embedding = self.embedding_model.encode(query)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Compute similarity with all images
            similarities = np.dot(self.embeddings, query_embedding)
            
            # Get top matches above threshold
            valid_indices = np.where(similarities >= threshold)[0]
            
            if len(valid_indices) == 0:
                logger.debug(f"No images found with similarity > {threshold}")
                return []
            
            # Sort by similarity
            top_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1][:top_k]]
            
            results = []
            for idx in top_indices:
                image_id = self.image_ids[idx]
                similarity = float(similarities[idx])
                
                image_data = self.image_store.get_image_with_metadata(image_id)
                if image_data:
                    image_data["similarity"] = similarity
                    results.append(image_data)
            
            logger.info(f"Image search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error in image search: {str(e)}")
            return []
    
    def _is_visual_query(self, query: str) -> bool:
        """
        Detect if query is asking for visual content.
        
        Args:
            query: Search query
            
        Returns:
            True if visual query
        """
        visual_keywords = [
            "show", "diagram", "figure", "drawing", "look",
            "image", "picture", "visualization", "visual",
            "schematic", "sketch", "layout", "design",
            "connection", "circuit"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in visual_keywords)
    
    def add_image(self, image_id: str, image_data: bytes, 
                  caption: str, metadata: Dict[str, Any]) -> bool:
        """
        Add image to search index.
        
        Args:
            image_id: Image identifier
            image_data: Image binary data
            caption: Image caption/description
            metadata: Image metadata
            
        Returns:
            True if successful
        """
        try:
            # Save image
            self.image_store.save_image(image_id, image_data, metadata)
            
            # Generate and save embedding
            embedding = self.embedding_model.encode(caption)
            embedding = embedding / np.linalg.norm(embedding)
            self.image_store.save_embedding(image_id, embedding)
            
            # Update in-memory index
            self.image_ids.append(image_id)
            self.embeddings.append(embedding)
            
            logger.debug(f"Added image to search index: {image_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding image: {str(e)}")
            return False
    
    def remove_image(self, image_id: str) -> bool:
        """
        Remove image from search index.
        
        Args:
            image_id: Image identifier
            
        Returns:
            True if successful
        """
        try:
            # Remove from storage
            self.image_store.delete_image(image_id)
            
            # Remove from in-memory index
            if image_id in self.image_ids:
                idx = self.image_ids.index(image_id)
                self.image_ids.pop(idx)
                self.embeddings = np.delete(self.embeddings, idx, axis=0)
            
            logger.debug(f"Removed image from search index: {image_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing image: {str(e)}")
            return False
    
    def get_images_by_page(self, page_number: int) -> List[Dict]:
        """
        Get all images on a specific page.
        
        Args:
            page_number: Page number
            
        Returns:
            List of images
        """
        image_ids = self.image_store.search_by_page(page_number)
        results = []
        
        for image_id in image_ids:
            image_data = self.image_store.get_image_with_metadata(image_id)
            if image_data:
                results.append(image_data)
        
        return results
    
    def get_images_by_document(self, document_name: str) -> List[Dict]:
        """
        Get all images in a specific document.
        
        Args:
            document_name: Document name
            
        Returns:
            List of images
        """
        image_ids = self.image_store.search_by_document(document_name)
        results = []
        
        for image_id in image_ids:
            image_data = self.image_store.get_image_with_metadata(image_id)
            if image_data:
                results.append(image_data)
        
        return results
    
    def update_image_embeddings(self, batch_captions: Dict[str, str]) -> bool:
        """
        Update embeddings for multiple images.
        
        Args:
            batch_captions: Dictionary of image_id -> caption
            
        Returns:
            True if successful
        """
        try:
            for image_id, caption in batch_captions.items():
                embedding = self.embedding_model.encode(caption)
                embedding = embedding / np.linalg.norm(embedding)
                self.image_store.save_embedding(image_id, embedding)
            
            # Reload embeddings
            self._load_embeddings()
            logger.info(f"Updated embeddings for {len(batch_captions)} images")
            return True
            
        except Exception as e:
            logger.error(f"Error updating embeddings: {str(e)}")
            return False
