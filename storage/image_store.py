"""
Image storage manager for handling images, metadata, and embeddings.
Uses filesystem storage with JSON metadata and numpy embeddings.
"""

import logging
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np
from PIL import Image
from io import BytesIO

logger = logging.getLogger(__name__)


class ImageStore:
    """Manages image storage, metadata, and embeddings."""
    
    def __init__(self, image_dir: Path, metadata_dir: Path, embeddings_dir: Path):
        """
        Initialize image store.
        
        Args:
            image_dir: Directory for storing image files
            metadata_dir: Directory for storing image metadata
            embeddings_dir: Directory for storing image embeddings
        """
        self.image_dir = Path(image_dir)
        self.metadata_dir = Path(metadata_dir)
        self.embeddings_dir = Path(embeddings_dir)
        
        # Create directories if they don't exist
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ImageStore initialized at {image_dir}")
    
    def save_image(self, image_id: str, image_data: bytes, 
                   metadata: Dict[str, Any]) -> bool:
        """
        Save image with metadata.
        
        Args:
            image_id: Unique image identifier
            image_data: Image binary data (bytes)
            metadata: Image metadata dictionary
            
        Returns:
            True if successful
        """
        try:
            # Save image file
            image_path = self.image_dir / f"{image_id}.png"
            
            # Handle different input types
            if isinstance(image_data, bytes):
                with open(image_path, "wb") as f:
                    f.write(image_data)
            else:
                # Assume it's PIL Image or similar
                if hasattr(image_data, "save"):
                    image_data.save(image_path, format="PNG")
                else:
                    logger.error(f"Cannot save image data of type {type(image_data)}")
                    return False
            
            # Save metadata
            metadata_path = self.metadata_dir / f"{image_id}.json"
            metadata["image_id"] = image_id
            metadata["saved_at"] = datetime.now().isoformat()
            metadata["file_path"] = str(image_path)
            
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug(f"Saved image: {image_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving image {image_id}: {str(e)}")
            return False
    
    def get_image(self, image_id: str) -> Optional[bytes]:
        """
        Retrieve image binary data.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Image bytes if found
        """
        try:
            image_path = self.image_dir / f"{image_id}.png"
            
            if image_path.exists():
                with open(image_path, "rb") as f:
                    return f.read()
            
            logger.warning(f"Image not found: {image_id}")
            return None
        
        except Exception as e:
            logger.error(f"Error retrieving image {image_id}: {str(e)}")
            return None
    
    def get_metadata(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve image metadata.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Metadata dictionary if found
        """
        try:
            metadata_path = self.metadata_dir / f"{image_id}.json"
            
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    return json.load(f)
            
            logger.warning(f"Metadata not found: {image_id}")
            return None
        
        except Exception as e:
            logger.error(f"Error retrieving metadata for {image_id}: {str(e)}")
            return None
    
    def get_image_with_metadata(self, image_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve image data and metadata together.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Dictionary with image info if found
        """
        try:
            image_data = self.get_image(image_id)
            metadata = self.get_metadata(image_id)
            
            if image_data and metadata:
                # Convert bytes to base64 for API response
                import base64
                image_b64 = base64.b64encode(image_data).decode("utf-8")
                
                return {
                    "image_id": image_id,
                    "image_data": image_b64,
                    **metadata
                }
            
            return None
        
        except Exception as e:
            logger.error(f"Error retrieving image with metadata: {str(e)}")
            return None
    
    def save_embedding(self, image_id: str, embedding: np.ndarray) -> bool:
        """
        Save image embedding vector.
        
        Args:
            image_id: Image identifier
            embedding: Embedding vector (numpy array)
            
        Returns:
            True if successful
        """
        try:
            embedding_path = self.embeddings_dir / f"{image_id}.npy"
            
            # Ensure embedding is numpy array
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            # Normalize embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            np.save(embedding_path, embedding)
            logger.debug(f"Saved embedding for image: {image_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving embedding for {image_id}: {str(e)}")
            return False
    
    def get_embedding(self, image_id: str) -> Optional[np.ndarray]:
        """
        Retrieve image embedding vector.
        
        Args:
            image_id: Image identifier
            
        Returns:
            Embedding vector if found
        """
        try:
            embedding_path = self.embeddings_dir / f"{image_id}.npy"
            
            if embedding_path.exists():
                return np.load(embedding_path)
            
            logger.debug(f"Embedding not found: {image_id}")
            return None
        
        except Exception as e:
            logger.error(f"Error retrieving embedding for {image_id}: {str(e)}")
            return None
    
    def delete_image(self, image_id: str) -> bool:
        """
        Delete image, metadata, and embedding.
        
        Args:
            image_id: Image identifier
            
        Returns:
            True if successful
        """
        try:
            image_path = self.image_dir / f"{image_id}.png"
            metadata_path = self.metadata_dir / f"{image_id}.json"
            embedding_path = self.embeddings_dir / f"{image_id}.npy"
            
            deleted_count = 0
            
            if image_path.exists():
                image_path.unlink()
                deleted_count += 1
            
            if metadata_path.exists():
                metadata_path.unlink()
                deleted_count += 1
            
            if embedding_path.exists():
                embedding_path.unlink()
                deleted_count += 1
            
            if deleted_count > 0:
                logger.debug(f"Deleted image {image_id} ({deleted_count} files)")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error deleting image {image_id}: {str(e)}")
            return False
    
    def get_all_image_ids(self) -> List[str]:
        """
        Get list of all image IDs in storage.
        
        Returns:
            List of image IDs
        """
        try:
            image_ids = []
            
            for metadata_file in self.metadata_dir.glob("*.json"):
                image_id = metadata_file.stem
                image_ids.append(image_id)
            
            logger.debug(f"Found {len(image_ids)} images in storage")
            return image_ids
        
        except Exception as e:
            logger.error(f"Error listing image IDs: {str(e)}")
            return []
    
    def get_image_count(self) -> int:
        """
        Get total number of images in storage.
        
        Returns:
            Number of images
        """
        return len(self.get_all_image_ids())
    
    def search_by_page(self, page_number: int) -> List[str]:
        """
        Search for images on a specific page.
        
        Args:
            page_number: Page number
            
        Returns:
            List of image IDs on that page
        """
        try:
            matching_ids = []
            
            for image_id in self.get_all_image_ids():
                metadata = self.get_metadata(image_id)
                if metadata and metadata.get("page_number") == page_number:
                    matching_ids.append(image_id)
            
            logger.debug(f"Found {len(matching_ids)} images on page {page_number}")
            return matching_ids
        
        except Exception as e:
            logger.error(f"Error searching by page: {str(e)}")
            return []
    
    def search_by_document(self, document_name: str) -> List[str]:
        """
        Search for images in a specific document.
        
        Args:
            document_name: Document name
            
        Returns:
            List of image IDs from that document
        """
        try:
            matching_ids = []
            
            for image_id in self.get_all_image_ids():
                metadata = self.get_metadata(image_id)
                if metadata and metadata.get("document_name") == document_name:
                    matching_ids.append(image_id)
            
            logger.debug(f"Found {len(matching_ids)} images in document: {document_name}")
            return matching_ids
        
        except Exception as e:
            logger.error(f"Error searching by document: {str(e)}")
            return []
    
    def update_metadata(self, image_id: str, metadata_updates: Dict[str, Any]) -> bool:
        """
        Update metadata for an existing image.
        
        Args:
            image_id: Image identifier
            metadata_updates: Dictionary of fields to update
            
        Returns:
            True if successful
        """
        try:
            current_metadata = self.get_metadata(image_id)
            
            if current_metadata is None:
                logger.warning(f"Image not found: {image_id}")
                return False
            
            # Update metadata
            current_metadata.update(metadata_updates)
            current_metadata["updated_at"] = datetime.now().isoformat()
            
            # Save updated metadata
            metadata_path = self.metadata_dir / f"{image_id}.json"
            with open(metadata_path, "w") as f:
                json.dump(current_metadata, f, indent=2)
            
            logger.debug(f"Updated metadata for image: {image_id}")
            return True
        
        except Exception as e:
            logger.error(f"Error updating metadata for {image_id}: {str(e)}")
            return False
    
    def get_image_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored images.
        
        Returns:
            Dictionary with image statistics
        """
        try:
            all_ids = self.get_all_image_ids()
            
            stats = {
                "total_images": len(all_ids),
                "images_with_embeddings": 0,
                "documents": set(),
                "pages": set(),
                "total_size_mb": 0.0
            }
            
            for image_id in all_ids:
                # Check for embeddings
                embedding_path = self.embeddings_dir / f"{image_id}.npy"
                if embedding_path.exists():
                    stats["images_with_embeddings"] += 1
                
                # Get metadata for document and page info
                metadata = self.get_metadata(image_id)
                if metadata:
                    doc = metadata.get("document_name")
                    page = metadata.get("page_number")
                    if doc:
                        stats["documents"].add(doc)
                    if page:
                        stats["pages"].add(page)
                
                # Get file size
                image_path = self.image_dir / f"{image_id}.png"
                if image_path.exists():
                    stats["total_size_mb"] += image_path.stat().st_size / (1024 * 1024)
            
            # Convert sets to counts
            stats["unique_documents"] = len(stats["documents"])
            stats["unique_pages"] = len(stats["pages"])
            del stats["documents"]
            del stats["pages"]
            
            stats["total_size_mb"] = round(stats["total_size_mb"], 2)
            
            logger.debug(f"Image stats: {stats}")
            return stats
        
        except Exception as e:
            logger.error(f"Error getting image stats: {str(e)}")
            return {"error": str(e)}
    
    def clear_storage(self, confirm: bool = False) -> bool:
        """
        Clear all images, metadata, and embeddings.
        
        Args:
            confirm: Must be True to actually delete data (safety measure)
            
        Returns:
            True if successful
        """
        if not confirm:
            logger.warning("Storage clear requires confirm=True")
            return False
        
        try:
            import shutil
            
            for directory in [self.image_dir, self.metadata_dir, self.embeddings_dir]:
                if directory.exists():
                    shutil.rmtree(directory)
                    directory.mkdir(parents=True, exist_ok=True)
            
            logger.warning("Image storage cleared successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error clearing storage: {str(e)}")
            return False
    
    def validate_image(self, image_id: str) -> bool:
        """
        Validate that an image and its metadata exist.
        
        Args:
            image_id: Image identifier
            
        Returns:
            True if image is valid and complete
        """
        try:
            image_path = self.image_dir / f"{image_id}.png"
            metadata_path = self.metadata_dir / f"{image_id}.json"
            
            if not image_path.exists():
                logger.warning(f"Image file missing: {image_id}")
                return False
            
            if not metadata_path.exists():
                logger.warning(f"Metadata missing: {image_id}")
                return False
            
            # Try to load image to verify integrity
            try:
                img = Image.open(image_path)
                img.verify()
            except Exception as e:
                logger.warning(f"Image corruption detected: {image_id} - {str(e)}")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error validating image: {str(e)}")
            return False
