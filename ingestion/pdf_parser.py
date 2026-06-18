"""
PDF parser using PyMuPDF (fitz).
Extracts text, tables, figures, and images from PDFs.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
from io import BytesIO

logger = logging.getLogger(__name__)


class PDFParser:
    """Parses PDFs using PyMuPDF."""
    
    def __init__(self):
        """Initialize PDF parser."""
        try:
            import fitz  # PyMuPDF
            self.fitz = fitz
            logger.info("PyMuPDF initialized")
        except ImportError as e:
            logger.error(f"PyMuPDF not installed: {str(e)}")
            logger.error("Install with: pip install pymupdf")
            raise
    
    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse PDF and extract all content.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with extracted content
        """
        try:
            pdf_path = Path(pdf_path)
            logger.info(f"Parsing PDF: {pdf_path}")
            
            if not pdf_path.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return self._empty_extraction()
            
            doc = self.fitz.open(str(pdf_path))
            
            # Extract content
            extracted = {
                "document_name": pdf_path.name,
                "file_path": str(pdf_path),
                "text": self._extract_text(doc),
                "tables": self._extract_tables(doc),
                "figures": self._extract_figures(doc),
                "images": self._extract_images(doc),
                "pages": self._extract_pages(doc)
            }
            
            logger.info(f"Total extracted: {len(extracted['text'])} text blocks, "
                       f"{len(extracted['tables'])} tables, "
                       f"{len(extracted['figures'])} figures, "
                       f"{len(extracted['images'])} images from {len(extracted['pages'])} pages")
            
            doc.close()
            return extracted
            
        except Exception as e:
            logger.error(f"Error parsing PDF {pdf_path}: {str(e)}", exc_info=True)
            return self._empty_extraction()
    
    def _empty_extraction(self) -> Dict[str, Any]:
        """Return empty extraction structure."""
        return {
            "document_name": "",
            "file_path": "",
            "text": [],
            "tables": [],
            "figures": [],
            "images": [],
            "pages": []
        }
    
    def _extract_text(self, doc) -> List[Dict]:
        """Extract text content from document using block-level extraction."""
        text_blocks = []
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                # Extract text blocks: (x0, y0, x1, y1, "text", block_no, block_type)
                # block_type 0 = text, 1 = image
                blocks = page.get_text("blocks")
                
                for block in blocks:
                    # Only process text blocks (type 0)
                    if block[6] == 0:
                        text_content = block[4].strip()
                        if len(text_content) > 0:
                            text_blocks.append({
                                "page_number": page_num + 1,
                                "content": text_content,
                                "type": "text",
                                "block_type": "textblock"
                            })
                
                logger.debug(f"Extracted {len([b for b in text_blocks if b['page_number'] == page_num + 1])} "
                           f"text blocks from page {page_num + 1}")
        
        except Exception as e:
            logger.warning(f"Error extracting text: {str(e)}")
        
        return text_blocks
    
    def _extract_table_captions_from_page(self, page) -> List[Dict]:
        """
        Extract table captions from page text.
        Looks for patterns like:
          - "Table 2-1. Minimum Separation Requirements"
          - "Table 1-1: Terminology and Definitions"
          - "TABLE 3-2 Conduit Requirements"
        
        Returns:
            List of caption dicts with keys: caption, name
        """
        import re
        captions = []
        
        blocks = page.get_text("blocks")
        for block in blocks:
            if block[6] != 0:  # Only text blocks
                continue
            
            text = block[4].strip()
            
            # Match table caption patterns
            patterns = [
                # "Table 2-1. Description" or "Table 2-1: Description"
                r'((?:Table|TABLE)\s*\d+[-.\s]*\d*\s*[.:—–\-]?\s*.+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    caption_text = match.group(1).strip()
                    # Take first line only
                    caption_text = caption_text.split('\n')[0].strip()
                    if len(caption_text) > 300:
                        caption_text = caption_text[:300]
                    
                    # Extract the short name (e.g., "Table 2-1")
                    name_match = re.match(
                        r'((?:Table|TABLE)\s*\d+[-.\s]*\d*)',
                        caption_text, re.IGNORECASE
                    )
                    short_name = name_match.group(1).strip() if name_match else caption_text[:30]
                    
                    captions.append({
                        "caption": caption_text,
                        "name": short_name,
                    })
                    break
        
        return captions
    
    def _extract_tables(self, doc) -> List[Dict]:
        """Extract tables from document with real table names from page text."""
        tables = []
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # PyMuPDF 1.23+ has built-in table detection
                try:
                    page_tables = page.find_tables()
                except AttributeError:
                    logger.debug("Table detection not available in this PyMuPDF version")
                    continue
                
                # Extract real table names from page text
                page_captions = self._extract_table_captions_from_page(page)
                caption_idx = 0
                
                for tab in page_tables:
                    try:
                        # Extract table data as a list of lists
                        table_data = tab.extract()
                        
                        if not table_data:
                            continue
                        
                        # Convert to markdown format
                        markdown = self._table_to_markdown(table_data)
                        # Convert to plain text
                        text = self._table_to_text(table_data)
                        
                        # Use real table name if available
                        if caption_idx < len(page_captions):
                            cap = page_captions[caption_idx]
                            table_name = cap["name"]
                            table_caption = cap["caption"]
                            caption_idx += 1
                        else:
                            table_name = f"Table on page {page_num + 1}"
                            table_caption = ""
                        
                        tables.append({
                            "page_number": page_num + 1,
                            "name": table_name,
                            "caption": table_caption,
                            "content": text,
                            "html": None,
                            "markdown": markdown,
                            "type": "table",
                            "block_type": "table"
                        })
                        logger.debug(f"Extracted table '{table_name}' from page {page_num + 1}")
                    except Exception as e:
                        logger.debug(f"Error extracting individual table: {str(e)}")
        
        except Exception as e:
            logger.warning(f"Error extracting tables: {str(e)}")
        
        return tables
    
    def _extract_figure_captions_from_page(self, page) -> List[Dict]:
        """
        Extract figure/diagram captions from page text.
        Looks for patterns like:
          - "Figure 2-1: Description..."
          - "Figure 2.1 Description..."
          - "FIGURE 3-4 — Description..."
          - "Fig. 5 Description..."
          - "Diagram 1: Description..."
        
        Returns:
            List of caption dicts with keys: caption, name, y_position
        """
        import re
        captions = []
        
        blocks = page.get_text("blocks")
        for block in blocks:
            if block[6] != 0:  # Only text blocks
                continue
            
            text = block[4].strip()
            y_pos = block[1]  # y0 coordinate (vertical position)
            
            # Match various figure caption patterns
            patterns = [
                # "Figure 2-1: Description" or "Figure 2-1 — Description"
                r'((?:Figure|FIGURE|Fig\.?)\s*\d+[-.\s]*\d*\s*[:—–\-]?\s*.+)',
                # "Diagram 1: Description"
                r'((?:Diagram|DIAGRAM)\s*\d+[-.\s]*\d*\s*[:—–\-]?\s*.+)',
                # "Drawing No. 123"
                r'((?:Drawing|DRAWING)\s*(?:No\.?\s*)?\d+[-.\s]*\d*\s*[:—–\-]?\s*.+)',
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    caption_text = match.group(1).strip()
                    # Clean up: take first line only (captions are usually single line)
                    caption_text = caption_text.split('\n')[0].strip()
                    # Limit length
                    if len(caption_text) > 300:
                        caption_text = caption_text[:300]
                    
                    # Extract the short name (e.g., "Figure 2-1")
                    name_match = re.match(
                        r'((?:Figure|FIGURE|Fig\.?|Diagram|DIAGRAM|Drawing|DRAWING)'
                        r'\s*(?:No\.?\s*)?\d+[-.\s]*\d*)',
                        caption_text, re.IGNORECASE
                    )
                    short_name = name_match.group(1).strip() if name_match else caption_text[:30]
                    
                    captions.append({
                        "caption": caption_text,
                        "name": short_name,
                        "y_position": y_pos,
                    })
                    break  # One caption per text block
        
        return captions
    
    def _extract_figures(self, doc) -> List[Dict]:
        """Extract figures and diagrams from document (image blocks with captions)."""
        figures = []
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                
                # Extract captions from this page's text
                page_captions = self._extract_figure_captions_from_page(page)
                caption_idx = 0  # Track which caption to assign
                
                for img_idx, img_info in enumerate(image_list):
                    # img_info: (xref, smask, width, height, bpc, colorspace, alt, name, filter, referencer)
                    xref = img_info[0]
                    width = img_info[2]
                    height = img_info[3]
                    
                    # Consider larger images as figures (likely charts, diagrams, etc.)
                    if width > 100 and height > 100:
                        # Try to match a caption from the page
                        if caption_idx < len(page_captions):
                            cap = page_captions[caption_idx]
                            name = cap["name"]
                            description = cap["caption"]
                            caption_idx += 1
                        else:
                            # Fallback: use image metadata or generic name
                            name = img_info[7] if img_info[7] else f"Figure on page {page_num + 1}"
                            description = f"Figure on page {page_num + 1} ({width}x{height})"
                        
                        figures.append({
                            "page_number": page_num + 1,
                            "name": name,
                            "description": description,
                            "type": "figure",
                            "block_type": "picture",
                            "image_index": img_idx,  # Link to image extraction
                        })
                        logger.debug(f"Extracted figure '{name}' from page {page_num + 1}")
        
        except Exception as e:
            logger.warning(f"Error extracting figures: {str(e)}")
        
        return figures
    
    def _extract_images(self, doc) -> List[Dict]:
        """Extract images from document as raw bytes, with figure captions attached."""
        images = []
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)
                
                # Extract captions from page text to attach to images
                page_captions = self._extract_figure_captions_from_page(page)
                figure_caption_idx = 0  # Track caption assignment for large images
                
                for img_idx, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        
                        # Extract image bytes
                        base_image = doc.extract_image(xref)
                        if base_image is None:
                            continue
                        
                        image_data = base_image["image"]  # raw bytes
                        image_format = base_image.get("ext", "png")
                        width = base_image.get("width", 0)
                        height = base_image.get("height", 0)
                        
                        # Skip very small images (icons, bullets, etc.)
                        if width < 50 or height < 50:
                            continue
                        
                        # Determine caption for this image
                        caption = ""
                        figure_name = ""
                        is_figure = width > 100 and height > 100
                        
                        if is_figure and figure_caption_idx < len(page_captions):
                            cap = page_captions[figure_caption_idx]
                            caption = cap["caption"]
                            figure_name = cap["name"]
                            figure_caption_idx += 1
                        
                        images.append({
                            "page_number": page_num + 1,
                            "image_id": f"img_{page_num + 1}_{img_idx}",
                            "data": image_data,
                            "type": "image",
                            "format": image_format,
                            "block_type": "picture",
                            "caption": caption,
                            "figure_name": figure_name,
                            "width": width,
                            "height": height,
                        })
                        logger.debug(f"Extracted image from page {page_num + 1}"
                                    f"{' (' + figure_name + ')' if figure_name else ''}")
                    except Exception as e:
                        logger.debug(f"Error extracting image: {str(e)}")
        
        except Exception as e:
            logger.warning(f"Error extracting images: {str(e)}")
        
        return images
    
    def _extract_pages(self, doc) -> List[Dict]:
        """Extract page information."""
        pages = []
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                rect = page.rect
                pages.append({
                    "page_number": page_num + 1,
                    "metadata": {
                        "width": rect.width,
                        "height": rect.height
                    }
                })
        except Exception as e:
            logger.warning(f"Error extracting page info: {str(e)}")
        
        return pages
    
    def _table_to_markdown(self, table_data: List[List]) -> str:
        """Convert table data (list of lists) to markdown format."""
        if not table_data:
            return ""
        
        lines = []
        # Header row
        header = [str(cell) if cell else "" for cell in table_data[0]]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        
        # Data rows
        for row in table_data[1:]:
            cells = [str(cell) if cell else "" for cell in row]
            # Pad row if needed
            while len(cells) < len(header):
                cells.append("")
            lines.append("| " + " | ".join(cells[:len(header)]) + " |")
        
        return "\n".join(lines)
    
    def _table_to_text(self, table_data: List[List]) -> str:
        """Convert table data (list of lists) to plain text."""
        if not table_data:
            return ""
        
        text_parts = []
        for row in table_data:
            row_text = " | ".join(str(cell) if cell else "" for cell in row)
            text_parts.append(row_text)
        
        return "\n".join(text_parts)
