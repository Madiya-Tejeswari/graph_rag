"""
Knowledge graph builder.
Converts extracted PDF content into Neo4j graph structure.

Enhanced with Named Entity Recognition (NER) and Relation Extraction
to create entity-to-entity edges for a richer knowledge graph.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from storage.neo4j_client import Neo4jClient
from ingestion.ner_extractor import NERExtractor

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Builds knowledge graph from extracted document content."""
    
    def __init__(self, neo4j_client: Neo4jClient, spacy_model: str = "en_core_web_sm"):
        """
        Initialize graph builder.
        
        Args:
            neo4j_client: Neo4j client instance
            spacy_model: spaCy model name for NER
        """
        self.client = neo4j_client
        self.ner = NERExtractor(spacy_model=spacy_model)
        
        # Cache for entity node IDs to avoid duplicate creation
        # key: "normalized_name|entity_type" → neo4j node ID
        self._entity_cache: Dict[str, int] = {}
        
        # Track created relationships to avoid duplicates
        self._relationship_cache: Set[str] = set()
    
    def build(self, extracted_data: Dict[str, Any], document_name: str):
        """
        Build graph from extracted content.
        
        Args:
            extracted_data: Extracted content from PDF
            document_name: Name of the document
        """
        try:
            logger.info(f"Building graph for: {document_name}")
            
            # Reset caches for this document
            self._entity_cache.clear()
            self._relationship_cache.clear()
            
            # Create document node
            doc_id = self._create_document_node(document_name)
            
            # Process pages
            page_ids = self._process_pages(extracted_data.get("pages", []), doc_id, document_name)
            
            # Process text — now with NER + relation extraction
            self._process_text(extracted_data.get("text", []), doc_id, document_name)
            
            # Process tables
            self._process_tables(extracted_data.get("tables", []), doc_id, document_name)
            
            # Process figures
            self._process_figures(extracted_data.get("figures", []), doc_id, document_name)
            
            entity_count = len(self._entity_cache)
            rel_count = len(self._relationship_cache)
            logger.info(
                f"Graph building completed for: {document_name} "
                f"({entity_count} entities, {rel_count} relationships)"
            )
            
        except Exception as e:
            logger.error(f"Error building graph: {str(e)}", exc_info=True)
            raise
    
    def _create_document_node(self, document_name: str) -> int:
        """Create document node in graph."""
        try:
            properties = {
                "name": document_name,
                "type": "document"
            }
            doc_id = self.client.create_node("Document", properties)
            logger.debug(f"Created document node: {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Error creating document node: {str(e)}")
            raise
    
    def _process_pages(self, pages: List[Dict], doc_id: int, document_name: str) -> List[int]:
        """Process page nodes."""
        page_ids = []
        
        try:
            for page in pages:
                page_number = page.get("page_number")
                properties = {
                    "page_number": page_number,
                    "document_name": document_name
                }
                
                page_id = self.client.create_node("Page", properties)
                page_ids.append(page_id)
                
                # Create relationship from document to page
                self.client.create_relationship(doc_id, page_id, "HAS_PAGE")
            
            logger.debug(f"Created {len(page_ids)} page nodes")
            return page_ids
            
        except Exception as e:
            logger.error(f"Error processing pages: {str(e)}")
            return []
    
    def _process_text(self, text_blocks: List[Dict], doc_id: int, document_name: str):
        """
        Process text content with NER-based entity and relation extraction.
        
        For each text block:
        1. Store the TextChunk node (unchanged)
        2. Run NER to extract entities → create Entity nodes
        3. Run relation extraction → create Entity-to-Entity edges
        4. Link entities to their source TextChunk and Document
        """
        try:
            for block in text_blocks:
                page_number = block.get("page_number")
                content = block.get("content", "")
                
                # Skip very short blocks
                if len(content.strip()) < 20:
                    continue
                
                # ── Step 1: Store the text chunk ─────────────────────
                chunk_properties = {
                    "content": content[:2000],
                    "page_number": page_number,
                    "document_name": document_name,
                    "source_type": "text"
                }
                chunk_id = self.client.create_node("TextChunk", chunk_properties)
                if chunk_id is not None:
                    self.client.create_relationship(doc_id, chunk_id, "HAS_CHUNK")
                
                # ── Step 2: NER — extract entities ───────────────────
                entities = self.ner.extract_entities(content)
                
                entity_ids_in_chunk = []
                for ent in entities:
                    entity_id = self._get_or_create_entity(
                        name=ent["name"],
                        entity_type=ent["type"],
                        page_number=page_number,
                        document_name=document_name,
                        doc_id=doc_id,
                    )
                    if entity_id is not None:
                        entity_ids_in_chunk.append((entity_id, ent))
                        
                        # Link entity ← TextChunk
                        if chunk_id is not None:
                            self._create_relationship_once(
                                chunk_id, entity_id, "MENTIONS_ENTITY"
                            )
                
                # ── Step 3: Relation extraction — entity-to-entity ───
                if len(entity_ids_in_chunk) >= 2:
                    relations = self.ner.extract_relations(content, entities)
                    
                    for rel in relations:
                        source_key = f"{rel['source'].lower()}|{rel['source_type']}"
                        target_key = f"{rel['target'].lower()}|{rel['target_type']}"
                        
                        source_id = self._entity_cache.get(source_key)
                        target_id = self._entity_cache.get(target_key)
                        
                        if source_id is not None and target_id is not None and source_id != target_id:
                            rel_props = {
                                "evidence": rel.get("evidence", "")[:300],
                                "confidence": rel.get("confidence", 0.5),
                                "page_number": page_number,
                                "document_name": document_name,
                            }
                            self._create_relationship_once(
                                source_id, target_id, rel["type"],
                                properties=rel_props,
                            )
            
            logger.info(
                f"NER extracted {len(self._entity_cache)} unique entities, "
                f"{len(self._relationship_cache)} relationships"
            )
            
        except Exception as e:
            logger.error(f"Error processing text: {str(e)}", exc_info=True)
    
    def _process_tables(self, tables: List[Dict], doc_id: int, document_name: str):
        """Process table content with NER on table text."""
        try:
            for table in tables:
                page_number = table.get("page_number")
                table_name = table.get("name", "")
                content = table.get("content", "")
                
                properties = {
                    "name": table_name,
                    "caption": table.get("caption", ""),
                    "page_number": page_number,
                    "document_name": document_name,
                    "content": content[:500],  # Truncate for storage
                    "markdown": table.get("markdown", "")[:1000],
                    "source_type": "table"
                }
                
                table_id = self.client.create_node("Table", properties)
                
                # Create relationship
                self.client.create_relationship(doc_id, table_id, "HAS_TABLE")
                
                # Extract facts from table
                self._extract_table_facts(table_id, content, page_number, document_name)
                
                # ── NER on table content ─────────────────────────────
                if content and len(content.strip()) > 20:
                    entities = self.ner.extract_entities(content)
                    for ent in entities:
                        entity_id = self._get_or_create_entity(
                            name=ent["name"],
                            entity_type=ent["type"],
                            page_number=page_number,
                            document_name=document_name,
                            doc_id=doc_id,
                        )
                        if entity_id is not None and table_id is not None:
                            self._create_relationship_once(
                                table_id, entity_id, "TABLE_MENTIONS"
                            )
            
            logger.debug(f"Processed {len(tables)} tables")
            
        except Exception as e:
            logger.error(f"Error processing tables: {str(e)}")
    
    def _process_figures(self, figures: List[Dict], doc_id: int, document_name: str):
        """Process figure content with NER on descriptions."""
        try:
            for figure in figures:
                page_number = figure.get("page_number")
                figure_name = figure.get("name", "")
                description = figure.get("description", "")
                
                properties = {
                    "name": figure_name,
                    "page_number": page_number,
                    "document_name": document_name,
                    "description": description[:500],
                    "source_type": "figure"
                }
                
                figure_id = self.client.create_node("Figure", properties)
                
                # Create relationship
                self.client.create_relationship(doc_id, figure_id, "HAS_FIGURE")
                
                # ── NER on figure description ────────────────────────
                if description and len(description.strip()) > 20:
                    entities = self.ner.extract_entities(description)
                    for ent in entities:
                        entity_id = self._get_or_create_entity(
                            name=ent["name"],
                            entity_type=ent["type"],
                            page_number=page_number,
                            document_name=document_name,
                            doc_id=doc_id,
                        )
                        if entity_id is not None and figure_id is not None:
                            self._create_relationship_once(
                                figure_id, entity_id, "FIGURE_MENTIONS"
                            )
                    
                    # Extract relations between entities in figure descriptions
                    if len(entities) >= 2:
                        relations = self.ner.extract_relations(description, entities)
                        for rel in relations:
                            source_key = f"{rel['source'].lower()}|{rel['source_type']}"
                            target_key = f"{rel['target'].lower()}|{rel['target_type']}"
                            source_id = self._entity_cache.get(source_key)
                            target_id = self._entity_cache.get(target_key)
                            if source_id and target_id and source_id != target_id:
                                self._create_relationship_once(
                                    source_id, target_id, rel["type"],
                                    properties={
                                        "evidence": rel.get("evidence", "")[:300],
                                        "confidence": rel.get("confidence", 0.5),
                                        "page_number": page_number,
                                        "document_name": document_name,
                                    }
                                )
                
                # Legacy: extract figure component facts
                self._extract_figure_facts(figure_id, description, page_number, document_name)
            
            logger.debug(f"Processed {len(figures)} figures")
            
        except Exception as e:
            logger.error(f"Error processing figures: {str(e)}")

    # ── Entity deduplication helper ──────────────────────────────────────

    def _get_or_create_entity(
        self,
        name: str,
        entity_type: str,
        page_number: int,
        document_name: str,
        doc_id: int,
    ) -> Optional[int]:
        """
        Get existing entity node ID from cache, or create a new one.
        Ensures we don't duplicate entity nodes across text blocks.
        
        Returns:
            Neo4j node ID, or None on failure
        """
        cache_key = f"{name.lower()}|{entity_type}"
        
        if cache_key in self._entity_cache:
            return self._entity_cache[cache_key]
        
        properties = {
            "name": name,
            "type": entity_type,
            "page_number": page_number,
            "document_name": document_name,
        }
        entity_id = self.client.create_node("Entity", properties)
        
        if entity_id is not None:
            self._entity_cache[cache_key] = entity_id
            # Link to document
            self.client.create_relationship(doc_id, entity_id, "MENTIONS")
        
        return entity_id

    def _create_relationship_once(
        self,
        source_id: int,
        target_id: int,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Create a relationship only if it hasn't been created before.
        Uses an in-memory cache keyed by (source, target, type).
        """
        cache_key = f"{source_id}|{target_id}|{rel_type}"
        if cache_key in self._relationship_cache:
            return False
        
        self._relationship_cache.add(cache_key)
        return self.client.create_relationship(
            source_id, target_id, rel_type, properties
        )

    # ── Legacy extraction methods (kept for backward compat) ─────────────

    def _extract_entities(self, text: str) -> List[tuple]:
        """
        Extract entities from text.
        Simple rule-based extraction for utility terminology.
        
        Returns:
            List of (entity_text, entity_type) tuples
        """
        entities = []
        
        # Utility terms to look for
        equipment_terms = {
            "transformer": "Equipment",
            "conduit": "Equipment",
            "meter": "Equipment",
            "socket": "Equipment",
            "lateral": "Component",
            "service": "Service",
            "single phase": "ServiceType",
            "three phase": "ServiceType",
            "underground": "InstallationType",
            "overhead": "InstallationType"
        }
        
        text_lower = text.lower()
        for term, entity_type in equipment_terms.items():
            if term in text_lower:
                entities.append((term, entity_type))
        
        # Extract numbers with units
        import re
        voltage_pattern = r"(\d+\s*(?:kV|V))"
        amperage_pattern = r"(\d+\s*(?:kVA|A|Amp))"
        
        for match in re.finditer(voltage_pattern, text):
            entities.append((match.group(1), "Voltage"))
        
        for match in re.finditer(amperage_pattern, text):
            entities.append((match.group(1), "Capacity"))
        
        return entities
    
    def _extract_table_facts(self, table_id: int, content: str, page_number: int, 
                            document_name: str):
        """Extract facts from table."""
        try:
            # Simple extraction: look for key-value patterns
            lines = content.split("\n")
            for line in lines[:10]:  # Process first 10 lines
                if "|" in line or "\t" in line:
                    # Potential fact
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 2:
                        # Create fact node
                        fact_text = " ".join(parts[:2])
                        if fact_text:
                            properties = {
                                "fact": fact_text,
                                "page_number": page_number,
                                "document_name": document_name,
                                "source_type": "table"
                            }
                            fact_id = self.client.create_node("Fact", properties)
                            self.client.create_relationship(table_id, fact_id, "CONTAINS")
        except Exception as e:
            logger.debug(f"Error extracting table facts: {str(e)}")
    
    def _extract_figure_facts(self, figure_id: int, description: str, page_number: int, 
                             document_name: str):
        """Extract facts from figure."""
        try:
            # Extract components mentioned in figure description
            components = self._extract_entities(description)
            
            for component_text, component_type in components:
                properties = {
                    "name": component_text,
                    "type": component_type,
                    "page_number": page_number,
                    "document_name": document_name,
                    "source_type": "figure"
                }
                component_id = self.client.create_node("Component", properties)
                self.client.create_relationship(figure_id, component_id, "CONTAINS")
                
        except Exception as e:
            logger.debug(f"Error extracting figure facts: {str(e)}")
