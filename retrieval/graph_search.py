"""
Graph search engine for entity and relationship retrieval.
Primary retrieval mechanism for text, tables, and figure facts.
"""

import logging
from typing import List, Dict, Any, Optional
from storage.neo4j_client import Neo4jClient

logger = logging.getLogger(__name__)


class GraphSearchEngine:
    """Search engine for graph-based retrieval."""
    
    def __init__(self, neo4j_client: Neo4jClient):
        """
        Initialize graph search engine.
        
        Args:
            neo4j_client: Neo4j database client
        """
        self.client = neo4j_client
    
    def search(self, query: str, top_k: int = 10, expansion_hops: int = 3,
               search_tables: bool = True, search_figures: bool = True) -> List[Dict]:
        """
        Search for entities, text chunks, and relationships matching the query.
        
        Pipeline:
        1. Full-text search on stored text chunks (primary)
        2. Entity extraction from query
        3. Graph search
        4. Table and figure search (conditional)
        
        Args:
            query: User query
            top_k: Number of top results
            expansion_hops: Number of relationship hops for expansion
            search_tables: Whether to search for table nodes
            search_figures: Whether to search for figure nodes
            
        Returns:
            List of retrieved facts with provenance
        """
        try:
            logger.info(f"Starting graph search for: {query[:100]}")
            
            results = []
            
            # Primary: Search text chunks for actual document content
            text_results = self._search_text_chunks(query, top_k)
            results.extend(text_results)
            
            # Extract entities from query
            entities = self._extract_entities_from_query(query)
            logger.debug(f"Extracted entities: {entities}")
            
            # Search for matching entities in graph
            entity_results = self._search_entities(entities, top_k=3)
            logger.debug(f"Found {len(entity_results)} entities")
            results.extend(entity_results)
            
            # Expand entity results via NER-created relationships (limited)
            if entity_results and expansion_hops > 0:
                rel_results = self._search_entity_relations(
                    entities, hops=min(expansion_hops, 2), top_k=5
                )
                results.extend(rel_results)
                logger.info(f"Entity-relation expansion returned {len(rel_results)} results")
            
            # Conditionally search for tables
            if search_tables:
                table_results = self._search_tables(query, top_k=3)
                results.extend(table_results)
                logger.info(f"Keyword table search returned {len(table_results)} results")
                
                # Co-located table search
                text_pages = list(set(
                    r.get("provenance", {}).get("page_number")
                    for r in text_results
                    if r.get("provenance", {}).get("page_number") is not None
                ))
                colocated_tables = self._search_colocated_tables_figures(text_pages, top_k=3)
                # Only add co-located tables, not figures
                for c in colocated_tables:
                    if c.get("type") == "table":
                        results.append(c)
            
            # Conditionally search for figures
            if search_figures:
                figure_results = self._search_figures(query, top_k=3)
                results.extend(figure_results)
                logger.info(f"Keyword figure search returned {len(figure_results)} results")
            
            # Deduplicate by content
            seen = set()
            unique_results = []
            for r in results:
                key = r.get("content", "")[:100]
                if key and key not in seen:
                    seen.add(key)
                    unique_results.append(r)
            
            # Sort by confidence to keep only the best evidence
            unique_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)
            
            logger.info(f"Graph search returned {len(unique_results)} results (tables={search_tables}, figures={search_figures})")
            # Limit to top 10 most relevant results
            return unique_results[:10]
            
        except Exception as e:
            logger.error(f"Error in graph search: {str(e)}")
            return []
    
    def _search_text_chunks(self, query: str, top_k: int) -> List[Dict]:
        """
        Search text chunks stored in the graph for actual document content.
        Uses keyword matching with relevance scoring.
        
        Args:
            query: User query
            top_k: Number of results
            
        Returns:
            List of matching text chunks with provenance
        """
        results = []
        
        # Extract meaningful keywords from query
        stop_words = {"what", "is", "the", "of", "a", "an", "for", "and", "or", "to", "in",
                     "how", "why", "when", "where", "which", "are", "can", "do", "does",
                     "this", "that", "these", "those", "be", "been", "being", "have", "has",
                     "it", "its", "they", "their", "we", "our", "you", "your", "like",
                     "look", "looks", "does", "typical", "about", "tell", "me", "show"}
        words = query.split()
        keywords = [w.strip("?.,;:").lower() for w in words 
                    if w.strip("?.,;:").lower() not in stop_words and len(w.strip("?.,;:")) > 2]
        
        # Generate bigrams (e.g., "single phase", "underground service")
        clean_words = [w.strip("?.,;:").lower() for w in words]
        bigrams = []
        for i in range(len(clean_words) - 1):
            if clean_words[i] not in stop_words and clean_words[i+1] not in stop_words:
                bigrams.append(f"{clean_words[i]} {clean_words[i+1]}")
                bigrams.append(f"{clean_words[i]}-{clean_words[i+1]}")
        
        # Trigrams for very specific phrases
        trigrams = []
        for i in range(len(clean_words) - 2):
            non_stop = [w for w in clean_words[i:i+3] if w not in stop_words]
            if len(non_stop) >= 2:
                trigrams.append(" ".join(clean_words[i:i+3]))
        
        all_keywords = list(dict.fromkeys(trigrams + bigrams + keywords))
        
        if not keywords:
            return results
        
        try:
            all_chunks = []
            
            # Pass 1: Bigram/trigram AND match (highest relevance)
            if bigrams:
                bigram_conditions = " AND ".join(
                    [f"toLower(tc.content) CONTAINS '{bg}'" for bg in bigrams[:2]]
                )
                cypher_bigram = f"""
                MATCH (tc:TextChunk)
                WHERE {bigram_conditions}
                RETURN tc.content as content, tc.page_number as page_number,
                       tc.document_name as document_name
                LIMIT $limit
                """
                bigram_chunks = self.client.execute_query(cypher_bigram, {"limit": top_k * 3})
                all_chunks.extend(bigram_chunks)
            
            # Pass 2: AND query with top 3 single-word keywords
            single_word_kws = [kw for kw in keywords if ' ' not in kw and '-' not in kw]
            and_kws = sorted(single_word_kws, key=len, reverse=True)[:3]
            if len(and_kws) >= 2 and len(all_chunks) < top_k * 2:
                and_conditions = " AND ".join([f"toLower(tc.content) CONTAINS '{kw}'" for kw in and_kws])
                cypher_and = f"""
                MATCH (tc:TextChunk)
                WHERE {and_conditions}
                RETURN tc.content as content, tc.page_number as page_number,
                       tc.document_name as document_name
                LIMIT $limit
                """
                and_chunks = self.client.execute_query(cypher_and, {"limit": top_k * 3})
                all_chunks.extend(and_chunks)
            
            # Pass 3: Broader OR query if we don't have enough results
            if len(all_chunks) < top_k:
                or_conditions = " OR ".join([f"toLower(tc.content) CONTAINS '{kw}'" for kw in all_keywords[:8]])
                cypher_or = f"""
                MATCH (tc:TextChunk)
                WHERE {or_conditions}
                RETURN tc.content as content, tc.page_number as page_number,
                       tc.document_name as document_name
                LIMIT $limit
                """
                or_chunks = self.client.execute_query(cypher_or, {"limit": top_k * 3})
                all_chunks.extend(or_chunks)
            
            # Deduplicate by content
            seen_content = set()
            unique_chunks = []
            for chunk in all_chunks:
                content_key = chunk.get("content", "")[:100]
                if content_key not in seen_content:
                    seen_content.add(content_key)
                    unique_chunks.append(chunk)
            
            # Score each chunk with weighted keyword relevance
            for chunk in unique_chunks:
                content = chunk.get("content", "")
                content_lower = content.lower()
                
                score = 0.0
                
                # Trigram matches (highest weight — very specific)
                for tg in trigrams:
                    if tg in content_lower:
                        score += 0.5
                
                # Bigram matches (high weight — phrase-level relevance)
                for bg in bigrams:
                    if bg in content_lower:
                        score += 0.35
                
                # Single keyword matches (base weight)
                matched_kws = [kw for kw in keywords if kw in content_lower]
                score += len(matched_kws) * 0.15
                
                # Bonus: fraction of all query keywords that matched
                if keywords:
                    coverage = len(matched_kws) / len(keywords)
                    score += coverage * 0.3
                
                results.append({
                    "type": "text_chunk",
                    "content": content,
                    "provenance": {
                        "page_number": chunk.get("page_number"),
                        "document_name": chunk.get("document_name"),
                        "source_type": "text"
                    },
                    "confidence": min(0.98, score)
                })
            
            # Sort by confidence (best matches first)
            results.sort(key=lambda x: x["confidence"], reverse=True)
            
        except Exception as e:
            logger.warning(f"Error searching text chunks: {str(e)}")
        
        # Return only the top 5 most relevant chunks
        return results[:min(top_k, 5)]
    
    def _extract_entities_from_query(self, query: str) -> List[str]:
        """
        Extract entity keywords from query.
        
        Simple keyword extraction. In production, use NER.
        
        Args:
            query: User query
            
        Returns:
            List of entity keywords
        """
        # Simple heuristic: capitalize words and common entities
        keywords = []
        
        # Common utility terms to search for
        utility_terms = [
            "transformer", "kVA", "single phase", "three phase",
            "service", "underground", "overhead", "conduit",
            "meter", "socket", "lateral", "equipment",
            "voltage", "amperage", "current", "load"
        ]
        
        query_lower = query.lower()
        for term in utility_terms:
            if term.lower() in query_lower:
                keywords.append(term)
        
        # Also extract individual significant words
        stop_words = {"what", "is", "the", "of", "a", "an", "for", "and", "or", "to", "in"}
        words = query.split()
        for word in words:
            clean_word = word.lower().strip("?.,;:")
            if clean_word not in stop_words and len(clean_word) > 2:
                keywords.append(clean_word)
        
        return list(set(keywords))[:10]
    
    def _search_entities(self, keywords: List[str], top_k: int) -> List[Dict]:
        """
        Search for entities matching keywords.
        
        Args:
            keywords: Search keywords
            top_k: Number of results
            
        Returns:
            List of matching entities with provenance
        """
        results = []
        
        for keyword in keywords:
            try:
                entities = self.client.search_entities(keyword, top_k=top_k)
                for entity in entities:
                    results.append({
                        "type": "entity",
                        "content": entity.get("name", ""),
                        "entity_type": entity.get("type", ""),
                        "provenance": {
                            "page_number": entity.get("page_number"),
                            "document_name": entity.get("document_name"),
                            "source_type": "entity"
                        },
                        "confidence": 0.8
                    })
            except Exception as e:
                logger.debug(f"Error searching for keyword {keyword}: {str(e)}")
        
        return results
    
    def _expand_relationships(self, entity_results: List[Dict], hops: int) -> List[Dict]:
        """
        Expand from entities to related nodes and relationships.
        
        Args:
            entity_results: Initial entity results
            hops: Number of relationship hops
            
        Returns:
            List of related facts
        """
        results = []
        
        # NER relationship types to traverse
        ner_rel_types = [
            "REQUIRES", "CONNECTS_TO", "HAS_SPECIFICATION",
            "INSTALLED_IN", "USED_FOR", "PART_OF", "REGULATED_BY",
            "SUPERSEDES", "CO_OCCURS_WITH", "RELATED_TO",
            "MENTIONS_ENTITY", "TABLE_MENTIONS", "FIGURE_MENTIONS",
        ]
        
        try:
            for rel_type in ner_rel_types:
                related = self.client.find_by_relationship(rel_type, limit=20)
                for rel in related:
                    source_data = rel.get("source", {})
                    target_data = rel.get("target", {})
                    
                    # Build a readable fact from the relationship
                    source_name = source_data.get("name", "")
                    target_name = target_data.get("name", "")
                    if source_name and target_name:
                        fact_content = f"{source_name} [{rel_type}] {target_name}"
                    else:
                        fact_content = str(source_data) + " -> " + str(target_data)
                    
                    results.append({
                        "type": "relationship",
                        "content": fact_content,
                        "source": source_data,
                        "target": target_data,
                        "relationship": rel_type,
                        "provenance": {
                            "page_number": rel.get("source_page"),
                            "document_name": rel.get("source_document"),
                            "source_type": "entity_relation"
                        },
                        "confidence": 0.7
                    })
        except Exception as e:
            logger.debug(f"Error expanding relationships: {str(e)}")
        
        return results
    
    def _search_entity_relations(self, keywords: List[str], hops: int = 2,
                                  top_k: int = 10) -> List[Dict]:
        """
        Search entity-to-entity relationships created by NER extraction.
        
        Finds entities matching query keywords, then traverses outgoing
        relationships up to `hops` levels deep to gather connected context.
        
        Args:
            keywords: Query keywords
            hops: Max relationship hops (1-3)
            top_k: Max results to return
            
        Returns:
            List of relationship-derived facts with provenance
        """
        results = []
        hops = min(hops, 3)  # Cap at 3 to avoid explosion
        
        if not keywords:
            return results
        
        try:
            # Build OR conditions for keyword matching
            or_conditions = " OR ".join(
                [f"toLower(e.name) CONTAINS '{kw.lower()}'" for kw in keywords[:8]]
            )
            
            # Variable-length path query for multi-hop traversal
            cypher_query = f"""
            MATCH (e:Entity)
            WHERE {or_conditions}
            MATCH (e)-[r*1..{hops}]-(related:Entity)
            WITH e, related, r,
                 [rel in r | type(rel)] AS rel_types,
                 [rel in r | rel.evidence] AS evidences,
                 [rel in r | rel.confidence] AS confidences
            RETURN e.name AS source_name, e.type AS source_type,
                   related.name AS target_name, related.type AS target_type,
                   rel_types, evidences, confidences,
                   related.page_number AS page_number,
                   related.document_name AS document_name
            LIMIT $limit
            """
            
            records = self.client.execute_query(cypher_query, {"limit": top_k * 3})
            
            seen = set()
            for record in records:
                source = record.get("source_name", "")
                target = record.get("target_name", "")
                rel_types = record.get("rel_types", [])
                evidences = record.get("evidences", [])
                confidences = record.get("confidences", [])
                
                if not source or not target:
                    continue
                
                dedup_key = f"{source}|{target}"
                if dedup_key in seen:
                    continue
                seen.add(dedup_key)
                
                # Build readable relationship chain
                rel_chain = " → ".join(
                    [f"[{rt}]" for rt in rel_types if rt]
                ) if rel_types else "[RELATED_TO]"
                
                content = f"{source} {rel_chain} {target}"
                
                # Use evidence from relationships if available
                evidence_text = next(
                    (e for e in (evidences or []) if e), ""
                )
                if evidence_text:
                    content = f"{content}\nEvidence: {evidence_text}"
                
                avg_confidence = 0.6
                valid_confidences = [c for c in (confidences or []) if c is not None]
                if valid_confidences:
                    avg_confidence = sum(valid_confidences) / len(valid_confidences)
                
                results.append({
                    "type": "entity_relation",
                    "content": content,
                    "source_entity": source,
                    "target_entity": target,
                    "relationship_chain": rel_types,
                    "provenance": {
                        "page_number": record.get("page_number"),
                        "document_name": record.get("document_name"),
                        "source_type": "entity_relation"
                    },
                    "confidence": avg_confidence
                })
            
            # Sort by confidence
            results.sort(key=lambda x: x["confidence"], reverse=True)
            
        except Exception as e:
            logger.debug(f"Error searching entity relations: {str(e)}")
        
        return results[:top_k]
    
    def _search_tables(self, query: str, top_k: int) -> List[Dict]:
        """
        Search for table-derived facts using keyword matching.
        
        Searches both table names and table content for keyword matches.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of table facts
        """
        results = []
        
        # Extract keywords from query
        stop_words = {"what", "is", "the", "of", "a", "an", "for", "and", "or", "to", "in",
                     "how", "why", "when", "where", "which", "are", "can", "do", "does",
                     "this", "that", "these", "those", "be", "been", "being", "have", "has",
                     "it", "its", "they", "their", "we", "our", "you", "your", "show", "me"}
        words = query.split()
        keywords = [w.strip("?.,;:").lower() for w in words 
                    if w.strip("?.,;:").lower() not in stop_words and len(w.strip("?.,;:")) > 2]
        
        if not keywords:
            return results
        
        try:
            # Search tables by keyword matches in content or name
            or_conditions = " OR ".join(
                [f"toLower(t.content) CONTAINS '{kw}'" for kw in keywords[:6]] +
                [f"toLower(t.name) CONTAINS '{kw}'" for kw in keywords[:6]]
            )
            
            cypher_query = f"""
            MATCH (t:Table)
            WHERE {or_conditions}
            RETURN t.name as name, t.page_number as page_number, 
                   t.document_name as document_name, t.content as content
            LIMIT $limit
            """
            tables = self.client.execute_query(cypher_query, {"limit": top_k * 2})
            
            for table in tables:
                content = table.get("content", "")
                content_lower = content.lower()
                match_count = sum(1 for kw in keywords if kw in content_lower)
                
                results.append({
                    "type": "table",
                    "content": content,
                    "table_name": table.get("name", ""),
                    "provenance": {
                        "page_number": table.get("page_number"),
                        "document_name": table.get("document_name"),
                        "source_type": "table",
                        "source_name": table.get("name")
                    },
                    "confidence": min(0.95, 0.5 + match_count * 0.15)
                })
            
            # Sort by confidence
            results.sort(key=lambda x: x["confidence"], reverse=True)
            
        except Exception as e:
            logger.debug(f"Error searching tables: {str(e)}")
        
        return results[:top_k]
    
    def _search_figures(self, query: str, top_k: int) -> List[Dict]:
        """
        Search for figure-derived facts using keyword matching.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of figure facts
        """
        results = []
        
        # Extract keywords from query
        stop_words = {"what", "is", "the", "of", "a", "an", "for", "and", "or", "to", "in",
                     "how", "why", "when", "where", "which", "are", "can", "do", "does",
                     "this", "that", "these", "those", "be", "been", "being", "have", "has",
                     "it", "its", "they", "their", "we", "our", "you", "your", "show", "me"}
        words = query.split()
        keywords = [w.strip("?.,;:").lower() for w in words 
                    if w.strip("?.,;:").lower() not in stop_words and len(w.strip("?.,;:")) > 2]
        
        if not keywords:
            return results
        
        try:
            or_conditions = " OR ".join(
                [f"toLower(f.description) CONTAINS '{kw}'" for kw in keywords[:6]] +
                [f"toLower(f.name) CONTAINS '{kw}'" for kw in keywords[:6]]
            )
            
            cypher_query = f"""
            MATCH (f:Figure)
            WHERE {or_conditions}
            RETURN f.name as name, f.page_number as page_number,
                   f.document_name as document_name, f.description as description
            LIMIT $limit
            """
            figures = self.client.execute_query(cypher_query, {"limit": top_k * 2})
            
            for figure in figures:
                desc = figure.get("description", "")
                desc_lower = desc.lower()
                match_count = sum(1 for kw in keywords if kw in desc_lower)
                
                results.append({
                    "type": "figure",
                    "content": desc,
                    "figure_name": figure.get("name", ""),
                    "provenance": {
                        "page_number": figure.get("page_number"),
                        "document_name": figure.get("document_name"),
                        "source_type": "figure",
                        "source_name": figure.get("name")
                    },
                    "confidence": min(0.95, 0.5 + match_count * 0.15)
                })
            
            # Sort by confidence
            results.sort(key=lambda x: x["confidence"], reverse=True)
            
        except Exception as e:
            logger.debug(f"Error searching figures: {str(e)}")
        
        return results[:top_k]
    
    def _search_colocated_tables_figures(self, page_numbers: List[int], top_k: int = 5) -> List[Dict]:
        """
        Find tables and figures co-located on the same pages as text chunk results.
        This catches tables/figures that keyword search might miss.
        
        Args:
            page_numbers: Page numbers from text chunk results
            top_k: Maximum results
            
        Returns:
            List of co-located table and figure facts
        """
        results = []
        
        if not page_numbers:
            return results
        
        try:
            # Find tables on same pages
            cypher_tables = """
            MATCH (t:Table)
            WHERE t.page_number IN $pages
            RETURN t.name as name, t.page_number as page_number,
                   t.document_name as document_name, t.content as content,
                   'table' as node_type
            LIMIT $limit
            """
            tables = self.client.execute_query(
                cypher_tables, {"pages": page_numbers, "limit": top_k}
            )
            
            for table in tables:
                results.append({
                    "type": "table",
                    "content": table.get("content", ""),
                    "table_name": table.get("name", ""),
                    "provenance": {
                        "page_number": table.get("page_number"),
                        "document_name": table.get("document_name"),
                        "source_type": "table",
                        "source_name": table.get("name")
                    },
                    "confidence": 0.80
                })
            
            # Find figures on same pages
            cypher_figures = """
            MATCH (f:Figure)
            WHERE f.page_number IN $pages
            RETURN f.name as name, f.page_number as page_number,
                   f.document_name as document_name, f.description as description,
                   'figure' as node_type
            LIMIT $limit
            """
            figures = self.client.execute_query(
                cypher_figures, {"pages": page_numbers, "limit": top_k}
            )
            
            for figure in figures:
                results.append({
                    "type": "figure",
                    "content": figure.get("description", ""),
                    "figure_name": figure.get("name", ""),
                    "provenance": {
                        "page_number": figure.get("page_number"),
                        "document_name": figure.get("document_name"),
                        "source_type": "figure",
                        "source_name": figure.get("name")
                    },
                    "confidence": 0.75
                })
            
        except Exception as e:
            logger.debug(f"Error searching co-located tables/figures: {str(e)}")
        
        return results
