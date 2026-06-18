"""
Context builder that combines graph facts and image results.
Prepares unified evidence package for LLM generation.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds unified context from multiple retrieval sources."""
    
    def __init__(self):
        """Initialize context builder."""
        self.max_context_tokens = 8000  # Approximate token limit
    
    def build(self, graph_results: List[Dict], image_results: List[Dict] = None) -> str:
        """
        Build unified context from graph and image retrieval results.
        
        Args:
            graph_results: Results from graph search
            image_results: Results from image search
            
        Returns:
            Formatted context string for LLM
        """
        context_parts = []
        
        # Add graph evidence
        if graph_results:
            graph_context = self._format_graph_context(graph_results)
            context_parts.append(graph_context)
        
        # Add image evidence
        if image_results:
            image_context = self._format_image_context(image_results)
            context_parts.append(image_context)
        
        # Combine and truncate if necessary
        full_context = "\n\n".join(context_parts)
        
        return self._truncate_context(full_context, self.max_context_tokens)
    
    def _format_graph_context(self, results: List[Dict]) -> str:
        """
        Format graph search results into readable context.
        
        Args:
            results: Graph search results
            
        Returns:
            Formatted text
        """
        context = ["=== DOCUMENT EVIDENCE ===\n"]
        
        # Organize by type
        text_chunks = [r for r in results if r.get("type") == "text_chunk"]
        entities = [r for r in results if r.get("type") == "entity"]
        relationships = [r for r in results if r.get("type") == "relationship"]
        tables = [r for r in results if r.get("type") == "table"]
        figures = [r for r in results if r.get("type") == "figure"]
        
        # Add text chunks (most important - actual document content)
        if text_chunks:
            context.append("Relevant Document Text:")
            for chunk in text_chunks[:10]:
                content = chunk.get("content", "")
                page = chunk.get("provenance", {}).get("page_number", "?")
                doc = chunk.get("provenance", {}).get("document_name", "")
                context.append(f"\n  [Page {page} - {doc}]")
                context.append(f"  {content[:800]}")
            context.append("")
        
        # Add table facts
        if tables:
            context.append("Table Data:")
            for table in tables[:5]:
                name = table.get("table_name", "")
                content = table.get("content", "")[:600]
                markdown_content = table.get("markdown", "")
                page = table.get("provenance", {}).get("page_number", "?")
                context.append(f"  • {name} (p.{page})")
                if markdown_content:
                    context.append(f"    {markdown_content[:600]}")
                else:
                    context.append(f"    {content}")
            context.append("")
        
        # Add figure facts
        if figures:
            context.append("Figure References:")
            for figure in figures[:5]:
                name = figure.get("figure_name", "")
                content = figure.get("content", "")[:300]
                page = figure.get("provenance", {}).get("page_number", "?")
                context.append(f"  • {name} (p.{page})")
                context.append(f"    {content}")
            context.append("")
        
        # Add entity facts (supplementary)
        if entities and not text_chunks:
            context.append("Entity Facts:")
            for entity in entities[:5]:
                content = entity.get("content", "")
                page = entity.get("provenance", {}).get("page_number", "?")
                context.append(f"  • {content} (p.{page})")
            context.append("")
        
        return "\n".join(context)
    
    def _format_image_context(self, results: List[Dict]) -> str:
        """
        Format image search results into context.
        
        Args:
            results: Image search results
            
        Returns:
            Formatted text
        """
        context = ["=== IMAGE EVIDENCE ===\n"]
        
        if not results:
            return ""
        
        context.append(f"Found {len(results)} relevant images:\n")
        
        for i, image in enumerate(results[:3], 1):
            page = image.get("page_number", "?")
            caption = image.get("caption", "")
            similarity = image.get("similarity", 0)
            
            context.append(f"{i}. {caption}")
            context.append(f"   Location: p.{page}, Relevance: {similarity:.2%}")
            context.append("")
        
        context.append("\nRefer to images above for visual reference.\n")
        
        return "\n".join(context)
    
    def _truncate_context(self, context: str, max_tokens: int) -> str:
        """
        Truncate context to approximate token limit.
        
        Args:
            context: Full context text
            max_tokens: Maximum tokens
            
        Returns:
            Truncated context
        """
        # Simple heuristic: approximately 4 characters per token
        max_chars = max_tokens * 4
        
        if len(context) > max_chars:
            # Truncate and add indicator
            truncated = context[:max_chars]
            # Try to end at a complete section
            last_newline = truncated.rfind("\n")
            if last_newline > 0:
                truncated = truncated[:last_newline]
            truncated += "\n\n[Context truncated for length...]"
            return truncated
        
        return context
    
    def build_with_raw_results(self, results: Dict[str, Any]) -> str:
        """
        Build context from raw retrieval results.
        
        Args:
            results: Dictionary with 'graph' and 'images' keys
            
        Returns:
            Formatted context
        """
        graph_results = results.get("graph", [])
        image_results = results.get("images", [])
        
        return self.build(graph_results, image_results)
    
    def estimate_tokens(self, context: str) -> int:
        """
        Estimate token count for context.
        
        Args:
            context: Context text
            
        Returns:
            Approximate token count
        """
        # Simple heuristic
        return len(context) // 4
    
    def merge_results(self, *result_lists: List[Dict]) -> List[Dict]:
        """
        Merge multiple result lists and deduplicate.
        
        Args:
            *result_lists: Variable number of result lists
            
        Returns:
            Merged and deduplicated results
        """
        merged = []
        seen = set()
        
        for results in result_lists:
            for result in results:
                # Create unique key
                key = (
                    result.get("type", ""),
                    result.get("content", "")[:50]
                )
                
                if key not in seen:
                    merged.append(result)
                    seen.add(key)
        
        return merged
    
    def rerank_results(self, results: List[Dict], query: str) -> List[Dict]:
        """
        Simple reranking of results by relevance.
        
        Args:
            results: Retrieved results
            query: Original query
            
        Returns:
            Reranked results
        """
        query_words = set(query.lower().split())
        
        def calculate_score(result):
            content = (result.get("content", "") + 
                      result.get("table_name", "") + 
                      result.get("figure_name", "")).lower()
            
            # Score based on word overlap
            content_words = set(content.split())
            overlap = len(query_words & content_words)
            
            # Boost confidence scores
            confidence = result.get("confidence", 0.5)
            
            return overlap * 0.5 + confidence * 0.5
        
        # Sort by score
        scored = [(r, calculate_score(r)) for r in results]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [r for r, _ in scored]
