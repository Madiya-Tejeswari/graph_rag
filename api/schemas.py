"""
Pydantic models for API request/response validation.
Matches the frontend contract schema exactly.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class RAGRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(..., description="User's question", min_length=1)
    model: str = Field(..., description="Model ID from /models")
    ragapproach: str = Field(default="vector_search", description="RAG approach")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is the maximum transformer size for single phase service?",
                "model": "groq/llama-3.3-70b-versatile",
                "ragapproach": "vector_search"
            }
        }


class Source(BaseModel):
    """Source citation model."""
    title: str = Field(..., description="Source title")
    url: str = Field(..., description="Source URL")
    pageno: str = Field(..., description="Page number as string")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "PG&E Greenbook Manual Full - Page 45",
                "url": "/docs/greenbook-manual-full.pdf#page=45",
                "pageno": "45"
            }
        }


class ImageResult(BaseModel):
    """Image in RAG response - base64-encoded with data URI prefix."""
    image_base64: str = Field(
        ...,
        description="Base64-encoded image with data URI prefix"
    )


class Metadata(BaseModel):
    """Response metadata model."""
    retrievaltimems: int = Field(..., description="Milliseconds spent on retrieval")
    generationtimems: int = Field(..., description="Milliseconds spent on generation")
    totaltimems: int = Field(..., description="Total processing time in milliseconds")
    generatedat: str = Field(..., description="ISO timestamp of generation")
    inputtokens: int = Field(..., description="Input tokens used")
    outputtokens: int = Field(..., description="Output tokens used")
    totaltokens: int = Field(..., description="Total tokens used")
    modelused: str = Field(..., description="LLM model used for generation")
    retrievalmethod: str = Field(default="vector_search", description="Retrieval method used")

    class Config:
        json_schema_extra = {
            "example": {
                "retrievaltimems": 245,
                "generationtimems": 1523,
                "totaltimems": 1768,
                "generatedat": "2024-01-15T10:30:00",
                "inputtokens": 1200,
                "outputtokens": 450,
                "totaltokens": 1650,
                "modelused": "groq/llama-3.3-70b-versatile",
                "retrievalmethod": "vector_search"
            }
        }


class RAGResponse(BaseModel):
    """Response model for RAG queries."""
    status: str = Field(..., description="Response status (success/error)")
    query: str = Field(..., description="Original user query")
    answer: str = Field(..., description="The RAG answer")
    sources: List[Source] = Field(default_factory=list, description="Source citations")
    images: List[ImageResult] = Field(default_factory=list, description="Relevant images as base64")
    metadata: Metadata = Field(..., description="Response metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "success",
                "query": "What is the maximum transformer size for single phase service?",
                "answer": "The maximum transformer size for single phase service is 100 kVA.",
                "sources": [
                    {
                        "title": "PG&E Greenbook Manual Full - Page 52",
                        "url": "/docs/greenbook-manual-full.pdf#page=52",
                        "pageno": "52"
                    }
                ],
                "images": [
                    {
                        "image_base64": "data:image/png;base64,iVBORw0KGgo..."
                    }
                ],
                "metadata": {
                    "retrievaltimems": 245,
                    "generationtimems": 1523,
                    "totaltimems": 1768,
                    "generatedat": "2024-01-15T10:30:00",
                    "inputtokens": 1200,
                    "outputtokens": 450,
                    "totaltokens": 1650,
                    "modelused": "groq/llama-3.3-70b-versatile",
                    "retrievalmethod": "vector_search"
                }
            }
        }


# -- Supporting models (used internally, not part of the main contract) --


class GraphNode(BaseModel):
    """Graph node representation."""
    id: str
    label: str
    properties: dict


class GraphRelationship(BaseModel):
    """Graph relationship representation."""
    source: str
    target: str
    type: str
    properties: dict


class GraphStatistics(BaseModel):
    """Graph database statistics."""
    documents: int
    pages: int
    entities: int
    relationships: int
    images: int
    tables: int
    figures: int
