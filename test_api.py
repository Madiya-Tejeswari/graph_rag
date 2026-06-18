"""
Example API usage and test cases for PG&E GraphRAG backend.
"""

import json
import requests
from typing import Dict, Any

# API Base URL
BASE_URL = "http://localhost:8000"


class GraphRAGClient:
    """Client for PG&E GraphRAG API."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
    
    def health_check(self) -> Dict[str, Any]:
        """Check if API is healthy."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def graph_status(self) -> Dict[str, Any]:
        """Get graph database status."""
        response = requests.get(f"{self.base_url}/graph/status")
        return response.json()
    
    def query(self, query_text: str, model: str = "gpt-4") -> Dict[str, Any]:
        """Submit RAG query."""
        payload = {
            "query": query_text,
            "model": model,
            "rag_approach": "graph_rag"
        }
        response = requests.post(
            f"{self.base_url}/rag",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        return response.json()
    
    def ingest(self) -> Dict[str, Any]:
        """Start ingestion process."""
        response = requests.post(f"{self.base_url}/ingest")
        return response.json()


def test_api():
    """Run API tests."""
    client = GraphRAGClient()
    
    print("=" * 80)
    print("PG&E GraphRAG API Tests")
    print("=" * 80)
    
    # Test 1: Health Check
    print("\n1. Health Check")
    print("-" * 40)
    try:
        health = client.health_check()
        print(f"Status: {health.get('status')}")
        print("✓ Health check passed")
    except Exception as e:
        print(f"✗ Health check failed: {e}")
        return
    
    # Test 2: Graph Status
    print("\n2. Graph Status")
    print("-" * 40)
    try:
        status = client.graph_status()
        print(f"Graph Status: {status.get('status')}")
        print(f"Documents: {status.get('documents')}")
        print(f"Entities: {status.get('entities')}")
        print(f"Relationships: {status.get('relationships')}")
        print(f"Images: {status.get('images')}")
        print("✓ Graph status retrieved")
    except Exception as e:
        print(f"✗ Graph status failed: {e}")
        return
    
    # Test 3: Query - Transformer Capacity
    print("\n3. Query: Transformer Capacity")
    print("-" * 40)
    try:
        response = client.query(
            "What is the maximum transformer size for single phase service?",
            model="gpt-4"
        )
        
        if response.get('status') == 'success':
            print(f"Query: {response['query']}")
            print(f"Answer: {response['answer'][:200]}...")
            print(f"Sources: {[s['title'] for s in response['sources']]}")
            print(f"Total Time: {response['metadata']['total_time_ms']}ms")
            print("✓ Query successful")
        else:
            print(f"✗ Query failed: {response.get('error')}")
    except Exception as e:
        print(f"✗ Query failed: {e}")
    
    # Test 4: Query - Service Connection Diagram
    print("\n4. Query: Service Connection Diagram")
    print("-" * 40)
    try:
        response = client.query(
            "How does a typical underground service connection look?",
            model="gpt-4"
        )
        
        if response.get('status') == 'success':
            print(f"Query: {response['query']}")
            print(f"Answer: {response['answer'][:200]}...")
            print(f"Sources: {[s['title'] for s in response['sources']]}")
            
            # Check if image was retrieved
            metadata = response.get('metadata', {})
            print(f"Retrieval Method: {metadata.get('retrieval_method')}")
            print("✓ Query successful")
        else:
            print(f"✗ Query failed: {response.get('error')}")
    except Exception as e:
        print(f"✗ Query failed: {e}")
    
    # Test 5: Query - Equipment Requirements
    print("\n5. Query: Equipment Requirements")
    print("-" * 40)
    try:
        response = client.query(
            "What equipment is required for a 75 kVA service?",
            model="gpt-4"
        )
        
        if response.get('status') == 'success':
            print(f"Query: {response['query']}")
            print(f"Answer: {response['answer'][:200]}...")
            print(f"Sources: {[s['title'] for s in response['sources']]}")
            print("✓ Query successful")
        else:
            print(f"✗ Query failed: {response.get('error')}")
    except Exception as e:
        print(f"✗ Query failed: {e}")
    
    # Test 6: Different Model
    print("\n6. Query with Different Model (Claude)")
    print("-" * 40)
    try:
        response = client.query(
            "What is the difference between single and three phase service?",
            model="claude-3-sonnet-20240229"
        )
        
        if response.get('status') == 'success':
            print(f"Model Used: {response['metadata']['model_used']}")
            print(f"Answer: {response['answer'][:200]}...")
            print("✓ Query successful with Claude")
        else:
            print(f"✗ Query failed: {response.get('error')}")
    except Exception as e:
        print(f"✗ Query failed: {e}")
    
    print("\n" + "=" * 80)
    print("Tests Completed")
    print("=" * 80)


# Example queries for manual testing
EXAMPLE_QUERIES = [
    # Text-based questions
    {
        "query": "What is the maximum transformer size for single phase service?",
        "expected_source": "table",
        "description": "Question answerable from table data"
    },
    {
        "query": "What equipment is shown in the underground service connection diagram?",
        "expected_source": "figure",
        "description": "Question about diagram components"
    },
    {
        "query": "How does a typical underground service connection look?",
        "expected_source": "image",
        "description": "Visual question expecting image retrieval"
    },
    {
        "query": "What is single phase service?",
        "expected_source": "text",
        "description": "Definition question from text"
    },
    {
        "query": "What are the maximum load requirements for residential service?",
        "expected_source": "table",
        "description": "Table lookup question"
    },
    {
        "query": "Describe the typical installation process for underground service.",
        "expected_source": "figure",
        "description": "Process description from figures"
    },
]


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_api()
    else:
        # Interactive mode
        client = GraphRAGClient()
        
        print("PG&E GraphRAG API Client")
        print("Type 'quit' to exit")
        print()
        
        while True:
            query = input("Query: ").strip()
            
            if query.lower() == 'quit':
                break
            
            if not query:
                continue
            
            model = input("Model (default: gpt-4): ").strip() or "gpt-4"
            
            try:
                response = client.query(query, model)
                
                if response.get('status') == 'success':
                    print("\nAnswer:")
                    print(response['answer'])
                    print(f"\nSources: {', '.join([s['title'] for s in response['sources']])}")
                    print(f"Time: {response['metadata']['total_time_ms']}ms")
                else:
                    print(f"Error: {response.get('error')}")
                
            except Exception as e:
                print(f"Error: {e}")
            
            print()
