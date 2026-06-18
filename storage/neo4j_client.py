"""
Neo4j database client for graph operations.
Handles connection, node/relationship creation, and Cypher queries.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from neo4j import GraphDatabase, Session, Transaction
from neo4j.exceptions import ServiceUnavailable, DriverError

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Client for Neo4j graph database operations."""
    
    def __init__(self, uri: str, username: str, password: str, database: str = "neo4j"):
        """
        Initialize Neo4j client.
        
        Args:
            uri: Neo4j connection URI (e.g., 'bolt://localhost:7687')
            username: Neo4j username
            password: Neo4j password
            database: Database name (default: 'neo4j')
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.database = database
        self.driver = None
        self.session: Optional[Session] = None
        
        logger.info(f"Neo4jClient initialized with URI: {uri}")
    
    def connect(self) -> bool:
        """
        Establish connection to Neo4j database.
        
        Returns:
            True if connection successful
        """
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                encrypted=False  # Set to True for production with SSL
            )
            
            # Verify connection
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1")
            
            logger.info(f"Successfully connected to Neo4j at {self.uri}")
            return True
            
        except ServiceUnavailable:
            logger.error(f"Neo4j service unavailable at {self.uri}")
            return False
        except DriverError as e:
            logger.error(f"Error connecting to Neo4j: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Neo4j: {str(e)}")
            return False
    
    def close(self) -> None:
        """Close Neo4j connection."""
        try:
            if self.driver:
                self.driver.close()
                logger.info("Neo4j connection closed")
        except Exception as e:
            logger.error(f"Error closing Neo4j connection: {str(e)}")
    
    def create_node(self, label: str, properties: Dict[str, Any]) -> Optional[int]:
        """
        Create a node in the graph.
        
        Args:
            label: Node label (type)
            properties: Node properties as dictionary
            
        Returns:
            Node ID if successful, None otherwise
        """
        try:
            if not self.driver:
                logger.error("Neo4j driver not initialized")
                return None
            
            # Build property string for Cypher
            prop_str = ", ".join([f"{k}: ${k}" for k in properties.keys()])
            
            query = f"""
            CREATE (n:{label} {{{prop_str}}})
            RETURN id(n) as node_id
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, **properties)
                record = result.single()
                
                if record:
                    node_id = record["node_id"]
                    logger.debug(f"Created {label} node with ID: {node_id}")
                    return node_id
                
        except Exception as e:
            logger.error(f"Error creating {label} node: {str(e)}")
        
        return None
    
    def create_relationship(self, source_id: int, target_id: int, relationship_type: str,
                           properties: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a relationship between two nodes.
        
        Args:
            source_id: Source node ID
            target_id: Target node ID
            relationship_type: Relationship type
            properties: Optional relationship properties
            
        Returns:
            True if successful
        """
        try:
            if not self.driver:
                logger.error("Neo4j driver not initialized")
                return False
            
            properties = properties or {}
            prop_str = ""
            
            if properties:
                prop_str = "{" + ", ".join([f"{k}: ${k}" for k in properties.keys()]) + "}"
            
            query = f"""
            MATCH (source) WHERE id(source) = $source_id
            MATCH (target) WHERE id(target) = $target_id
            CREATE (source)-[r:{relationship_type}{prop_str}]->(target)
            RETURN r
            """
            
            with self.driver.session(database=self.database) as session:
                params = {
                    "source_id": source_id,
                    "target_id": target_id,
                    **properties
                }
                result = session.run(query, **params)
                
                if result.single():
                    logger.debug(f"Created {relationship_type} relationship: {source_id}->{target_id}")
                    return True
        
        except Exception as e:
            logger.error(f"Error creating relationship: {str(e)}")
        
        return False
    
    def search_entities(self, keyword: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Search for entities matching keyword.
        
        Args:
            keyword: Search keyword
            top_k: Maximum number of results
            
        Returns:
            List of matching entities
        """
        try:
            if not self.driver:
                logger.error("Neo4j driver not initialized")
                return []
            
            query = """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($keyword)
            RETURN e.name as name, e.type as type, e.page_number as page_number,
                   e.document_name as document_name, id(e) as node_id
            LIMIT $limit
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, keyword=keyword, limit=top_k)
                entities = [dict(record) for record in result]
                
                logger.debug(f"Found {len(entities)} entities for keyword: {keyword}")
                return entities
        
        except Exception as e:
            logger.error(f"Error searching entities: {str(e)}")
            return []
    
    def execute_query(self, cypher_query: str, parameters: Optional[Dict] = None) -> List[Dict]:
        """
        Execute a custom Cypher query.
        
        Args:
            cypher_query: Cypher query string
            parameters: Query parameters
            
        Returns:
            List of result records
        """
        try:
            if not self.driver:
                logger.error("Neo4j driver not initialized")
                return []
            
            parameters = parameters or {}
            
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher_query, **parameters)
                records = [dict(record) for record in result]
                
                logger.debug(f"Query executed, returned {len(records)} records")
                return records
        
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            return []
    
    def find_by_relationship(self, relationship_type: str, limit: int = 20) -> List[Dict]:
        """
        Find relationships of a specific type.
        
        Args:
            relationship_type: Type of relationship to find
            limit: Maximum number of results
            
        Returns:
            List of relationship records
        """
        try:
            if not self.driver:
                logger.error("Neo4j driver not initialized")
                return []
            
            query = f"""
            MATCH (source)-[r:{relationship_type}]->(target)
            RETURN source, r, target
            LIMIT $limit
            """
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, limit=limit)
                relationships = []
                
                for record in result:
                    relationships.append({
                        "source": dict(record["source"]),
                        "relationship": record["r"].type,
                        "target": dict(record["target"]),
                        "source_page": record["source"].get("page_number"),
                        "source_document": record["source"].get("document_name")
                    })
                
                logger.debug(f"Found {len(relationships)} {relationship_type} relationships")
                return relationships
        
        except Exception as e:
            logger.error(f"Error finding relationships: {str(e)}")
            return []
    
    def create_indexes(self) -> bool:
        """
        Create indexes on commonly searched properties.
        Improves query performance significantly.
        
        Returns:
            True if successful
        """
        try:
            if not self.driver:
                logger.error("Neo4j driver not initialized")
                return False
            
            indexes = [
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.document_name)",
                "CREATE INDEX IF NOT EXISTS FOR (t:Table) ON (t.name)",
                "CREATE INDEX IF NOT EXISTS FOR (f:Figure) ON (f.name)",
                "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.name)",
                "CREATE INDEX IF NOT EXISTS FOR (p:Page) ON (p.page_number)",
                "CREATE INDEX IF NOT EXISTS FOR (c:Component) ON (c.name)",
                "CREATE INDEX IF NOT EXISTS FOR (fa:Fact) ON (fa.fact)",
                "CREATE INDEX IF NOT EXISTS FOR (tc:TextChunk) ON (tc.page_number)",
                "CREATE INDEX IF NOT EXISTS FOR (tc:TextChunk) ON (tc.document_name)",
            ]
            
            with self.driver.session(database=self.database) as session:
                for index_query in indexes:
                    try:
                        session.run(index_query)
                        logger.debug(f"Index created: {index_query[:50]}...")
                    except Exception as e:
                        logger.debug(f"Index already exists or error: {str(e)}")
            
            logger.info("Indexes created/verified successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error creating indexes: {str(e)}")
            return False
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the graph database.
        
        Returns:
            Dictionary with graph statistics
        """
        try:
            if not self.driver:
                return {
                    "status": "disconnected",
                    "error": "Neo4j driver not initialized"
                }
            
            with self.driver.session(database=self.database) as session:
                # Count nodes by label
                node_counts = {}
                query = "MATCH (n) RETURN labels(n)[0] as label, count(*) as count"
                result = session.run(query)
                
                for record in result:
                    label = record["label"]
                    count = record["count"]
                    if label:
                        node_counts[label] = count
                
                # Count relationships
                rel_query = "MATCH ()-[r]-() RETURN count(r) as rel_count"
                rel_result = session.run(rel_query)
                rel_count = rel_result.single()["rel_count"]
                
                # Count documents
                doc_query = "MATCH (d:Document) RETURN count(d) as doc_count"
                doc_result = session.run(doc_query)
                doc_count = doc_result.single()["doc_count"]
                
                stats = {
                    "status": "connected",
                    "node_counts": node_counts,
                    "total_nodes": sum(node_counts.values()),
                    "total_relationships": rel_count,
                    "documents": doc_count,
                    "database": self.database
                }
                
                logger.debug(f"Graph stats: {stats}")
                return stats
        
        except Exception as e:
            logger.error(f"Error getting graph stats: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def clear_database(self, confirm: bool = False) -> bool:
        """
        Clear all data from the database.
        
        Args:
            confirm: Must be True to actually delete data (safety measure)
            
        Returns:
            True if successful
        """
        if not confirm:
            logger.warning("Database clear requires confirm=True")
            return False
        
        try:
            if not self.driver:
                logger.error("Neo4j driver not initialized")
                return False
            
            with self.driver.session(database=self.database) as session:
                session.run("MATCH (n) DETACH DELETE n")
                logger.warning("Database cleared successfully")
                return True
        
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")
            return False
    
    def get_node(self, node_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a node by its ID.
        
        Args:
            node_id: Node ID
            
        Returns:
            Node data if found
        """
        try:
            if not self.driver:
                logger.error("Neo4j driver not initialized")
                return None
            
            query = "MATCH (n) WHERE id(n) = $node_id RETURN n"
            
            with self.driver.session(database=self.database) as session:
                result = session.run(query, node_id=node_id)
                record = result.single()
                
                if record:
                    return dict(record["n"])
        
        except Exception as e:
            logger.error(f"Error retrieving node: {str(e)}")
        
        return None
    
    def delete_node(self, node_id: int) -> bool:
        """
        Delete a node and its relationships.
        
        Args:
            node_id: Node ID to delete
            
        Returns:
            True if successful
        """
        try:
            if not self.driver:
                logger.error("Neo4j driver not initialized")
                return False
            
            query = "MATCH (n) WHERE id(n) = $node_id DETACH DELETE n"
            
            with self.driver.session(database=self.database) as session:
                session.run(query, node_id=node_id)
                logger.debug(f"Deleted node: {node_id}")
                return True
        
        except Exception as e:
            logger.error(f"Error deleting node: {str(e)}")
            return False
