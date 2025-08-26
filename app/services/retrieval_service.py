import logging
from typing import List, Dict, Any
from app.services.embedding_service import EmbeddingService
from app.core.logger import setup_logger

logger = setup_logger(__name__)

class RetrievalService:
    """
    Service responsible for retrieving relevant chunks based on a user query.
    
    This service provides the retrieval functionality for the RAG system by:
    - Converting user queries to embedding vectors
    - Searching the vector database for semantically similar content
    - Retrieving the original text and metadata for the most relevant chunks
    - Returning ranked results with similarity scores
    
    The service acts as the bridge between user queries and the document knowledge base,
    enabling efficient semantic search capabilities.
    """
    
    def __init__(self, embedding_service: EmbeddingService):
        """
        Initialize the retrieval service with an embedding service instance.
        
        The retrieval service depends on the embedding service for:
        - Access to the sentence transformer model for query encoding
        - Access to the FAISS vector index for similarity search
        - Retrieval of metadata associated with embeddings
        
        Args:
            embedding_service (EmbeddingService): An initialized embedding service instance
                that provides access to embedding generation and vector search capabilities.
        """
        self.embedding_service = embedding_service
    
    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant document chunks for a given user query.
        
        This method implements the core retrieval functionality by:
        1. Converting the user's natural language query to an embedding vector
        2. Searching the vector database for semantically similar content
        3. Retrieving the original text and metadata for the top-k results
        4. Formatting results with similarity scores for ranking
        
        The retrieval is based on semantic similarity rather than keyword matching,
        enabling the system to find content that is conceptually related to the query
        even when different terminology is used.
        
        Args:
            query (str): The user's question or search query. This should be a
                meaningful, natural language question or statement that the user
                wants to find information about in the uploaded documents.
            k (int, optional): The number of most relevant chunks to retrieve.
                Defaults to 5. Must be a positive integer.
                
        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the retrieved chunks,
                ordered by relevance (most similar first). Each dictionary contains:
                - 'text' (str): The original text content of the retrieved chunk
                - 'metadata' (Dict[str, Any]): Complete metadata associated with the chunk,
                  including source document, chunk index, and other tracking information
                - 'distance' (float): Similarity distance score (lower values indicate
                  higher similarity). This is the L2 distance between query and chunk embeddings.
                  
                Returns an empty list if no relevant chunks are found or if the vector
                database is empty.
                
        Raises:
            Exception: If any error occurs during the retrieval process, such as issues
                with embedding generation, vector search, or data access.
                
        Example:
            >>> # Initialize services
            >>> embedding_service = EmbeddingService()
            >>> retrieval_service = RetrievalService(embedding_service)
            
            >>> # Perform retrieval
            >>> results = retrieval_service.retrieve_relevant_chunks(
            ...     "What is the power consumption of the motor?",
            ...     k=3
            ... )
            
            >>> # Process results
            >>> print(f"Found {len(results)} relevant chunks")
            >>> for i, result in enumerate(results, 1):
            ...     print(f"Result {i} (distance: {result['distance']:.4f}):")
            ...     print(f"  Source: {result['metadata'].get('source', 'Unknown')}")
            ...     print(f"  Text: {result['text'][:100]}...")
        """
        logger.info(f"Retrieving relevant chunks for query: {query}")
        try:
            # Generate embedding for the query
            query_embedding = self.embedding_service.model.encode([query], convert_to_tensor=False)
            
            # Search in the index
            distances, indices = self.embedding_service.search_similar(query_embedding, k)
            
            # Retrieve the corresponding chunks and metadata
            results = []
            for idx, distance in zip(indices, distances):
                # Check if the index is valid to prevent out-of-range errors
                if idx < len(self.embedding_service.metadata):
                    chunk_data = {
                        "text": self.embedding_service.metadata[idx].get("chunk_text", ""),
                        "metadata": self.embedding_service.metadata[idx],
                        "distance": float(distance)
                    }
                    results.append(chunk_data)
            
            logger.info(f"Retrieved {len(results)} relevant chunks")
            return results
        except Exception as e:
            logger.error(f"Error retrieving relevant chunks: {str(e)}")
            raise