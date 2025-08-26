import os
import pickle
from typing import List, Dict, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

from app.core.config import config
from app.core.logger import setup_logger

logger = setup_logger(__name__)

class EmbeddingService:
    """
    Service responsible for generating embeddings and managing the vector database.
    
    This class provides the core functionality for converting text documents into
    numerical embeddings and managing a FAISS-based vector database for efficient
    similarity search. It handles the complete pipeline from text processing to
    vector storage and retrieval.
    
    Key Features:
    - Text embedding generation using Sentence Transformers
    - FAISS index creation and management
    - Metadata storage alongside embeddings
    - Persistent storage of vector databases
    - Similarity search functionality
    
    The service is designed to be a critical component of the RAG (Retrieval-Augmented Generation)
    system, enabling efficient retrieval of relevant document sections based on semantic similarity.
    """
    
    def __init__(self):
        """
        Initialize the embedding service with the configured model.
        
        Sets up the sentence transformer model, prepares the vector database
        directory structure, and initializes internal state variables.
        The service will attempt to load any existing index during initialization.
        """
        self.model_name = config.EMBEDDING_MODEL
        self.model = SentenceTransformer(self.model_name)
        self.vector_db_path = config.VECTOR_DB_PATH
        self.index = None
        self.metadata = []
        
        # Create vector database directory if it doesn't exist
        if not os.path.exists(self.vector_db_path):
            os.makedirs(self.vector_db_path)
    
    def generate_embeddings(self, chunks: List[str]) -> np.ndarray:
        """
        Generate numerical embeddings for a list of text chunks.
        
        This method uses the configured Sentence Transformer model to convert
        text chunks into high-dimensional vector representations that capture
        the semantic meaning of the text.
        
        Args:
            chunks (List[str]): List of text chunks to embed. Each chunk should be
                               a string containing a portion of document text.
        
        Returns:
            np.ndarray: A numpy array containing the generated embeddings. The shape
                       will be (n_chunks, embedding_dimension), where n_chunks is
                       the number of input chunks and embedding_dimension is
                       determined by the model.
        
        Raises:
            Exception: If there is an error during the embedding generation process.
                       The original exception is logged and re-raised.
        
        Example:
            >>> service = EmbeddingService()
            >>> embeddings = service.generate_embeddings(["Hello world", "How are you?"])
            >>> print(embeddings.shape)
            (2, 384)  # 2 chunks, 384-dimensional embeddings
        """
        logger.info(f"Generating embeddings for {len(chunks)} chunks")
        try:
            embeddings = self.model.encode(chunks, convert_to_tensor=False)
            logger.info("Embeddings generated successfully")
            return np.array(embeddings)
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def create_index(self, chunks_with_metadata: List[Dict[str, Any]]):
        """
        Create a FAISS index and store embeddings with metadata.
        
        This method processes a list of text chunks along with their metadata,
        generates embeddings for all chunks, creates a FAISS index for efficient
        similarity search, and persists both the index and metadata to disk.
        
        The process involves:
        1. Extracting text chunks and metadata from input
        2. Generating embeddings for all text chunks
        3. Creating a FAISS index with the generated embeddings
        4. Storing metadata alongside the original text for retrieval
        5. Saving both index and metadata to persistent storage
        
        Args:
            chunks_with_metadata (List[Dict[str, Any]]): A list of dictionaries where
                each dictionary contains:
                - 'text': The text content to embed (str)
                - 'metadata': Additional information about the chunk (Dict[str, Any])
        
        Raises:
            Exception: If there is an error during index creation. The original
                       exception is logged and re-raised.
        
        Example:
            >>> service = EmbeddingService()
            >>> chunks = [
            ...     {"text": "Hello world", "metadata": {"source": "doc1.pdf"}},
            ...     {"text": "How are you?", "metadata": {"source": "doc1.pdf"}}
            ... ]
            >>> service.create_index(chunks)
        """
        logger.info("Creating FAISS index")
        try:
            # Extract texts and metadata from input
            chunks = [item["text"] for item in chunks_with_metadata]
            metadata = [item["metadata"] for item in chunks_with_metadata]
            
            # Generate embeddings for all text chunks
            embeddings = self.generate_embeddings(chunks)
            
            # Create FAISS index with the generated embeddings
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            
            # Add the original chunk text to metadata for later retrieval
            for i, meta in enumerate(metadata):
                meta["chunk_text"] = chunks[i]
            self.metadata = metadata
            
            # Save the index and metadata to disk for persistence
            faiss.write_index(self.index, os.path.join(self.vector_db_path, "faiss.index"))
            with open(os.path.join(self.vector_db_path, "metadata.pkl"), "wb") as f:
                pickle.dump(self.metadata, f)
            
            logger.info("FAISS index created and saved successfully")
        except Exception as e:
            logger.error(f"Error creating FAISS index: {str(e)}")
            raise
    
    def load_index(self):
        """
        Load an existing FAISS index and its metadata from disk.
        
        This method attempts to load a previously saved FAISS index and its
        associated metadata from persistent storage. If the index files don't
        exist, it logs a warning message but doesn't raise an exception, allowing
        the service to continue without an existing index (a new one will be
        created when documents are added).
        
        The method checks for both the index file (.index) and metadata file (.pkl)
        and only proceeds if both files exist.
        
        Raises:
            Exception: If there is an error during the loading process (other than
                       missing files). The original exception is logged and re-raised.
        """
        logger.info("Loading existing FAISS index")
        try:
            index_path = os.path.join(self.vector_db_path, "faiss.index")
            metadata_path = os.path.join(self.vector_db_path, "metadata.pkl")
            
            if os.path.exists(index_path) and os.path.exists(metadata_path):
                # Load the FAISS index from file
                self.index = faiss.read_index(index_path)
                
                # Load the metadata from pickle file
                with open(metadata_path, "rb") as f:
                    self.metadata = pickle.load(f)
                
                logger.info("FAISS index loaded successfully")
            else:
                logger.warning("No FAISS index found. A new one will be created when documents are added.")
        except Exception as e:
            logger.error(f"Error loading FAISS index: {str(e)}")
            raise
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> tuple:
        """
        Search for the k most similar vectors in the index.
        
        This method performs a similarity search using the FAISS index to find
        the k vectors most similar to the query embedding. The search is based
        on Euclidean distance (L2) as specified by the IndexFlatL2 configuration.
        
        If no index is currently loaded, the method attempts to load an existing
        index before performing the search.
        
        Args:
            query_embedding (np.ndarray): The embedding vector of the query.
                                         Should be a 1D numpy array with the same
                                         dimension as the indexed embeddings.
            k (int, optional): The number of results to return. Defaults to 5.
                                Must be a positive integer.
        
        Returns:
            tuple: A tuple containing two numpy arrays:
                - distances: The distances from the query to each result.
                            Lower values indicate higher similarity.
                - indices: The indices of the results in the metadata list.
                           These can be used to retrieve the corresponding
                           metadata and original text.
        
        Raises:
            Exception: If there is an error during the search process. The original
                       exception is logged and re-raised.
        
        Example:
            >>> service = EmbeddingService()
            >>> query_emb = service.generate_embeddings(["Hello world"])[0]
            >>> distances, indices = service.search_similar(query_emb, k=3)
            >>> print(f"Found {len(indices)} similar chunks")
        """
        logger.info(f"Searching for the {k} most similar vectors")
        try:
            # Load index if not already loaded
            if self.index is None:
                self.load_index()
            
            # Perform the similarity search
            distances, indices = self.index.search(query_embedding, k)
            logger.info("Similarity search completed")
            
            # Return the first (and only) query's results
            return distances[0], indices[0]
        except Exception as e:
            logger.error(f"Error during similarity search: {str(e)}")
            raise