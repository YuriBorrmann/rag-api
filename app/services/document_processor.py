import os
from typing import List, Dict, Any
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.core.config import config
from app.core.logger import setup_logger

logger = setup_logger(__name__)

class DocumentProcessor:
    """
    Service responsible for extracting text from PDFs and splitting it into chunks.
    
    This class handles the core document processing tasks:
    1. Extracting text content from PDF files
    2. Splitting large text documents into smaller, manageable chunks
    3. Adding metadata to track the source of each chunk
    4. Processing multiple documents with error handling
    
    The processed chunks are suitable for embedding and storage in vector databases.
    """
    
    def __init__(self):
        """
        Initialize the document processor with chunking configurations.
        
        Sets up the text splitter with parameters from the configuration:
        - chunk_size: Maximum number of characters per chunk
        - chunk_overlap: Number of overlapping characters between chunks
        """
        self.chunk_size = config.CHUNK_SIZE
        self.chunk_overlap = config.CHUNK_OVERLAP
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text content from a PDF file.
        
        This method reads a PDF file and concatenates text from all pages.
        It handles various PDF formats and provides detailed logging.
        
        Args:
            file_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text content from the PDF
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            Exception: If text extraction fails
            
        Example:
            >>> processor = DocumentProcessor()
            >>> text = processor.extract_text_from_pdf("document.pdf")
            >>> print(f"Extracted {len(text)} characters")
        """
        logger.info(f"Starting text extraction from file: {file_path}")
        
        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                text = ""
                
                # Extract text from each page
                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            text += page_text + "\n"
                        else:
                            logger.warning(f"Page {page_num} is empty or contains no extractable text")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num}: {str(e)}")
                        continue
                
                logger.info(f"Text extraction completed for file: {file_path}")
                return text.strip()
                
        except Exception as e:
            logger.error(f"Error extracting text from file {file_path}: {str(e)}")
            raise
    
    def split_text_into_chunks(self, text: str) -> List[str]:
        """
        Split text into smaller chunks using the configured text splitter.
        
        This method takes a large text document and splits it into smaller,
        overlapping chunks suitable for processing by embedding models.
        
        Args:
            text (str): Text content to split into chunks
            
        Returns:
            List[str]: List of text chunks
            
        Raises:
            ValueError: If text is empty or invalid
            Exception: If chunking fails
            
        Example:
            >>> processor = DocumentProcessor()
            >>> chunks = processor.split_text_into_chunks("Large text document...")
            >>> print(f"Created {len(chunks)} chunks")
        """
        logger.info("Starting text splitting into chunks")
        
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []
        
        try:
            chunks = self.text_splitter.split_text(text)
            logger.info(f"Text split into {len(chunks)} chunks")
            
            # Log chunk statistics
            if chunks:
                chunk_lengths = [len(chunk) for chunk in chunks]
                logger.debug(f"Chunk sizes - min: {min(chunk_lengths)}, max: {max(chunk_lengths)}, avg: {sum(chunk_lengths)/len(chunk_lengths):.1f}")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error splitting text into chunks: {str(e)}")
            raise
    
    def process_documents(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple PDF documents, extracting text and creating chunks with metadata.
        
        This is the main processing method that:
        1. Handles multiple files in batch
        2. Extracts text from each PDF
        3. Splits text into manageable chunks
        4. Adds metadata to track the source of each chunk
        5. Provides error recovery for individual files
        
        Args:
            file_paths (List[str]): List of paths to PDF files
            
        Returns:
            List[Dict[str, Any]]: List of dictionaries containing:
                - text: The chunk content
                - metadata: Source information and tracking data
                
        Raises:
            ValueError: If no valid files are provided
            Exception: If critical processing errors occur
            
        Example:
            >>> processor = DocumentProcessor()
            >>> results = processor.process_documents(["doc1.pdf", "doc2.pdf"])
            >>> print(f"Processed {len(results)} chunks total")
        """
        if not file_paths:
            raise ValueError("No file paths provided")
        
        logger.info(f"Starting batch processing of {len(file_paths)} documents")
        
        chunks_with_metadata = []
        successful_files = 0
        failed_files = 0
        
        for file_path in file_paths:
            try:
                logger.debug(f"Processing file: {file_path}")
                
                # Extract text from PDF
                text = self.extract_text_from_pdf(file_path)
                
                if not text.strip():
                    logger.warning(f"No extractable text found in {file_path}")
                    failed_files += 1
                    continue
                
                # Split text into chunks
                chunks = self.split_text_into_chunks(text)
                
                if not chunks:
                    logger.warning(f"No chunks created from {file_path}")
                    failed_files += 1
                    continue
                
                # Create chunks with metadata
                source = os.path.basename(file_path)
                file_chunks = self._create_chunks_with_metadata(chunks, source)
                chunks_with_metadata.extend(file_chunks)
                
                successful_files += 1
                logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks")
                
            except Exception as e:
                failed_files += 1
                logger.error(f"Failed to process document {file_path}: {str(e)}")
                continue
        
        # Log processing summary
        total_chunks = len(chunks_with_metadata)
        logger.info(f"Batch processing completed: {successful_files} successful, {failed_files} failed, {total_chunks} total chunks")
        
        if successful_files == 0:
            logger.error("No files were successfully processed")
            raise Exception("No files were successfully processed")
        
        return chunks_with_metadata
    
    def _create_chunks_with_metadata(self, chunks: List[str], source: str) -> List[Dict[str, Any]]:
        """
        Create chunk dictionaries with metadata for tracking.
        
        Args:
            chunks (List[str]): List of text chunks
            source (str): Source file name
            
        Returns:
            List[Dict[str, Any]]: List of chunks with metadata
        """
        chunks_with_metadata = []
        
        for i, chunk in enumerate(chunks, 1):
            chunk_data = {
                "text": chunk,
                "metadata": {
                    "source": source,
                    "chunk_index": i,
                    "chunk_size": len(chunk),
                    "word_count": len(chunk.split())
                }
            }
            chunks_with_metadata.append(chunk_data)
        
        return chunks_with_metadata