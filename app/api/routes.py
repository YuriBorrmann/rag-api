import os
import tempfile
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.retrieval_service import RetrievalService
from app.services.llm_service import LLMService
from app.api.schemas import QuestionRequest, QuestionResponse
from app.core.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

router = APIRouter()

# Initialize services
document_processor = DocumentProcessor()
embedding_service = EmbeddingService()
retrieval_service = RetrievalService(embedding_service)
llm_service = LLMService()

# Constants
ALLOWED_MIME_TYPE = "application/pdf"
ERROR_MESSAGES = {
    "invalid_file_type": "Only PDF files are allowed",
    "no_relevant_docs": "No relevant documents were found to answer the question.",
    "processing_error": "Error processing documents",
    "question_error": "Error processing question"
}

@router.get("/")
async def root():
    """Root endpoint to check if API is running"""
    return {"message": "RAG System API is running", "status": "healthy"}

@router.post("/documents")
async def upload_documents(files: List[UploadFile] = File(...)):
    """
    Endpoint for uploading PDF documents.
    
    Processes the documents by:
    1. Validating file types
    2. Saving files temporarily
    3. Extracting text and creating chunks
    4. Generating embeddings
    5. Storing in vector index
    6. Cleaning up temporary files
    
    Args:
        files: List of PDF files to upload
        
    Returns:
        JSONResponse with processing results
        
    Raises:
        HTTPException: If file validation fails or processing error occurs
    """
    temp_dir = None
    try:
        logger.info(f"Starting upload of {len(files)} documents")
        
        # Validate and save files temporarily
        temp_dir = tempfile.mkdtemp()
        file_paths = await _save_temp_files(files, temp_dir)
        
        # Process documents and create embeddings
        chunks_with_metadata = document_processor.process_documents(file_paths)
        embedding_service.create_index(chunks_with_metadata)
        
        logger.info(f"Successfully processed {len(files)} documents into {len(chunks_with_metadata)} chunks")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Documents processed successfully",
                "documents_indexed": len(files),
                "total_chunks": len(chunks_with_metadata)
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=ERROR_MESSAGES["processing_error"])
    finally:
        # Ensure cleanup happens even if an error occurs
        if temp_dir:
            await _cleanup_temp_files(temp_dir)

async def _save_temp_files(files: List[UploadFile], temp_dir: str) -> List[str]:
    """
    Save uploaded files to temporary directory.
    
    Args:
        files: List of uploaded files
        temp_dir: Temporary directory path
        
    Returns:
        List of file paths
        
    Raises:
        HTTPException: If file type is invalid
    """
    file_paths = []
    
    for file in files:
        if file.content_type != ALLOWED_MIME_TYPE:
            logger.warning(f"Invalid file type: {file.content_type}")
            raise HTTPException(status_code=400, detail=ERROR_MESSAGES["invalid_file_type"])
        
        file_path = os.path.join(temp_dir, file.filename)
        try:
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            file_paths.append(file_path)
            logger.debug(f"Saved temporary file: {file_path}")
        except Exception as e:
            logger.error(f"Error saving file {file.filename}: {str(e)}")
            raise
    
    return file_paths

async def _cleanup_temp_files(temp_dir: str) -> None:
    """
    Clean up temporary files and directory.
    
    Args:
        temp_dir: Temporary directory path
    """
    try:
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                os.remove(file_path)
                logger.debug(f"Removed temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to remove file {file_path}: {str(e)}")
        
        try:
            os.rmdir(temp_dir)
            logger.debug(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to remove directory {temp_dir}: {str(e)}")
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")

@router.post("/question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Endpoint for asking questions about indexed documents.
    
    Process:
    1. Receives user question
    2. Retrieves relevant document chunks
    3. Generates answer using LLM
    4. Returns answer with source references
    
    Args:
        request: QuestionRequest containing the user's question
        
    Returns:
        QuestionResponse with answer and references
        
    Raises:
        HTTPException: If question processing fails
    """
    try:
        question = request.question
        logger.info(f"Processing question: {question}")
        
        # Retrieve relevant chunks
        relevant_chunks = retrieval_service.retrieve_relevant_chunks(question)
        logger.debug(f"Retrieved {len(relevant_chunks)} relevant chunks")
        
        if not relevant_chunks:
            logger.info("No relevant documents found for the question")
            return {
                "answer": ERROR_MESSAGES["no_relevant_docs"],
                "references": []
            }
        
        # Generate response using LLM
        response = llm_service.generate_answer(question, relevant_chunks)
        logger.info("Successfully generated answer")
        
        return response
    except Exception as e:
        logger.error(f"Error processing question: {str(e)}")
        raise HTTPException(status_code=500, detail=ERROR_MESSAGES["question_error"])