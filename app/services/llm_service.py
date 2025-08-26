from typing import List, Dict, Any
from app.core.config import config
from app.core.logger import setup_logger
from langchain.schema import HumanMessage

logger = setup_logger(__name__)

class LLMService:
    """
    Service responsible for interacting with Google Gemini models to generate responses.
    
    This service is specifically designed to work with Google's Gemma models,
    providing optimized prompt formatting and response generation for these models.
    """
    
    def __init__(self):
        """
        Initialize the LLM service with Google Gemini configuration.
        """
        self.model = config.LLM_MODEL
        self.api_key = config.LLM_API_KEY
        
        # Initialize the Google Gemini model
        self.llm = self._initialize_gemini_model()
    
    def _initialize_gemini_model(self):
        """
        Initialize the Google Gemini model.
        
        Returns:
            Initialized Google Gemini model
            
        Raises:
            Exception: If model initialization fails
        """
        logger.info(f"Initializing Google Gemini model: {self.model}")
        
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=self.model,
                google_api_key=self.api_key,
                temperature=0.2
            )
        except ImportError:
            logger.error("langchain-google-genai is not installed. Run: pip install langchain-google-genai")
            raise
        except Exception as e:
            logger.error(f"Error initializing Google Gemini model: {str(e)}")
            raise
    
    def generate_answer(self, question: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a response to the question based on the provided context chunks.
        
        Args:
            question (str): User's question
            context_chunks (List[Dict[str, Any]]): Relevant context chunks
            
        Returns:
            Dict[str, Any]: Dictionary containing the answer and references
            
        Raises:
            Exception: If an error occurs during response generation
        """
        logger.info(f"Generating response for question: {question}")
        try:
            # Prepare context by concatenating chunk texts
            context = "\n\n".join([chunk["text"] for chunk in context_chunks])
            
            # Build the prompt optimized for Google Gemini
            prompt = self._build_gemini_prompt(question, context)
            
            # Generate response using Google Gemini
            messages = [HumanMessage(content=prompt)]
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # Extract references (original chunk texts)
            references = [chunk["text"] for chunk in context_chunks]
            
            logger.info("Response generated successfully")
            return {
                "answer": response_text.strip(),
                "references": references
            }
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise
    
    def _build_gemini_prompt(self, question: str, context: str) -> str:
        """
        Build a prompt optimized for Google Gemini models.
        
        Args:
            question (str): User's question
            context (str): Context text
            
        Returns:
            str: Formatted prompt optimized for Google Gemini
        """
        return f"""
        You are an assistant specialized in answering questions based on provided documents.
        
        Follow these instructions carefully:
        1. Use ONLY the information explicitly present in the given context below.
        2. If the answer cannot be found in the context, explicitly reply: "I don't know based on the available documents."
        3. Be clear, concise, and objective.
        4. Whenever possible, cite the specific passages or document references that support your answer.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        ANSWER:
        """