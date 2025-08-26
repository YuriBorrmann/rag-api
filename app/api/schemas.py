from typing import List
from pydantic import BaseModel, Field, validator

class QuestionRequest(BaseModel):
    """
    Schema for question requests.
    
    This schema defines the structure for incoming question requests
    to the RAG system API. It includes validation for the question field.
    """
    
    question: str = Field(
        ...,
        description="The user's question about the uploaded documents",
        min_length=1,
        max_length=1000,
        example="What is the power consumption of the motor?"
    )
    
    @validator('question')
    def validate_question(cls, v):
        """
        Validate the question field.
        
        Args:
            v: The question string to validate
            
        Returns:
            The validated question string
            
        Raises:
            ValueError: If the question is empty or contains only whitespace
        """
        if not v.strip():
            raise ValueError("Question cannot be empty or contain only whitespace")
        return v.strip()

class QuestionResponse(BaseModel):
    """
    Schema for question responses.
    
    This schema defines the structure for responses from the RAG system.
    It includes the generated answer and references to source documents.
    """
    
    answer: str = Field(
        ...,
        description="The generated answer to the user's question",
        min_length=1,
        example="The motor's power consumption is 2.3 kW."
    )
    
    references: List[str] = Field(
        default_factory=list,
        description="List of source text chunks used to generate the answer",
        example=[
            "the motor xxx has requires 2.3kw to operate at a 60hz line frequency"
        ]
    )
    
    @validator('references')
    def validate_references(cls, v):
        """
        Validate the references field.
        
        Args:
            v: The list of references to validate
            
        Returns:
            The validated list of references
            
        Raises:
            ValueError: If any reference is empty or contains only whitespace
        """
        validated_refs = []
        for ref in v:
            if ref.strip():  # Only include non-empty references
                validated_refs.append(ref.strip())
        return validated_refs
    
    class Config:
        """
        Configuration class for the Pydantic model.
        """
        json_schema_extra = {
            "example": {
                "answer": "The motor's power consumption is 2.3 kW.",
                "references": [
                    "the motor xxx has requires 2.3kw to operate at a 60hz line frequency"
                ]
            }
        }