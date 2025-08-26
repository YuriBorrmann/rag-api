#!/usr/bin/env python3
"""
Test script for the RAG system validation.
Uploads a document and tests questions against expected answers.
"""
import requests
import time
import os
import re
from typing import Dict

# API endpoints
DOCUMENTS_URL = "http://localhost:8000/api/documents"
QUESTION_URL = "http://localhost:8000/api/question"

# Test document path
DOCUMENT_PATH = "files/LB5001.pdf"

# Test questions and expected answers
TEST_QUESTIONS = [
    {
        "question": "What should be done if damage is found when receiving the motor?",
        "expected_answer": "Report any damage immediately to the commercial carrier that delivered the motor."
    },
    {
        "question": "Who is allowed to install the motor according to the safety notice?",
        "expected_answer": "Only qualified personnel trained in the safe installation and operation of the equipment should install the motor."
    },
    {
        "question": "Can open drip proof (ODP) motors be used in areas with flammable materials?",
        "expected_answer": "No. ODP motors should not be used in the presence of flammable or combustible materials, as they can emit flame or molten metal in the event of insulation failure."
    },
    {
        "question": "What type of foundation is recommended for foot-mounted machines?",
        "expected_answer": "They should be mounted to a rigid foundation to prevent excessive vibration. Shims may be used if the location is uneven."
    },
    {
        "question": "What is the recommended action if a motor does not start quickly and smoothly?",
        "expected_answer": "Stop the motor immediately and determine the cause. Possible causes include low voltage at the motor, incorrect connections, or the load being too heavy."
    },
    {
        "question": "What is the normal lubricant used in Baldor motors at the factory?",
        "expected_answer": "Polyrex EM (Exxon Mobil)."
    },
    {
        "question": "How often should a shaft grounding brush assembly be replaced on a motor running at 1800 RPM?",
        "expected_answer": "Every 44,000 hours."
    },
    {
        "question": "For a motor with frame size over 210 to 280 (NEMA), operating at 1800 RPM, what is the relubrication interval?",
        "expected_answer": "9,500 hours."
    },
    {
        "question": "How much grease should be added for a motor with frame size over 360 to 5000 (NEMA)?",
        "expected_answer": "2.12 ounces (60 grams), which equals 4.1 cubic inches or 13.4 teaspoons."
    },
    {
        "question": "What precaution should be taken when regreasing a motor?",
        "expected_answer": "Too much grease or injecting grease too quickly can cause premature bearing failure. Grease should be applied slowly, taking about 1 minute."
    }
]

def check_file_exists(file_path: str) -> bool:
    """
    Check if the file exists and get information about it.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        print(f"   Current working directory: {os.getcwd()}")
        print(f"   Please make sure the file exists in the correct location.")
        return False
    
    if not os.path.isfile(file_path):
        print(f"❌ Error: Path is not a file: {file_path}")
        return False
    
    if not os.access(file_path, os.R_OK):
        print(f"❌ Error: File is not readable: {file_path}")
        return False
    
    # Get file info
    file_size = os.path.getsize(file_path)
    file_ext = os.path.splitext(file_path)[1].lower()
    
    print(f"📄 File information:")
    print(f"   Path: {os.path.abspath(file_path)}")
    print(f"   Size: {file_size} bytes")
    print(f"   Extension: {file_ext}")
    print(f"   Is readable: {os.access(file_path, os.R_OK)}")
    
    return True

def upload_document(file_path: str) -> bool:
    """
    Upload a document to the RAG system.
    """
    print(f"\n📤 Uploading document: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'files': (os.path.basename(file_path), f, 'application/pdf')}
            response = requests.post(DOCUMENTS_URL, files=files)
        
        print(f"   Status: {response.status_code}")
        print(f"   Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Document uploaded successfully")
            print(f"   Documents indexed: {result.get('documents_indexed', 0)}")
            print(f"   Total chunks: {result.get('total_chunks', 0)}")
            return True
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def ask_question(question: str) -> Dict:
    """
    Ask a question to the RAG system.
    """
    try:
        payload = {"question": question}
        headers = {"Content-Type": "application/json"}
        
        print(f"\n🤔 Asking question: {question[:50]}...")
        response = requests.post(QUESTION_URL, json=payload, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Question answered successfully")
            return result
        else:
            print(f"❌ Question request failed with status code: {response.status_code}")
            print(f"   Response: {response.text}")
            return {"answer": "ERROR", "references": []}
            
    except Exception as e:
        print(f"❌ Error asking question: {e}")
        return {"answer": "ERROR", "references": []}

def evaluate_answer(generated_answer: str, expected_answer: str) -> Dict:
    """
    Evaluate a generated answer against the expected answer using Google Gemini model.
    """
    # Import the Google Gemini model
    from langchain_google_genai import ChatGoogleGenerativeAI
    from app.core.config import config
    
    # Initialize the model for evaluation
    evaluation_model = ChatGoogleGenerativeAI(
        model=config.LLM_MODEL,
        google_api_key=config.LLM_API_KEY,
        temperature=0.2  # Low temperature for consistent evaluation
    )
    
    # Construct the evaluation prompt
    prompt = f"""
    You are an expert evaluator for a RAG (Retrieval-Augmented Generation) system. Your task is to compare a generated answer with an expected answer and determine if the generated answer is satisfactory.
    
    Expected Answer: "{expected_answer}"
    
    Generated Answer: "{generated_answer}"
    
    Please evaluate the generated answer based on the following criteria:
    1. Accuracy: Does the generated answer provide the same key information as the expected answer?
    2. Completeness: Does the generated answer cover all the important points from the expected answer?
    3. Correctness: Is the information in the generated answer correct and not misleading?
    
    Provide your evaluation in the following format:
    Score: [A number from 0 to 10, where 0 is completely wrong and 10 is perfect]
    Reasoning: [A brief explanation of your scoring]
    
    Consider that minor differences in wording or additional information that doesn't contradict the expected answer should not significantly lower the score.
    """
    
    try:
        # Get the model's evaluation
        response = evaluation_model.invoke(prompt)
        evaluation_text = response.content
        
        # Parse the response to extract score and reasoning
        score_match = re.search(r'Score:\s*(\d+(?:\.\d+)?)', evaluation_text)
        if score_match:
            score = float(score_match.group(1))
        else:
            # If we can't find the score line, try to find any number in the text
            numbers = re.findall(r'\b\d+(?:\.\d+)?\b', evaluation_text)
            score = float(numbers[0]) if numbers else 0.0
        
        # Extract reasoning
        reasoning_match = re.search(r'Reasoning:\s*(.*?)(?:\n\n|$)', evaluation_text, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).strip()
        else:
            # If we can't find the reasoning line, use the entire response
            reasoning = evaluation_text
        
        # Determine if the answer is acceptable (score >= 6.0)
        is_acceptable = score >= 6.0
        
        return {
            "score": score / 10.0,  # Normalize to 0-1 scale
            "is_acceptable": is_acceptable,
            "reasoning": reasoning,
            "raw_evaluation": evaluation_text
        }
        
    except Exception as e:
        print(f"Warning: Model evaluation failed ({e}), falling back to keyword-based evaluation")
        
        # Fallback to keyword-based evaluation
        expected_keywords = expected_answer.lower().split()
        generated_lower = generated_answer.lower()
        
        # Count how many expected keywords appear in the generated answer
        matches = sum(1 for keyword in expected_keywords if keyword in generated_lower and len(keyword) > 3)
        
        # Calculate a simple score
        total_keywords = len([k for k in expected_keywords if len(k) > 3])
        score = matches / total_keywords if total_keywords > 0 else 0
        
        # Determine if the answer is acceptable (score > 0.5)
        is_acceptable = score > 0.5
        
        return {
            "score": score,
            "is_acceptable": is_acceptable,
            "reasoning": f"Fallback evaluation: {matches}/{total_keywords} keywords matched",
            "raw_evaluation": "Fallback to keyword matching due to model evaluation failure"
        }

def run_tests():
    """
    Run the complete test suite.
    """
    print("=" * 80)
    print("RAG SYSTEM VALIDATION TEST")
    print("=" * 80)
    
    # Step 1: Check if file exists
    if not check_file_exists(DOCUMENT_PATH):
        return False
    
    # Step 2: Upload the document
    if not upload_document(DOCUMENT_PATH):
        print("❌ Cannot proceed without successful document upload")
        return False
    
    print("\n" + "-" * 80)
    print("TESTING QUESTIONS")
    print("-" * 80)
    
    # Step 3: Test each question
    results = []
    
    for i, test_case in enumerate(TEST_QUESTIONS, 1):
        question = test_case["question"]
        expected_answer = test_case["expected_answer"]
        
        print(f"\n📝 Question {i}: {question}")
        
        # Ask the question
        response = ask_question(question)
        generated_answer = response.get("answer", "")
        references = response.get("references", [])
        
        # Evaluate the answer
        evaluation = evaluate_answer(generated_answer, expected_answer)
        
        # Store results
        results.append({
            "question": question,
            "expected_answer": expected_answer,
            "generated_answer": generated_answer,
            "evaluation": evaluation,
            "references_count": len(references)
        })
        
        # Print results
        status = "✅" if evaluation["is_acceptable"] else "❌"
        print(f"{status} Score: {evaluation['score']:.2f} ({evaluation['score']*10:.1f}/10)")
        print(f"   Expected: {expected_answer}")
        print(f"   Generated: {generated_answer[:100]}{'...' if len(generated_answer) > 100 else ''}")
        print(f"   Reasoning: {evaluation['reasoning'][:100]}{'...' if len(evaluation['reasoning']) > 100 else ''}")
        print(f"   References: {len(references)} chunks used")
        
        # Small delay to avoid overwhelming the API
        time.sleep(1)  # Increased delay for model evaluation
    
    # Step 4: Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    acceptable_count = sum(1 for r in results if r["evaluation"]["is_acceptable"])
    total_count = len(results)
    success_rate = (acceptable_count / total_count) * 100 if total_count > 0 else 0
    
    print(f"✅ Acceptable answers: {acceptable_count}/{total_count} ({success_rate:.1f}%)")
    print(f"❌ Unacceptable answers: {total_count - acceptable_count}/{total_count} ({100 - success_rate:.1f}%)")
    
    # Calculate average score
    avg_score = sum(r["evaluation"]["score"] for r in results) / total_count if total_count > 0 else 0
    print(f"📊 Average score: {avg_score:.2f}")
    
    # Print detailed results for unacceptable answers
    if acceptable_count < total_count:
        print("\n❌ FAILED QUESTIONS:")
        for i, result in enumerate(results, 1):
            if not result["evaluation"]["is_acceptable"]:
                print(f"\n   Question {i}: {result['question']}")
                print(f"   Expected: {result['expected_answer']}")
                print(f"   Generated: {result['generated_answer']}")
                print(f"   Score: {result['evaluation']['score']:.2f}")
                print(f"   Reasoning: {result['evaluation']['reasoning']}")
    
    # Overall result
    print("\n" + "=" * 80)
    if success_rate >= 80:
        print("🎉 TEST PASSED: System is working well!")
    elif success_rate >= 60:
        print("⚠️  TEST PARTIALLY PASSED: System needs improvement.")
    else:
        print("❌ TEST FAILED: System needs significant improvement.")
    print("=" * 80)
    
    # Return overall success
    return success_rate >= 80

if __name__ == "__main__":
    # Run the tests
    success = run_tests()
    exit(0 if success else 1)