#!/usr/bin/env python3
"""
Clinical Guideline PDF to JSON Processor
Converts clinical guideline PDFs into structured JSON with boolean logic
"""

import os
import json
import re
import time
from typing import Dict, Any, List
from pathlib import Path
import pymupdf  # PyMuPDF
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

load_dotenv()


class GuidelineRequirement(BaseModel):
    """Pydantic model for individual requirements"""
    id: str = Field(description="Unique requirement ID (e.g., req1, req2)")
    description: str = Field(description="Detailed description of the requirement")


class ClinicalGuideline(BaseModel):
    """Pydantic model for the complete clinical guideline structure"""
    subject: str = Field(description="Subject of the guideline")
    guideline: str = Field(description="Guideline number/identifier")
    publish_date: str = Field(description="Publication date")
    status: str = Field(description="Current status")
    last_review_date: str = Field(description="Last review date")
    requirements: Dict[str, str] = Field(description="Dictionary of requirement IDs and descriptions")
    decision_logic: str = Field(description="Boolean logic expression using requirement IDs")


class ClinicalGuidelineProcessor:
    """Main processor class for converting clinical guideline PDFs to structured JSON"""
    
    def __init__(self):
        """Initialize the processor with Groq API"""
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        # Initialize Groq LLM
        self.llm = ChatGroq(
            temperature=0,
            groq_api_key=self.groq_api_key,
            model_name="llama-3.1-8b-instant"  # You can change this to other Groq models
        )
        
        # Set up JSON output parser
        self.parser = JsonOutputParser(pydantic_object=ClinicalGuideline)
        
        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            ("human", "Please analyze this clinical guideline text and convert it to the specified JSON format:\n\n{text}")
        ])
        
        # Create the chain
        self.chain = self.prompt | self.llm
    
    def _get_system_prompt(self) -> str:
        """Generate the system prompt for the LLM"""
        return """You are an expert medical informatics specialist. Your task is to analyze clinical guideline documents and convert them into structured JSON format with boolean logic.

INSTRUCTIONS:
1. Extract the guideline metadata (subject, guideline number, dates, status)
2. Identify ALL medical criteria for when treatment is "Medically Necessary"
3. Assign unique IDs (req1, req2, req3, etc.) to each distinct criterion
4. Convert the criteria relationships into boolean logic using AND, OR, NOT operators
5. Pay special attention to nested conditions and sub-requirements

BOOLEAN LOGIC RULES:
- Use req1, req2, etc. as variables in the logic expression
- Use AND for conditions that must ALL be met
- Use OR for conditions where ANY one can be met
- Use parentheses to group related conditions
- For continued therapy requirements, they should be ANDed with initial requirements

EXAMPLE OUTPUT FORMAT:
{{
  "subject": "Treatment Name",
  "guideline": "Guideline-ID", 
  "publish_date": "MM/DD/YYYY",
  "status": "Status",
  "last_review_date": "MM/DD/YYYY",
  "requirements": {{
    "req1": "First requirement description",
    "req2": "Second requirement description",
    "req3": "Third requirement description"
  }},
  "decision_logic": "req1 OR req2 OR (req3 AND req4)"
}}

Focus on the "Medically Necessary" section and ignore "Not Medically Necessary" conditions.
Be precise with the boolean logic - ensure it accurately represents the clinical decision tree.

You must return valid JSON only. Do not include any explanatory text before or after the JSON."""

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = pymupdf.open(pdf_path)
            text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
                text += "\n\n"  # Add page breaks
            
            doc.close()
            return text.strip()
            
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess the extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (simple approach)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip likely page numbers (standalone numbers)
            if line.isdigit() and len(line) <= 3:
                continue
            # Skip very short lines that are likely formatting artifacts
            if len(line) < 10:
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def process_pdf(self, pdf_path: str, max_retries: int = 3) -> Dict[str, Any]:
        """Main method to process a PDF and return structured JSON"""
        try:
            # Extract text from PDF
            print(f"Extracting text from: {pdf_path}")
            raw_text = self.extract_text_from_pdf(pdf_path)
            
            # Clean the text
            print("Cleaning extracted text...")
            cleaned_text = self.clean_text(raw_text)
            
            # Limit text length to avoid token limits
            max_chars = 25000  # Adjust based on model limits
            if len(cleaned_text) > max_chars:
                print(f"Text too long ({len(cleaned_text)} chars), truncating to {max_chars} chars...")
                cleaned_text = cleaned_text[:max_chars]
            
            # Process with LLM with retry logic
            print("Processing with Groq API...")
            for attempt in range(max_retries):
                try:
                    response = self.chain.invoke({"text": cleaned_text})
                    
                    # Handle response content
                    if hasattr(response, 'content'):
                        response_text = response.content
                    else:
                        response_text = str(response)
                    
                    # Parse JSON manually with better error handling
                    result = self._parse_json_response(response_text)
                    return result
                    
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        print(f"Retrying in {2 ** attempt} seconds...")
                        time.sleep(2 ** attempt)  # Exponential backoff
                    else:
                        raise e
            
        except Exception as e:
            raise Exception(f"Error processing PDF: {str(e)}")
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response with error handling and cleanup"""
        try:
            # Try direct JSON parsing first
            return json.loads(response_text)
        except json.JSONDecodeError:
            # Clean up common issues with LLM JSON output
            cleaned_response = self._clean_json_response(response_text)
            try:
                return json.loads(cleaned_response)
            except json.JSONDecodeError as e:
                print(f"JSON parsing failed. Response was: {response_text[:500]}...")
                raise Exception(f"Failed to parse JSON response: {str(e)}")
    
    def _clean_json_response(self, response_text: str) -> str:
        """Clean up common JSON formatting issues from LLM responses"""
        # Remove any text before the first {
        start_index = response_text.find('{')
        if start_index != -1:
            response_text = response_text[start_index:]
        
        # Remove any text after the last }
        end_index = response_text.rfind('}')
        if end_index != -1:
            response_text = response_text[:end_index + 1]
        
        # Fix common issues
        response_text = response_text.replace('\n', ' ')  # Remove newlines
        response_text = re.sub(r'\s+', ' ', response_text)  # Normalize whitespace
        
        return response_text.strip()
    
    def save_json(self, data: Dict[str, Any], output_path: str) -> None:
        """Save the result to a JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"Results saved to: {output_path}")
        except Exception as e:
            raise Exception(f"Error saving JSON: {str(e)}")
    
    def validate_result(self, result: Dict[str, Any]) -> bool:
        """Basic validation of the output structure"""
        required_fields = ["subject", "guideline", "requirements", "decision_logic"]
        
        for field in required_fields:
            if field not in result:
                print(f"Warning: Missing required field '{field}'")
                return False
        
        if not isinstance(result["requirements"], dict):
            print("Warning: 'requirements' should be a dictionary")
            return False
        
        if not result["requirements"]:
            print("Warning: No requirements found")
            return False
        
        print(f"✓ Validation passed. Found {len(result['requirements'])} requirements")
        return True


def process_single_pdf(pdf_path: str, output_path: str = None):
    """Process a single PDF file"""
    try:
        # Initialize the processor
        processor = ClinicalGuidelineProcessor()
        
        # Process the PDF
        print(f"Processing: {pdf_path}")
        result = processor.process_pdf(pdf_path)
        
        # Print summary
        print("\n" + "="*50)
        print("PROCESSING RESULTS")
        print("="*50)
        print(f"Subject: {result.get('subject', 'N/A')}")
        print(f"Guideline: {result.get('guideline', 'N/A')}")
        print(f"Status: {result.get('status', 'N/A')}")
        print(f"Requirements found: {len(result.get('requirements', {}))}")
        
        # Show first few requirements
        requirements = result.get('requirements', {})
        if requirements:
            print("\nFirst 3 requirements:")
            for i, (req_id, description) in enumerate(list(requirements.items())[:3]):
                print(f"  {req_id}: {description[:100]}...")
        
        print(f"\nDecision Logic: {result.get('decision_logic', 'N/A')}")
        print("="*50)
        
        # Save if output path provided
        if output_path:
            processor.save_json(result, output_path)
        
        return result
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")
        return None
    
