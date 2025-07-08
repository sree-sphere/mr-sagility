"""
Clinical Requirements Evaluation Agent using Groq Llama 3.1 70B Versatile

Required dependencies:
pip install langchain langchain-groq chromadb pymupdf python-dotenv

Setup:
1. Create a .env file in your project directory with:
   GROQ_API_KEY=your_groq_api_key_here

2. Ensure you have the clinical_requirements.json file in your project directory
3. Have PDF evidence documents in the "evidence" directory
"""

import json
import re
import os
from typing import Dict, List, Any, Tuple, Optional
import logging
from dataclasses import dataclass
from dotenv import load_dotenv

# LangChain imports
from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType
from langchain.schema import BaseOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import BaseTool
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain_groq import ChatGroq

# Import our previous ChromaDB ingester
from vector_db_utils import PDFChromaIngester
from mr_parser import process_single_pdf

# Ingest once to produce clinical_requirements.json
process_single_pdf(
    pdf_path="/Users/raj/WaveITLabs/sagility/Clinical UM Guideline.pdf",
    output_path='clinical_requirements.json'
)

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RequirementEvaluation:
    """Data class to store requirement evaluation results"""
    requirement_id: str
    requirement_text: str
    evidence_found: bool
    evidence_summary: str
    confidence_score: float
    sources: List[str]

class ChromaDBSearchTool(BaseTool):
    """Custom LangChain tool for searching ChromaDB"""
    
    name: str = "chromadb_search"
    description: str = "Search for evidence in the medical document database. Input should be a medical condition or treatment description."
    ingester: PDFChromaIngester
    
    def __init__(self, ingester: PDFChromaIngester):
        super().__init__(ingester=ingester)
    
    def _run(
        self, 
        query: str, 
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        """Execute the search and return raw results plus a formatted string for the agent"""
        try:
            results = self.ingester.search_similar_documents(query, n_results=3)
            
            raw_hits = results.get('results', [])
            if not raw_hits:
                formatted = f"No evidence found for: {query}"
                return {"formatted": formatted, "hits": []}
            
            # Format the results for the LLM agent, but also keep raw hits for source extraction
            formatted_results = []
            for i, hit in enumerate(raw_hits, 1):
                formatted_results.append(
                    f"Result {i}:\n"
                    f"Source: {hit['metadata']['filename']}\n"
                    f"Page: {hit['metadata']['page']}\n"
                    f"Content: {hit['document'][:300]}...\n"
                )
            formatted = f"Found {len(raw_hits)} relevant documents:\n\n" + "\n".join(formatted_results)
            
            return {"formatted": formatted, "hits": raw_hits}
            
        except Exception as e:
            err = f"Error searching database: {str(e)}"
            return {"formatted": err, "hits": []}
    
    async def _arun(
        self, 
        query: str, 
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> Dict[str, Any]:
        """Async version of the tool"""
        return self._run(query, run_manager)

class ClinicalRequirementsAgent:
    """AI Agent for evaluating clinical requirements against evidence"""
    
    def __init__(self, chroma_ingester: PDFChromaIngester):
        """
        Initialize the clinical requirements agent
        
        Args:
            chroma_ingester: Initialized ChromaDB ingester instance
        """
        self.chroma_ingester = chroma_ingester
        
        # Get Groq API key from environment
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
        
        # Initialize the language model with Groq Llama 3.1 70B Versatile
        self.llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,  # Low temperature for consistency
            max_tokens=2048   # Reasonable limit for responses
        )
        
        # Create the ChromaDB search tool
        self.search_tool = ChromaDBSearchTool(chroma_ingester)
        
        # Create the evaluation prompt (optimized for Llama models)
        self.evaluation_prompt = PromptTemplate(
            input_variables=["requirement", "search_results"],
            template="""
You are a medical evidence evaluator analyzing clinical documentation. Your task is to determine if sufficient evidence exists for a specific medical requirement.

MEDICAL REQUIREMENT TO EVALUATE:
{requirement}

SEARCH RESULTS FROM MEDICAL DOCUMENTS:
{search_results}

INSTRUCTIONS:
1. Carefully analyze the search results for evidence supporting the medical requirement
2. Look for direct mentions, case studies, treatment protocols, or related medical conditions
3. Consider both explicit and implicit evidence
4. If no relevant evidence is found, clearly state this

You must respond using this EXACT format (do not deviate):

EVIDENCE_FOUND: YES
SUMMARY: Brief explanation of the evidence found and how it supports the requirement
CONFIDENCE: 0.85

OR

EVIDENCE_FOUND: NO
SUMMARY: Explanation of why no supporting evidence was found
CONFIDENCE: 0.90

Remember:
- Use only YES or NO for EVIDENCE_FOUND
- Keep SUMMARY concise but informative
- CONFIDENCE should be between 0.0 and 1.0
- Base your assessment solely on the provided search results
"""
        )
        
        # Create the evaluation chain
        self.evaluation_chain = LLMChain(
            llm=self.llm,
            prompt=self.evaluation_prompt
        )
    
    def load_requirements(self, json_path: str) -> Dict[str, Any]:
        """Load clinical requirements from JSON file"""
        try:
            with open(json_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            logger.error(f"Error loading requirements: {str(e)}")
            raise
    
    def evaluate_single_requirement(self, req_id: str, req_text: str) -> RequirementEvaluation:
        """
        Evaluate a single requirement against the evidence database
        
        Args:
            req_id: Requirement identifier (e.g., 'req1')
            req_text: Requirement description
            
        Returns:
            RequirementEvaluation object with results, including source files
        """
        logger.info(f"Evaluating requirement {req_id}: {req_text}")
        
        # Search for evidence (gets both formatted text and raw hits)
        result_payload = self.search_tool._run(req_text)
        search_str = result_payload["formatted"]
        raw_hits = result_payload["hits"]
        
        # Extract unique source filenames
        sources = list({hit['metadata']['filename'] for hit in raw_hits})
        
        # Use LLM to evaluate the evidence
        evaluation_response = self.evaluation_chain.run(
            requirement=req_text,
            search_results=search_str
        )
        
        # Parse the response
        evidence_found = "YES" in evaluation_response.split("EVIDENCE_FOUND:")[1].split("\n")[0].strip().upper()
        
        summary_match = re.search(r"SUMMARY:\s*(.+?)(?=\nCONFIDENCE:|$)", evaluation_response, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else "No summary provided"
        
        confidence_match = re.search(r"CONFIDENCE:\s*([0-9.]+)", evaluation_response)
        confidence = float(confidence_match.group(1)) if confidence_match else 0.0
        
        return RequirementEvaluation(
            requirement_id=req_id,
            requirement_text=req_text,
            evidence_found=evidence_found,
            evidence_summary=summary,
            confidence_score=confidence,
            sources=sources
        )
    
    def evaluate_all_requirements(self, requirements_data: Dict[str, Any]) -> Dict[str, RequirementEvaluation]:
        """
        Evaluate all requirements in the dataset
        
        Args:
            requirements_data: Loaded JSON requirements data
            
        Returns:
            Dictionary mapping requirement IDs to evaluation results
        """
        evaluations = {}
        requirements = requirements_data.get('requirements', {})
        
        for req_id, req_text in requirements.items():
            evaluation = self.evaluate_single_requirement(req_id, req_text)
            evaluations[req_id] = evaluation
            
            logger.info(
                f"{req_id}: {'✓' if evaluation.evidence_found else '✗'} "
                f"(Confidence: {evaluation.confidence_score:.2f})"
            )
        
        return evaluations
    
    def apply_decision_logic(
        self,
        evaluations: Dict[str, RequirementEvaluation], 
        decision_logic: str
    ) -> Tuple[bool, str]:
        """
        Apply the boolean decision logic to the requirement evaluations
        
        Args:
            evaluations: Dictionary of requirement evaluations
            decision_logic: Boolean logic expression from JSON
            
        Returns:
            Tuple of (final_decision, explanation)
        """
        # Create a mapping of requirement IDs to their boolean values
        req_values = {
            req_id: eval_result.evidence_found 
            for req_id, eval_result in evaluations.items()
        }
        
        # Replace requirement IDs in the logic with their boolean values
        logic_expression = decision_logic
        for req_id in sorted(req_values, key=len, reverse=True):
            logic_expression = logic_expression.replace(req_id, str(req_values[req_id]))
        
        # Replace logical operators
        logic_expression = logic_expression.replace(" OR ", " or ")
        logic_expression = logic_expression.replace(" AND ", " and ")
        
        try:
            # Evaluate the boolean expression
            final_decision = eval(logic_expression)
            
            # Generate explanation
            true_requirements = [req_id for req_id, result in evaluations.items() if result.evidence_found]
            false_requirements = [req_id for req_id, result in evaluations.items() if not result.evidence_found]
            
            # Build per-requirement lines including their sources
            true_lines = []
            for req in true_requirements:
                srcs = evaluations[req].sources
                true_lines.append(f"{req} ({', '.join(srcs) if srcs else 'None'})")

            false_lines = []
            for req in false_requirements:
                srcs = evaluations[req].sources
                false_lines.append(f"{req} ({', '.join(srcs) if srcs else 'None'})")

            explanation = f"""
Decision Logic: {decision_logic}
Evaluated Expression: {logic_expression}
Final Decision: {'APPROVED' if final_decision else 'DENIED'}

Requirements with Evidence Found ({len(true_lines)}):
{chr(10).join(true_lines)}

Requirements without Evidence ({len(false_lines)}):
{chr(10).join(false_lines)}
""".strip()
            
            return final_decision, explanation.strip()
            
        except Exception as e:
            logger.error(f"Error evaluating decision logic: {str(e)}")
            return False, f"Error in decision logic evaluation: {str(e)}"
    
    def generate_detailed_report(
        self,
        requirements_data: Dict[str, Any], 
        evaluations: Dict[str, RequirementEvaluation],
        final_decision: bool, 
        explanation: str
    ) -> str:
        """Generate a detailed evaluation report including sources"""
        
        report = f"""
=== CLINICAL REQUIREMENTS EVALUATION REPORT ===

Subject: {requirements_data.get('subject', 'Unknown')}
Guideline: {requirements_data.get('guideline', 'Unknown')}
Publish Date: {requirements_data.get('publish_date', 'Unknown')}
Evaluation Date: {requirements_data.get('last_review_date', 'Unknown')}

FINAL DECISION: {'✓ APPROVED' if final_decision else '✗ DENIED'}

=== DETAILED REQUIREMENT ANALYSIS ===
"""
        
        for req_id, evaluation in evaluations.items():
            status = "✓ EVIDENCE FOUND" if evaluation.evidence_found else "✗ NO EVIDENCE"
            report += f"""
{req_id}: {status} (Confidence: {evaluation.confidence_score:.2f})
Requirement: {evaluation.requirement_text}
Evidence Summary: {evaluation.evidence_summary}
Sources: {', '.join(evaluation.sources) if evaluation.sources else 'None'}
{'='*60}
"""
        
        report += f"""
=== DECISION LOGIC EVALUATION ===
{explanation}

=== SUMMARY ===
Total Requirements: {len(evaluations)}
Requirements with Evidence: {sum(1 for e in evaluations.values() if e.evidence_found)}
Requirements without Evidence: {sum(1 for e in evaluations.values() if not e.evidence_found)}
Average Confidence: {sum(e.confidence_score for e in evaluations.values()) / len(evaluations):.2f}
"""
        
        return report
    
    def run_full_evaluation(self, json_path: str) -> str:
        """
        Run the complete evaluation process
        
        Args:
            json_path: Path to the clinical requirements JSON file
            
        Returns:
            Detailed evaluation report
        """
        logger.info("Starting clinical requirements evaluation...")
        
        # Load requirements
        requirements_data = self.load_requirements(json_path)
        
        # Evaluate all requirements
        evaluations = self.evaluate_all_requirements(requirements_data)
        
        # Apply decision logic
        decision_logic = requirements_data.get('decision_logic', '')
        final_decision, explanation = self.apply_decision_logic(evaluations, decision_logic)
        
        # Generate detailed report
        report = self.generate_detailed_report(
            requirements_data, evaluations, final_decision, explanation
        )
        
        logger.info(f"Evaluation complete. Final decision: {'APPROVED' if final_decision else 'DENIED'}")
        
        return report


def main():
    """Example usage of the Clinical Requirements Agent"""
    
    # Configuration
    CHROMA_DB_PATH = "./chroma_db"
    EVIDENCE_DIR = "evidence"
    JSON_PATH = "clinical_requirements.json"
    
    try:
        # Initialize ChromaDB ingester
        logger.info("Initializing ChromaDB...")
        chroma_ingester = PDFChromaIngester(chroma_db_path=CHROMA_DB_PATH)
        
        # Check if we need to ingest documents
        collection_info = chroma_ingester.get_collection_info()
        if collection_info.get('document_count', 0) == 0:
            logger.info("No documents found in ChromaDB. Ingesting documents...")
            chroma_ingester.ingest_pdfs_from_directory(EVIDENCE_DIR)
        else:
            logger.info(f"Found {collection_info['document_count']} documents in ChromaDB")
        
        # Initialize the clinical agent
        logger.info("Initializing Clinical Requirements Agent with Groq Llama 3.1 70B...")
        agent = ClinicalRequirementsAgent(chroma_ingester)
        
        # Run the evaluation
        logger.info("Running clinical requirements evaluation...")
        report = agent.run_full_evaluation(JSON_PATH)
        
        # Print the report
        print(report)
        
        # Optionally save the report
        with open("evaluation_report.txt", "w") as f:
            f.write(report)
        logger.info("Report saved to evaluation_report.txt")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()
