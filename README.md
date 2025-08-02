# Clinical Representation Evaluator
A set of tools for evaluating clinical requirements against medical evidence using a custom parser, embedding-based similarity matching, and retrieval over a vector database.

## Architecture Components
1. `mr_parser.py` # PDF to JSON converter
- Parses structured meaning representations from raw inputs into a canonical form with boolean logic expressions.

**Key features:**
- Parses intents, slot-value pairs, and operators.
- Cleans and standardizes string-based representations into machine-readable formats.
- Extracts both flat and nested slot structures.
- Supports custom operator tokens (`IN`, `GT`, `LT`, etc.).
- Supports logical decision trees with AND/OR/NOT operations

**Output Structure:**
```json
{
  "subject": "Treatment Name",
  "guideline": "Guideline-ID",
  "publish_date": "MM/DD/YYYY",
  "status": "Status",
  "last_review_date": "MM/DD/YYYY",
  "requirements": {
    "req1": "First requirement description",
    "req2": "Second requirement description"
  },
  "decision_logic": "req1 OR req2 OR (req3 AND req4)"
}
```

2. `vector_db_utils.py` # ChromaDB utilities
- Handles storage and retrieval from a vector similarity index on ChromaDB.
- Chunks documents for optimal retrieval (configurable chunk size and overlap)
- Uses FAISS for approximate nearest neighbor search.
- Loads and queries vector indices based on precomputed sentence embeddings.
- Supports normalized cosine similarity-based ranking.

3. `evaluator.py` # Main evaluation engine
- Evaluates model outputs (Eg: generated or predicted meaning representations) against a reference set.
- Compares MR predictions using semantic similarity via embeddings.
- Optionally computes top-k accuracy and retrieval precision from the vector index.
- Allows customization of evaluation metrics and retrieval thresholds.

## Workflow
1. **Parsing**: Raw MR strings are parsed using MRParser into a structured dictionary format.
2. **Embedding & Indexing**: Parsed MRs or natural language forms are embedded and stored in a FAISS index for fast similarity-based lookup.
3. **Evaluation**: A model's predicted MR is parsed and compared semantically against ground-truth references using similarity scoring or top-k retrieval.

```
         +-----------------+
         | Raw MR String   |
         +--------+--------+
                  |
                  v
         +--------+--------+
         |  MR Parser      |  ---> Structured dict
         +--------+--------+
                  |
                  v
       +----------+-----------+
       |  Sentence Embedding  |  (Eg: SBERT)
       +----------+-----------+
                  |
                  v
         +--------+--------+
         |  FAISS Vector DB |
         +--------+--------+
                  |
        +---------+---------+
        |   MR Evaluator    |
        +-------------------+

```

## Usage Highlights
1. Designed to benchmark MR generation models where exact-match metrics are insufficient.
2. Embedding-based evaluation captures paraphrastic and structural similarity.
3. Modular design allows flexible parser and retriever configurations.

### Required dependencies:
pip install langchain langchain-groq chromadb pymupdf python-dotenv

### Setup:
1. Create a .env file in your project directory with:
   GROQ_API_KEY
2. Ensure you have the clinical_requirements.json file in your project directory
3. Have PDF source documents in the "evidence" directory
4. Run CLI command `python evaluator.py`