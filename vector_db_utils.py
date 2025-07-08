
import os
import fitz  # PyMuPDF
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFChromaIngester:
    def __init__(self, chroma_db_path: str = "./chroma_db", collection_name: str = "pdf_documents"):
        """
        Initialize the PDF ingester with ChromaDB setup.
        
        Args:
            chroma_db_path: Path where ChromaDB will store its data
            collection_name: Name of the collection to store documents
        """
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=chroma_db_path)
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(name=collection_name)
            logger.info(f"Created new collection: {collection_name}")

    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract text from PDF file using PyMuPDF.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of dictionaries containing page text and metadata
        """
        documents = []
        
        try:
            # Open the PDF
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # Skip empty pages
                if text.strip():
                    documents.append({
                        'text': text,
                        'metadata': {
                            'source': pdf_path,
                            'page': page_num + 1,
                            'filename': os.path.basename(pdf_path)
                        }
                    })
            
            doc.close()
            logger.info(f"Extracted {len(documents)} pages from {pdf_path}")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            
        return documents

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """
        Split text into overlapping chunks for better retrieval.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # If this isn't the last chunk, try to break at a word boundary
            if end < len(text):
                # Look for the last space within the chunk
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunks.append(text[start:end])
            start = end - overlap
            
            # Avoid infinite loop
            if start >= end:
                start = end
                
        return chunks

    def ingest_pdfs_from_directory(self, evidence_dir: str, chunk_documents: bool = True):
        """
        Ingest all PDF files from the evidence directory into ChromaDB.
        
        Args:
            evidence_dir: Path to the directory containing PDF files
            chunk_documents: Whether to split documents into smaller chunks
        """
        if not os.path.exists(evidence_dir):
            logger.error(f"Evidence directory not found: {evidence_dir}")
            return
        
        pdf_files = [f for f in os.listdir(evidence_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {evidence_dir}")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        all_documents = []
        all_metadatas = []
        all_ids = []
        
        for pdf_file in pdf_files:
            pdf_path = os.path.join(evidence_dir, pdf_file)
            logger.info(f"Processing: {pdf_file}")
            
            # Extract text from PDF
            extracted_docs = self.extract_text_from_pdf(pdf_path)
            
            for doc in extracted_docs:
                if chunk_documents:
                    # Split into chunks
                    chunks = self.chunk_text(doc['text'])
                    
                    for i, chunk in enumerate(chunks):
                        chunk_id = f"{doc['metadata']['filename']}_page_{doc['metadata']['page']}_chunk_{i}"
                        
                        all_documents.append(chunk)
                        all_metadatas.append({
                            **doc['metadata'],
                            'chunk_id': i,
                            'total_chunks': len(chunks)
                        })
                        all_ids.append(chunk_id)
                else:
                    # Use full page text
                    page_id = f"{doc['metadata']['filename']}_page_{doc['metadata']['page']}"
                    
                    all_documents.append(doc['text'])
                    all_metadatas.append(doc['metadata'])
                    all_ids.append(page_id)
        
        # Add documents to ChromaDB collection
        if all_documents:
            self.collection.add(
                documents=all_documents,
                metadatas=all_metadatas,
                ids=all_ids
            )
            logger.info(f"Successfully ingested {len(all_documents)} document chunks into ChromaDB")
        else:
            logger.warning("No documents to ingest")

    def search_similar_documents(self, query: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Search for documents similar to the given query.
        
        Args:
            query: Search query string
            n_results: Number of results to return
            
        Returns:
            Dictionary containing search results
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Format results for better readability
            formatted_results = []
            
            if results['documents'] and results['documents'][0]:
                for i in range(len(results['documents'][0])):
                    result = {
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None,
                        'id': results['ids'][0][i]
                    }
                    formatted_results.append(result)
            
            return {
                'query': query,
                'results': formatted_results,
                'total_results': len(formatted_results)
            }
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}")
            return {'query': query, 'results': [], 'total_results': 0, 'error': str(e)}

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the current collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                'collection_name': self.collection_name,
                'document_count': count,
                'db_path': self.chroma_db_path
            }
        except Exception as e:
            logger.error(f"Error getting collection info: {str(e)}")
            return {'error': str(e)}
        
# # Initialize ingester
# ingester = PDFChromaIngester(chroma_db_path="./my_chroma_db", collection_name="legal_docs")

# # Ingest documents
# ingester.ingest_pdfs_from_directory("evidence")

# # Search for similar documents
# search_results = ingester.search_similar_documents("Chronic refractory osteomyelitis", n_results=2)

# # Print results
# for result in search_results['results']:
#     print(f"File: {result['metadata']['filename']}")
#     print(f"Page: {result['metadata']['page']}")
#     print(f"Content: {result['document']}...\n")