import pdfplumber
import docx
import spacy
import nltk
import re
from typing import List, Dict, Any, Tuple
from pathlib import Path
import logging
from config import config

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If model not available, use basic tokenization
    nlp = None

class DocumentProcessor:
    def __init__(self):
        self.processing_config = config.get_document_processing_config()
        self.chunk_size = self.processing_config.get('chunk_size', 1000)
        self.chunk_overlap = self.processing_config.get('chunk_overlap', 200)
        
        # Keywords for access level classification
        self.manager_keywords = [
            'confidential', 'internal', 'strategy', 'forecast', 'budget',
            'hiring', 'layoff', 'compensation', 'executive', 'board',
            'financial', 'revenue', 'profit', 'acquisition', 'merger',
            'restructuring', 'performance', 'target', 'goal', 'planning',
            'salary'
        ]
        
        self.employer_keywords = [
            'general', 'public', 'announcement', 'update', 'news',
            'policy', 'procedure', 'guideline', 'information', 'overview'
        ]
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            logging.error(f"Error extracting text from PDF {file_path}: {e}")
            raise
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logging.error(f"Error extracting text from DOCX {file_path}: {e}")
            raise
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            logging.error(f"Error extracting text from TXT {file_path}: {e}")
            raise
    
    def extract_text(self, file_path: str) -> str:
        """Extract text from various file formats"""
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.pdf':
            return self.extract_text_from_pdf(str(file_path))
        elif file_path.suffix.lower() == '.docx':
            return self.extract_text_from_docx(str(file_path))
        elif file_path.suffix.lower() == '.txt':
            return self.extract_text_from_txt(str(file_path))
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks using NLP-aware boundaries"""
        if nlp:
            return self._chunk_with_spacy(text)
        else:
            return self._chunk_with_nltk(text)
    
    def _chunk_with_spacy(self, text: str) -> List[str]:
        """Chunk text using spaCy for better sentence boundaries"""
        doc = nlp(text)
        chunks = []
        current_chunk = ""
        
        for sent in doc.sents:
            sent_text = sent.text.strip()
            if len(current_chunk) + len(sent_text) <= self.chunk_size:
                current_chunk += " " + sent_text if current_chunk else sent_text
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sent_text
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _chunk_with_nltk(self, text: str) -> List[str]:
        """Chunk text using NLTK sentence tokenization"""
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def classify_access_level(self, chunk: str) -> str:
        """Classify chunk access level based on content analysis."""
        chunk_lower = chunk.lower()
        
        # Priority rule: If 'benefits' is in the text, it's for Employees
        # unless it's clearly about financial strategy.
        if 'benefits' in chunk_lower:
            manager_override_keywords = ['confidential', 'budget', 'strategy', 'forecast', 'acquisition', 'financial']
            if not any(keyword in chunk_lower for keyword in manager_override_keywords):
                return "Employer"

        # Fallback to original keyword scoring for all other cases
        manager_score = sum(1 for keyword in self.manager_keywords if keyword in chunk_lower)
        employer_score = sum(1 for keyword in self.employer_keywords if keyword in chunk_lower)
        
        # Additional heuristics
        if any(word in chunk_lower for word in ['confidential', 'internal', 'private']):
            manager_score += 3
        
        if any(word in chunk_lower for word in ['public', 'announcement', 'general']):
            employer_score += 2
        
        # Check for financial/revenue numbers
        if re.search(r'\\$\\d+', chunk) or re.search(r'\\d+%', chunk):
            manager_score += 2
        
        # Check for executive names or titles
        if re.search(r'\b(CEO|CFO|CTO|VP|Director|Manager)\b', chunk, re.IGNORECASE):
            manager_score += 1
        
        # Decision logic
        if manager_score > employer_score:
            return "Manager"
        else:
            return "Employer"
    
    def process_document(self, file_path: str) -> List[Dict[str, Any]]:
        """Process document and return chunks with metadata"""
        try:
            # Extract text
            text = self.extract_text(file_path)
            text = self.clean_text(text)
            
            # Chunk text
            chunks = self.chunk_text(text)
            
            # Limit chunks per document
            max_chunks = self.processing_config.get('max_chunks_per_document', 50)
            if len(chunks) > max_chunks:
                chunks = chunks[:max_chunks]
            
            # Process chunks with metadata
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                access_level = self.classify_access_level(chunk)
                
                processed_chunk = {
                    'content': chunk,
                    'access_level': access_level,
                    'source': Path(file_path).name,
                    'chunk_id': i,
                    'chunk_size': len(chunk)
                }
                
                processed_chunks.append(processed_chunk)
            
            logging.info(f"Processed {len(processed_chunks)} chunks from {file_path}")
            return processed_chunks
            
        except Exception as e:
            logging.error(f"Error processing document {file_path}: {e}")
            raise 