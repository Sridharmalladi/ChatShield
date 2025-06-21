#!/usr/bin/env python3
"""
Test script for ChatShield - Secure Document Chatbot
This script tests the core functionality with sample documents
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from document_processor import DocumentProcessor
from vector_store import SecureVectorStore
from rag_engine import SecureRAGEngine

def setup_test_environment():
    """Set up test environment with sample documents"""
    print("🔧 Setting up test environment...")
    
    # Create temporary directory for test
    test_dir = tempfile.mkdtemp()
    print(f"Test directory: {test_dir}")
    
    # Copy sample documents
    sample_docs = [
        "test_documents/sample_strategy.txt",
        "test_documents/employee_handbook.txt"
    ]
    
    for doc in sample_docs:
        if os.path.exists(doc):
            shutil.copy(doc, test_dir)
            print(f"Copied {doc} to test directory")
        else:
            print(f"Warning: {doc} not found")
    
    return test_dir

def test_document_processing(test_dir):
    """Test document processing functionality"""
    print("\n📄 Testing document processing...")
    
    processor = DocumentProcessor()
    
    # Process sample documents
    processed_chunks = []
    for file_path in Path(test_dir).glob("*.txt"):
        try:
            chunks = processor.process_document(str(file_path))
            processed_chunks.extend(chunks)
            print(f"✅ Processed {file_path.name}: {len(chunks)} chunks")
            
            # Show access level distribution
            manager_chunks = [c for c in chunks if c['access_level'] == 'Manager']
            employer_chunks = [c for c in chunks if c['access_level'] == 'Employer']
            print(f"   - Manager access: {len(manager_chunks)} chunks")
            print(f"   - Employer access: {len(employer_chunks)} chunks")
            
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")
    
    return processed_chunks

def test_vector_store(processed_chunks):
    """Test vector store functionality"""
    print("\n🗄️ Testing vector store...")
    
    # Create vector store
    vector_store = SecureVectorStore("test_vector_store")
    
    # Add documents
    if processed_chunks:
        vector_store.add_documents(processed_chunks)
        print(f"✅ Added {len(processed_chunks)} chunks to vector store")
    
    # Get stats
    stats = vector_store.get_document_stats()
    print(f"📊 Vector store stats: {stats}")
    
    return vector_store

def test_rag_engine(vector_store):
    """Test RAG engine functionality"""
    print("\n🧠 Testing RAG engine...")
    
    rag_engine = SecureRAGEngine(vector_store)
    
    # Test queries for different roles
    test_queries = [
        ("What is the company vision?", "Employer"),
        ("What is the company vision?", "Manager"),
        ("What are the hiring plans?", "Employer"),
        ("What are the hiring plans?", "Manager"),
        ("What are the financial projections?", "Employer"),
        ("What are the financial projections?", "Manager"),
        ("What are the employee benefits?", "Employer"),
        ("What are the employee benefits?", "Manager"),
    ]
    
    for query, role in test_queries:
        print(f"\n🔍 Testing: '{query}' (Role: {role})")
        
        # Get query analysis
        analysis = rag_engine.get_query_analysis(query, role)
        print(f"   Analysis: {analysis}")
        
        # Get context preview
        preview = rag_engine.get_context_preview(query, role, max_chunks=2)
        print(f"   Context preview: {len(preview)} chunks available")
        
        # Test actual query (if not blocked)
        if not analysis['blocked']:
            result = rag_engine.query(query, role)
            print(f"   Response: {result['answer'][:100]}...")
            print(f"   Sources: {result['sources']}")
        else:
            print(f"   ❌ Query blocked: {analysis['reason']}")

def test_access_control():
    """Test access control functionality"""
    print("\n🔐 Testing access control...")
    
    # Test sensitive topic detection
    sensitive_queries = [
        "What is the CEO salary?",
        "What are the internal forecasts?",
        "What are the hiring plans?",
        "What are the financial projections?",
    ]
    
    for query in sensitive_queries:
        is_sensitive = config.is_topic_sensitive(query)
        employer_access = config.can_access_topic("Employer", query)
        manager_access = config.can_access_topic("Manager", query)
        
        print(f"Query: '{query}'")
        print(f"  - Sensitive: {is_sensitive}")
        print(f"  - Employer access: {employer_access}")
        print(f"  - Manager access: {manager_access}")

def cleanup_test_environment(test_dir):
    """Clean up test environment"""
    print(f"\n🧹 Cleaning up test environment...")
    try:
        shutil.rmtree(test_dir)
        shutil.rmtree("test_vector_store", ignore_errors=True)
        print("✅ Cleanup completed")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

def main():
    """Main test function"""
    print("🚀 Starting ChatShield Tests")
    print("=" * 50)
    
    # Check if OpenAI API key is set
    if not config.openai_api_key or config.openai_api_key == "your_openai_api_key_here":
        print("❌ Error: OpenAI API key not set. Please set OPENAI_API_KEY environment variable.")
        print("You can copy env_example.txt to .env and add your API key.")
        return
    
    try:
        # Setup
        test_dir = setup_test_environment()
        
        # Test document processing
        processed_chunks = test_document_processing(test_dir)
        
        if not processed_chunks:
            print("❌ No documents were processed successfully. Exiting.")
            return
        
        # Test vector store
        vector_store = test_vector_store(processed_chunks)
        
        # Test RAG engine
        test_rag_engine(vector_store)
        
        # Test access control
        test_access_control()
        
        print("\n✅ All tests completed successfully!")
        print("\n🎯 To run the full application:")
        print("   streamlit run app.py")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        cleanup_test_environment(test_dir)

if __name__ == "__main__":
    main() 