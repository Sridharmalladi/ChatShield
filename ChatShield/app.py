import streamlit as st
import os
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Any
import time

# Import our modules
from config import config
from document_processor import DocumentProcessor
from vector_store import SecureVectorStore
from rag_engine import SecureRAGEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize session state
if 'user_role' not in st.session_state:
    st.session_state.user_role = None
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = []

def initialize_components():
    """Initialize vector store and RAG engine"""
    if st.session_state.vector_store is None:
        st.session_state.vector_store = SecureVectorStore()
    if st.session_state.rag_engine is None:
        st.session_state.rag_engine = SecureRAGEngine(st.session_state.vector_store)

def login_page():
    """Display login page with ID and password."""
    st.title("ChatShield")
    st.markdown("### Secure Document Chatbot - Please log in to continue")

    with st.form("login_form"):
        user_id = st.text_input("User ID", placeholder="e.g., E001 or M001")
        password = st.text_input("Password", type="password", placeholder="e.g., employee or manager")
        submitted = st.form_submit_button("Login")

        if submitted:
            # Validate credentials
            if user_id == "E001" and password == "employee":
                st.session_state.user_role = "Employer"
                st.success("Logged in as Employer!")
                time.sleep(1)
                st.rerun()
            elif user_id == "M001" and password == "manager":
                st.session_state.user_role = "Manager"
                st.success("Logged in as Manager!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Invalid User ID or Password")

def document_upload_section():
    """Document upload and processing section for the sidebar."""
    st.header("📄 Document Upload")
    
    uploaded_files = st.file_uploader(
        "Upload (PDF, DOCX, TXT)",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Upload documents to be processed and made available for querying"
    )
    
    if uploaded_files:
        if st.button("🔄 Process Documents", type="primary"):
            process_documents(uploaded_files)

def process_documents(uploaded_files):
    """Process uploaded documents and update vector store."""
    processor = DocumentProcessor()
    
    with st.spinner("Processing documents... This may take a moment."):
        status_placeholder = st.empty()
        processed_chunks = []
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_placeholder.text(f"Processing: {uploaded_file.name}...")
            
            try:
                # Save uploaded file temporarily to process it
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                chunks = processor.process_document(tmp_file_path)
                processed_chunks.extend(chunks)
                
                os.unlink(tmp_file_path) # Clean up the temporary file
                if uploaded_file.name not in st.session_state.documents_processed:
                    st.session_state.documents_processed.append(uploaded_file.name)
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        
        if processed_chunks:
            st.session_state.vector_store.add_documents(processed_chunks)
            st.success(f"✅ Processed {len(uploaded_files)} documents.")
        
        status_placeholder.empty()

def chat_interface():
    """Main chat interface with modern design."""
    col1, col2 = st.columns([5, 1])
    with col1:
        st.header("💬 Secure Chat")
    with col2:
        if st.button("Reset 🔄"):
            st.session_state.chat_history = []
            st.rerun()

    # Display chat history in a scrollable container
    with st.container():
        display_chat_history()

    # Use st.chat_input for a non-looping chat experience
    if prompt := st.chat_input("Ask a question..."):
        process_query(prompt)

def display_chat_history():
    """Display chat history in a modern, scrollable chat interface."""
    if not st.session_state.chat_history:
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #888;">
            <h3>👋 Welcome!</h3>
            <p>Ask a question about your documents or any general topic.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    chat_html = ""
    for chat in st.session_state.chat_history:
        role = chat.get('role')
        message = chat['content']

        if role == 'user':
            chat_html += f'<div style="text-align: right; margin-bottom: 10px;"><div style="display: inline-block; background-color: #007bff; color: white; padding: 10px 15px; border-radius: 15px 15px 0 15px;">{message}</div></div>'
        elif role == 'assistant':
            # The 'response' object is stored at the time of creation
            response = chat.get('response', {}) 
            bg_color = "#f8f9fa"
            color = "#333"
            if response.get('blocked'):
                bg_color = "#dc3545"
                color = "white"
            elif response.get('response_type') == 'general_knowledge':
                bg_color = "#28a745"
                color = "white"

            chat_html += f'<div style="text-align: left; margin-bottom: 10px;"><div style="display: inline-block; background-color: {bg_color}; color: {color}; padding: 10px 15px; border-radius: 15px 15px 15px 0;">{message}</div></div>'

    st.markdown(f'<div id="chat-box" style="height: 50vh; overflow-y: auto; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">{chat_html}</div>', unsafe_allow_html=True)

    # JavaScript to scroll to the bottom of the chat box
    st.components.v1.html("""
        <script>
            var chatBox = document.getElementById('chat-box');
            if (chatBox) {
                chatBox.scrollTop = chatBox.scrollHeight;
            }
        </script>
    """, height=0)

def process_query(query: str):
    """Process user query and update chat history."""
    # Append user message in the new standardized format
    st.session_state.chat_history.append({'role': 'user', 'content': query})

    with st.spinner("Thinking..."):
        # Pass the existing chat history to the RAG engine
        result = st.session_state.rag_engine.query(
            query, 
            st.session_state.user_role,
            chat_history=st.session_state.chat_history[:-1] # Pass history BEFORE the current query
        )

    # Append assistant response with metadata
    assistant_message = {
        'role': 'assistant', 
        'content': result['answer'], 
        'response': result # Store the full response for styling/metadata
    }
    st.session_state.chat_history.append(assistant_message)
    st.rerun()

def main():
    """Main application layout and logic."""
    st.set_page_config(page_title="ChatShield - Secure Document Chatbot", page_icon="🔐", layout="wide", initial_sidebar_state="expanded")
    
    # --- Custom CSS ---
    st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
        }
        .stDeployButton {
            display: none;
        }
    </style>
    """, unsafe_allow_html=True)
    
    initialize_components()

    # --- Login Gate ---
    if not st.session_state.user_role:
        login_page()
        return

    # --- Main App Layout (if logged in) ---
    with st.sidebar:
        st.title("🔐 ChatShield")
        st.success(f"Logged in as: **{st.session_state.user_role}**")
        if st.button("Logout"):
            st.session_state.user_role = None
            st.session_state.chat_history = []
            st.rerun()
        
        st.divider()
        document_upload_section()

    # --- Main Chat Area ---
    chat_interface()

if __name__ == "__main__":
    main() 