#!/usr/bin/env python3
"""
Setup script for ChatShield - Secure Document Chatbot
Helps users install dependencies and configure the system
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def install_spacy_model():
    """Install spaCy English model"""
    print("\n🧠 Installing spaCy English model...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy model installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Warning: Could not install spaCy model: {e}")
        print("The system will use NLTK as fallback")
        return False

def setup_environment():
    """Set up environment variables"""
    print("\n🔧 Setting up environment...")
    
    env_file = Path(".env")
    env_example = Path("env_example.txt")
    
    if env_file.exists():
        print("✅ .env file already exists")
        return True
    
    if env_example.exists():
        try:
            shutil.copy(env_example, env_file)
            print("✅ Created .env file from template")
            print("⚠️ Please edit .env file and add your OpenAI API key")
            return True
        except Exception as e:
            print(f"❌ Error creating .env file: {e}")
            return False
    else:
        print("❌ env_example.txt not found")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = ["vector_store", "test_documents"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def test_imports():
    """Test if all modules can be imported"""
    print("\n🧪 Testing imports...")
    
    modules = [
        "streamlit",
        "openai",
        "langchain",
        "faiss",
        "spacy",
        "nltk",
        "pdfplumber",
        "docx",
        "yaml"
    ]
    
    failed_imports = []
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️ Failed to import: {', '.join(failed_imports)}")
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 ChatShield Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed: Could not install dependencies")
        return
    
    # Install spaCy model
    install_spacy_model()
    
    # Setup environment
    if not setup_environment():
        print("❌ Setup failed: Could not setup environment")
        return
    
    # Create directories
    create_directories()
    
    # Test imports
    if not test_imports():
        print("❌ Setup failed: Import test failed")
        return
    
    print("\n✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit .env file and add your OpenAI API key")
    print("2. Run: python test_system.py (to test the system)")
    print("3. Run: streamlit run app.py (to start the application)")
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main() 