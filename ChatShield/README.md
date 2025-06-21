# ChatShield - Secure Document Chatbot with RAG Architecture

A secure, role-based document chatbot that uses Retrieval-Augmented Generation (RAG) with strict access control. ChatShield ensures that users only see information they're authorized to access, preventing data leaks and maintaining security boundaries.

## 🏗️ Architecture Overview

```
        ┌──────────────┐
        │ PDF / DOCX   │
        └─────┬────────┘
              ▼
 ┌────────────────────────────┐
 │ NLP Parsing & Chunking     │ ← spaCy / NLTK
 │   + Access Tagging         │ ← Employer / Manager
 └────┬─────────────┬─────────┘
      ▼             ▼
Access = Employer  Access = Manager
      ▼             ▼
   Stored in      Stored in
  Vector DB       Vector DB
      └────┬────────────┘
           ▼
    Access-Controlled LLM
           ▼
     Response to User
```

## ✨ Key Features

### 🔐 Role-Based Access Control
- **Manager**: Full access to all document content
- **Employer**: Limited access to general information only
- Automatic content classification based on sensitivity

### 📄 Document Processing
- Support for PDF, DOCX, and TXT files
- NLP-powered text chunking using spaCy/NLTK
- Intelligent access level tagging
- Vector storage with FAISS

### 🛡️ Security Features
- Query validation and filtering
- Sensitive topic detection
- Secure LLM prompting
- Access control at document chunk level

### 💬 Smart Chat Interface
- Real-time query analysis
- Context preview before answering
- Chat history with source tracking
- Beautiful Streamlit UI

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install spaCy Model (Optional but Recommended)

```bash
python -m spacy download en_core_web_sm
```

### 3. Set Up Environment Variables

Copy `env_example.txt` to `.env` and add your OpenAI API key:

```bash
cp env_example.txt .env
```

Edit `.env`:
```
OPENAI_API_KEY=your_actual_openai_api_key_here
```

### 4. Run the Application

```bash
streamlit run app.py
```

## 📖 Usage Guide

### 1. Role Selection
- Choose your role (Manager or Employer) at login
- Different roles have different access levels

### 2. Document Upload
- Upload PDF, DOCX, or TXT files
- System automatically processes and classifies content
- Documents are chunked and tagged with access levels

### 3. Secure Chatting
- Ask questions about your documents
- System shows context preview before answering
- Only authorized content is used for responses

### 4. Admin Functions
- Clear all documents
- Reset session
- View system statistics

## 🔧 Configuration

### Guardrails Configuration (`guardrails.yaml`)

```yaml
sensitive_topics:
  - ceo salary
  - internal forecast
  - hiring plans

access_levels:
  Manager:
    can_access: all
  Employer:
    can_access: []
    restricted_topics:
      - hiring
      - financial
```

### Document Processing Settings

```yaml
document_processing:
  chunk_size: 1000
  chunk_overlap: 200
  max_chunks_per_document: 50
```

## 🧠 How It Works

### 1. Document Processing Pipeline
1. **Text Extraction**: Extract text from PDF/DOCX/TXT files
2. **NLP Chunking**: Split text into semantic chunks using spaCy/NLTK
3. **Access Classification**: Tag each chunk as "Manager" or "Employer" based on content
4. **Vector Storage**: Store chunks with embeddings in FAISS vector database

### 2. Query Processing Pipeline
1. **Query Validation**: Check for sensitive topics and access permissions
2. **Vector Search**: Find relevant chunks based on semantic similarity
3. **Access Filtering**: Only return chunks the user can access
4. **Secure LLM**: Pass filtered context to LLM with strict prompting
5. **Response Generation**: Generate answer based only on authorized content

### 3. Security Measures
- **Content Classification**: Automatic detection of sensitive information
- **Query Filtering**: Block queries about restricted topics
- **Context Isolation**: LLM only sees authorized content
- **Prompt Engineering**: Secure system prompts prevent information leakage

## 📊 Example Workflow

### Manager Uploads Strategy Document
```
Chunk 1: "Company Vision" → Employer access
Chunk 2: "Hiring Forecast: 50 new engineers" → Manager access
Chunk 3: "Financial Projections: $10M revenue" → Manager access
```

### Employer Queries
- **"What is the company vision?"** ✅ Allowed
- **"What is the hiring forecast?"** ❌ Blocked (sensitive topic)
- **"What are the financial projections?"** ❌ Blocked (not in Employer chunks)

### Manager Queries
- **"What is the company vision?"** ✅ Allowed
- **"What is the hiring forecast?"** ✅ Allowed (sees Manager chunks)
- **"What are the financial projections?"** ✅ Allowed (sees Manager chunks)

## 🛠️ Technical Stack

| Component | Technology |
|-----------|------------|
| **UI Framework** | Streamlit |
| **NLP Processing** | spaCy, NLTK |
| **Vector Database** | FAISS |
| **LLM** | OpenAI GPT-3.5/4 |
| **Embeddings** | OpenAI text-embedding-ada-002 |
| **Document Parsing** | pdfplumber, python-docx |
| **Configuration** | YAML |

## 🔒 Security Features

### Access Control
- Role-based document access
- Chunk-level security tagging
- Query-level permission checking

### Data Protection
- No unauthorized content exposure
- Secure LLM prompting
- Input validation and sanitization

### Audit Trail
- Query logging
- Access level tracking
- Source attribution

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── config.py             # Configuration management
├── document_processor.py # Document processing and NLP
├── vector_store.py       # FAISS vector database
├── rag_engine.py         # RAG query processing
├── guardrails.yaml       # Security configuration
├── requirements.txt      # Python dependencies
├── env_example.txt       # Environment variables template
└── README.md            # This file
```

## 🧪 Testing

### Test Scenarios
1. **Manager Access**: Should see all content
2. **Employer Access**: Should see only Employer-tagged content
3. **Sensitive Topics**: Should be blocked for unauthorized users
4. **Document Processing**: Should correctly classify access levels

### Sample Documents
Create test documents with mixed content:
- General company information (Employer access)
- Financial data (Manager access)
- Hiring plans (Manager access)
- Public announcements (Employer access)

## 🚨 Troubleshooting

### Common Issues

1. **spaCy Model Not Found**
   ```bash
   python -m spacy download en_core_web_sm
   ```

2. **OpenAI API Key Error**
   - Ensure `.env` file exists with correct API key
   - Check API key permissions

3. **Memory Issues with Large Documents**
   - Reduce `chunk_size` in `guardrails.yaml`
   - Limit `max_chunks_per_document`

4. **Slow Processing**
   - Use smaller chunk sizes
   - Reduce overlap between chunks
   - Consider using GPU for FAISS

## 🔮 Future Enhancements

- [ ] Multi-language support
- [ ] Advanced role hierarchies
- [ ] Document versioning
- [ ] Real-time collaboration
- [ ] Advanced analytics dashboard
- [ ] Integration with enterprise SSO
- [ ] Custom embedding models
- [ ] Document comparison features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the configuration files
3. Open an issue on GitHub

---

**Built with ❤️ for secure document management by ChatShield** 