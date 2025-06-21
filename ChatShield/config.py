import os
import yaml
from dotenv import load_dotenv
from typing import Dict, List, Any

# Load environment variables
load_dotenv()

class Config:
    def __init__(self):
        self.load_guardrails()
        self.load_env_vars()
    
    def load_guardrails(self):
        """Load guardrails configuration from YAML file"""
        with open('guardrails.yaml', 'r') as file:
            self.guardrails = yaml.safe_load(file)
    
    def load_env_vars(self):
        """Load environment variables"""
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.model_name = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        self.embedding_model = os.getenv('OPENAI_EMBEDDING_MODEL', 'text-embedding-ada-002')
    
    def get_sensitive_topics(self) -> List[str]:
        """Get list of sensitive topics"""
        return self.guardrails.get('sensitive_topics', [])
    
    def get_access_levels(self) -> Dict[str, Any]:
        """Get access level configurations"""
        return self.guardrails.get('access_levels', {})
    
    def get_document_processing_config(self) -> Dict[str, Any]:
        """Get document processing configuration"""
        return self.guardrails.get('document_processing', {})
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration"""
        return self.guardrails.get('security', {})
    
    def is_topic_sensitive(self, query: str) -> bool:
        """Check if a query contains sensitive topics"""
        query_lower = query.lower()
        sensitive_topics = self.get_sensitive_topics()
        return any(topic.lower() in query_lower for topic in sensitive_topics)
    
    def can_access_topic(self, user_role: str, query: str) -> bool:
        """Check if user can access a specific topic"""
        access_levels = self.get_access_levels()
        
        if user_role not in access_levels:
            return False
        
        role_config = access_levels[user_role]
        
        # Manager has access to all topics
        if role_config.get('can_access') == 'all':
            return True
        
        # Check if query contains restricted topics for Employer
        if user_role == 'Employer':
            restricted_topics = role_config.get('restricted_topics', [])
            query_lower = query.lower()
            return not any(topic.lower() in query_lower for topic in restricted_topics)
        
        return False

# Global config instance
config = Config() 