import logging
from typing import List, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from config import config
from vector_store import SecureVectorStore

class SecureRAGEngine:
    def __init__(self, vector_store: SecureVectorStore):
        self.vector_store = vector_store
        self.llm = ChatOpenAI(
            model=config.model_name,
            openai_api_key=config.openai_api_key,
            temperature=0.1,
            max_tokens=1000
        )
        self.security_config = config.get_security_config()
    
    def _validate_query(self, query: str, user_role: str) -> Dict[str, Any]:
        """Validate query for security and access control"""
        validation_result = {
            'valid': True,
            'reason': None,
            'blocked': False
        }
        
        # Check query length
        max_length = self.security_config.get('max_query_length', 500)
        if len(query) > max_length:
            validation_result['valid'] = False
            validation_result['reason'] = f"Query too long. Maximum length is {max_length} characters."
            validation_result['blocked'] = True
            return validation_result
        
        # Check for sensitive topics
        if config.is_topic_sensitive(query):
            if not config.can_access_topic(user_role, query):
                validation_result['valid'] = False
                validation_result['reason'] = "Access Denied."
                validation_result['blocked'] = True
                return validation_result
        
        return validation_result
    
    def _create_system_prompt(self, user_role: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Create a flexible system prompt for the LLM."""
        
        context_text = "\\n\\n---\\n\\n".join([chunk['content'] for chunk in context_chunks]) if context_chunks else "No relevant documents found."
        
        system_prompt = f"""You are a helpful AI assistant for a user with the role '{user_role}'. Your task is to answer the user's question based on the rules below.

First, review the following document excerpts to see if they are relevant to the user's question.

--- DOCUMENT EXCERPTS ---
{context_text}
--- END EXCERPTS ---

**Instructions:**
1.  **Analyze Relevance:** First, determine if the document excerpts are directly relevant to the user's question.
2.  **Document-Based Answer:** If the excerpts are clearly relevant and contain the answer, you MUST base your answer exclusively on them.
3.  **General Knowledge Answer:** In all other cases (e.g., excerpts are not relevant, or they are relevant but don't contain the specific answer), you MUST ignore the excerpts and answer using your general knowledge.
4.  **Do Not Mention Documents:** Never refer to "the documents" or "the excerpts provided." Just provide the best, most direct answer.
"""
        return system_prompt
    
    def _create_user_prompt(self, query: str) -> str:
        """Create the user prompt for the query"""
        return f"User's Question: {query}"
    
    def query(self, query: str, user_role: str, chat_history: List[Dict[str, str]] = None) -> Dict[str, Any]:
        """Process a query with full security and access control, considering chat history."""
        if chat_history is None:
            chat_history = []
            
        try:
            # Validate query
            validation = self._validate_query(query, user_role)
            if not validation['valid']:
                return {
                    'answer': validation['reason'],
                    'sources': [],
                    'blocked': True, # Treat as blocked
                    'access_level': user_role,
                    'chunks_used': 0,
                    'response_type': 'blocked'
                }
            
            # Search for relevant documents and inaccessible sources
            search_results, inaccessible_sources = self.vector_store.search(query, user_role, k=3)
            
            # If no results are accessible, but relevant inaccessible docs were found
            if not search_results and inaccessible_sources:
                return {
                    'answer': "Access Denied.",
                    'sources': [],
                    'blocked': True,
                    'access_level': user_role,
                    'chunks_used': 0,
                    'response_type': 'blocked'
                }
            
            # Determine response type for UI coloring.
            # This is an assumption for the UI, the LLM makes the final call.
            response_type = 'document_based' if search_results else 'general_knowledge'
            
            # Create the flexible prompt
            system_prompt = self._create_system_prompt(user_role, search_results)
            
            # Reconstruct the message history for the LLM
            messages = [SystemMessage(content=system_prompt)]
            for message in chat_history:
                if message["role"] == "user":
                    messages.append(HumanMessage(content=message["content"]))
                elif message["role"] == "assistant":
                    messages.append(AIMessage(content=message["content"]))
            
            # Add the current user query
            messages.append(HumanMessage(content=self._create_user_prompt(query)))
            
            response = self.llm.invoke(messages)
            
            # Extract sources
            sources = list(set([result['source'] for result in search_results])) if search_results else []
            
            return {
                'answer': response.content,
                'sources': sources,
                'blocked': False,
                'access_level': user_role,
                'chunks_used': len(search_results),
                'search_results': search_results,
                'response_type': response_type 
            }
            
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            return {
                'answer': "I apologize, but I encountered an error while processing your query. Please try again.",
                'sources': [],
                'blocked': False,
                'access_level': user_role,
                'chunks_used': 0,
                'error': str(e),
                'response_type': 'error'
            }
    
    def get_query_analysis(self, query: str, user_role: str) -> Dict[str, Any]:
        """Analyze a query for security and access control without executing it"""
        validation = self._validate_query(query, user_role)
        
        # Check what would be accessible
        search_results, inaccessible_sources = self.vector_store.search(query, user_role, k=3)
        
        analysis = {
            'query_valid': validation['valid'],
            'blocked': validation['blocked'],
            'reason': validation['reason'],
            'accessible_chunks': len(search_results),
            'sensitive_topic_detected': config.is_topic_sensitive(query),
            'access_level': user_role,
            'estimated_sources': list(set([result['source'] for result in search_results])),
            'will_use_general_knowledge': len(search_results) == 0,
            'inaccessible_sources_found': inaccessible_sources
        }
        
        return analysis
    
    def get_context_preview(self, query: str, user_role: str, max_chunks: int = 3) -> List[Dict[str, Any]]:
        """Get a preview of what context would be used for a query"""
        search_results, _ = self.vector_store.search(query, user_role, k=max_chunks)
        
        if not search_results:
            return [{
                'content_preview': "No relevant document content found. Will use general knowledge to answer.",
                'source': 'general_knowledge',
                'access_level': 'general',
                'relevance_score': 1.0
            }]
        
        preview = []
        for result in search_results:
            preview.append({
                'content_preview': result['content'][:200] + "..." if len(result['content']) > 200 else result['content'],
                'source': result['source'],
                'access_level': result['access_level'],
                'relevance_score': 1.0 - (result['distance'] / 10.0)  # Normalize distance to 0-1 score
            })
        
        return preview 