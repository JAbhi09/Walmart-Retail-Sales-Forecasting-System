"""
Base Agent Class for Multi-Agent AI System
Provides common functionality for all AI agents
"""
import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import google.generativeai as genai
from datetime import datetime

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for AI agents
    """
    
    def __init__(self, name: str, model_name: str = "gemini-2.5-flash"):
        """
        Initialize base agent
        
        Args:
            name: Agent name
            model_name: Gemini model to use
        """
        self.name = name
        self.model_name = model_name
        
        # Configure Gemini API
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Agent state
        self.conversation_history = []
        
        logger.info(f"✓ {self.name} initialized with {model_name}")
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent
        
        Returns:
            str: System prompt
        """
        pass
    
    @abstractmethod
    def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request with the given context
        
        Args:
            context: Context dictionary with relevant data
        
        Returns:
            Dict containing agent's response and metadata
        """
        pass
    
    def generate_response(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate a response using Gemini
        """
        system_prompt = self.get_system_prompt()

        if context:
            context_str = self._format_context(context)
            full_prompt = f"{system_prompt}\n\n{context_str}\n\n{prompt}"
        else:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        try:
            response = self.model.generate_content(full_prompt)
            text = response.text
            text = text.replace('$', '\\$')

            # Downsize headings: convert # and ## to ### and ####
            import re
            text = re.sub(r'^# ', '### ', text, flags=re.MULTILINE)
            text = re.sub(r'^## ', '#### ', text, flags=re.MULTILINE)
            # # Clean up formatting issues
            # # Fix broken bold markers that Streamlit can't render
            # import re
            # # Ensure ** markers have spaces around them properly
            # text = re.sub(r'\*\*\s+', '** ', text)
            # text = re.sub(r'\s+\*\*', ' **', text)
            # # Fix cases where ** runs into dollar amounts or numbers
            # text = re.sub(r'\$([0-9,]+\.?\d*)\*\*', r'$\1** ', text)
            # text = re.sub(r'\*\*\$', r'** $', text)

            self.conversation_history.append({
                'timestamp': datetime.now(),
                'prompt': prompt,
                'response': text
            })

            return text

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: Unable to generate response - {str(e)}"
            
    def _format_context(self, context: Dict[str, Any]) -> str:
        """
        Format context dictionary into a readable string
        
        Args:
            context: Context dictionary
        
        Returns:
            str: Formatted context
        """
        lines = ["CONTEXT:"]
        for key, value in context.items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info(f"{self.name}: Conversation history cleared")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """
        Get conversation history
        
        Returns:
            List of conversation entries
        """
        return self.conversation_history
