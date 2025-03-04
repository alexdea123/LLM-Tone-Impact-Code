import os
import json
import time
from typing import Dict, List, Optional
import httpx

class GroqAPI:
    """
    Class to handle interactions with the Groq API for generating code.
    """
    def __init__(self, api_key: Optional[str] = None, model: str = "llama3-70b-8192"):
        """
        Initialize the GroqAPI client.
        
        Args:
            api_key: Groq API key. If None, will try to get from environment variable GROQ_API_KEY
            model: Model identifier to use (default: llama3-70b-8192)
        """
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("API key must be provided or set as GROQ_API_KEY environment variable")
        
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"
        self.client = httpx.Client(
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
        )
    
    def generate_code(self, prompt: str, temperature: float = 0.2, max_tokens: int = 4000) -> Dict:
        """
        Generate code using the Groq API.
        
        Args:
            prompt: The prompt to send to the model
            temperature: Sampling temperature (lower = more deterministic)
            max_tokens: Maximum number of tokens to generate
            
        Returns:
            Dictionary containing the full API response
        """
        try:
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            print(f"Error generating code: {e}")
            return {"error": str(e)}
    
    def extract_code_from_response(self, response: Dict) -> str:
        """
        Extract the generated code from the API response.
        
        Args:
            response: Full API response dictionary
            
        Returns:
            The extracted code as a string
        """
        try:
            content = response["choices"][0]["message"]["content"]
            return content
        except (KeyError, IndexError) as e:
            print(f"Error extracting code from response: {e}")
            return ""
