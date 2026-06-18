"""
Model-agnostic LLM client supporting multiple providers.
Abstraction layer for OpenAI, Anthropic, Groq, Ollama, and more.
"""

import logging
import os
from typing import Tuple, Dict, Any
from config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Model-agnostic LLM client."""
    
    def __init__(self):
        """Initialize LLM client."""
        self.default_temperature = 0.3
        self.default_max_tokens = 2500
    
    def generate(self, prompt: str, model: str, temperature: float = None, 
                max_tokens: int = None) -> Tuple[str, Dict[str, int]]:
        """
        Generate text using specified model.
        
        Args:
            prompt: Input prompt
            model: Model identifier (e.g., "gpt-4", "claude-3-sonnet", "groq/mixtral")
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (generated_text, token_usage)
        """
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        
        # Determine provider from model name
        provider = self._get_provider(model)
        
        try:
            if "gpt" in model.lower():
                return self._generate_openai(prompt, model, temperature, max_tokens)
            elif "claude" in model.lower():
                return self._generate_anthropic(prompt, model, temperature, max_tokens)
            elif "groq" in model.lower():
                return self._generate_groq(prompt, model, temperature, max_tokens)
            elif "gemini" in model.lower():
                return self._generate_google(prompt, model, temperature, max_tokens)
            elif "ollama" in model.lower() or provider == "ollama":
                return self._generate_ollama(prompt, model, temperature, max_tokens)
            elif "bedrock" in model.lower():
                return self._generate_bedrock(prompt, model, temperature, max_tokens)
            else:
                # Default to OpenAI for unknown models
                logger.warning(f"Unknown model {model}, defaulting to OpenAI")
                return self._generate_openai(prompt, model, temperature, max_tokens)
                
        except Exception as e:
            logger.error(f"Error generating with {model}: {str(e)}")
            raise
    
    def _get_provider(self, model: str) -> str:
        """Determine provider from model name."""
        if "gpt" in model.lower():
            return "openai"
        elif "claude" in model.lower():
            return "anthropic"
        elif "groq" in model.lower():
            return "groq"
        elif "gemini" in model.lower():
            return "google"
        elif "ollama" in model.lower():
            return "ollama"
        elif "bedrock" in model.lower():
            return "bedrock"
        return "unknown"
    
    def _generate_openai(self, prompt: str, model: str, temperature: float, 
                        max_tokens: int) -> Tuple[str, Dict[str, int]]:
        """Generate using OpenAI API."""
        try:
            import openai
            
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set")
            
            client = openai.OpenAI(api_key=settings.openai_api_key)
            
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            text = response.choices[0].message.content
            token_usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return text, token_usage
            
        except ImportError:
            raise RuntimeError("openai package not installed. Install with: pip install openai")
    
    def _generate_anthropic(self, prompt: str, model: str, temperature: float, 
                           max_tokens: int) -> Tuple[str, Dict[str, int]]:
        """Generate using Anthropic API."""
        try:
            import anthropic
            
            if not settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            
            client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            
            text = response.content[0].text
            token_usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens
            }
            
            return text, token_usage
            
        except ImportError:
            raise RuntimeError("anthropic package not installed. Install with: pip install anthropic")
    
    def _generate_groq(self, prompt: str, model: str, temperature: float, 
                      max_tokens: int) -> Tuple[str, Dict[str, int]]:
        """Generate using Groq API."""
        try:
            import groq
            
            if not settings.groq_api_key:
                raise ValueError("GROQ_API_KEY not set")
            
            client = groq.Groq(api_key=settings.groq_api_key)
            
            # Extract model name if using groq/ prefix
            model_name = model.replace("groq/", "")
            
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            text = response.choices[0].message.content
            token_usage = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
            
            return text, token_usage
            
        except ImportError:
            raise RuntimeError("groq package not installed. Install with: pip install groq")
    
    def _generate_google(self, prompt: str, model: str, temperature: float, 
                        max_tokens: int) -> Tuple[str, Dict[str, int]]:
        """Generate using Google Gemini API."""
        try:
            import google.generativeai as genai
            
            if not settings.google_api_key:
                raise ValueError("GOOGLE_API_KEY not set")
            
            genai.configure(api_key=settings.google_api_key)
            
            model_name = model.replace("gemini/", "")
            gemini_model = genai.GenerativeModel(model_name)
            
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens
                )
            )
            
            text = response.text
            # Google doesn't return token counts, estimate
            token_usage = {
                "input_tokens": len(prompt) // 4,
                "output_tokens": len(text) // 4,
                "total_tokens": (len(prompt) + len(text)) // 4
            }
            
            return text, token_usage
            
        except ImportError:
            raise RuntimeError("google-generativeai package not installed. Install with: pip install google-generativeai")
    
    def _generate_ollama(self, prompt: str, model: str, temperature: float, 
                        max_tokens: int) -> Tuple[str, Dict[str, int]]:
        """Generate using Ollama local model."""
        try:
            import requests
            
            # Extract model name
            model_name = model.replace("ollama/", "")
            
            url = f"{settings.ollama_base_url}/api/generate"
            
            response = requests.post(
                url,
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stream": False
                },
                timeout=300
            )
            
            if response.status_code != 200:
                raise ValueError(f"Ollama error: {response.text}")
            
            result = response.json()
            text = result.get("response", "")
            
            token_usage = {
                "input_tokens": result.get("prompt_eval_count", len(prompt) // 4),
                "output_tokens": result.get("eval_count", len(text) // 4),
                "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
            }
            
            return text, token_usage
            
        except ImportError:
            raise RuntimeError("requests package not installed. Install with: pip install requests")
    
    def _generate_bedrock(self, prompt: str, model: str, temperature: float, 
                         max_tokens: int) -> Tuple[str, Dict[str, int]]:
        """Generate using AWS Bedrock."""
        try:
            import boto3
            import json
            
            client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
            
            # Extract model name
            model_id = model.replace("bedrock/", "")
            
            # Different models have different payload formats
            if "claude" in model_id:
                payload = {
                    "anthropic_version": "bedrock-2023-06-01",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "messages": [{"role": "user", "content": prompt}]
                }
            else:
                payload = {
                    "inputText": prompt,
                    "textGenerationConfig": {
                        "maxTokenCount": max_tokens,
                        "temperature": temperature
                    }
                }
            
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(payload)
            )
            
            result = json.loads(response["body"].read())
            
            if "claude" in model_id:
                text = result["content"][0]["text"]
                token_usage = {
                    "input_tokens": result.get("usage", {}).get("input_tokens", 0),
                    "output_tokens": result.get("usage", {}).get("output_tokens", 0),
                    "total_tokens": result.get("usage", {}).get("input_tokens", 0) + result.get("usage", {}).get("output_tokens", 0)
                }
            else:
                text = result.get("outputText", "")
                token_usage = {
                    "input_tokens": len(prompt) // 4,
                    "output_tokens": len(text) // 4,
                    "total_tokens": (len(prompt) + len(text)) // 4
                }
            
            return text, token_usage
            
        except ImportError:
            raise RuntimeError("boto3 package not installed. Install with: pip install boto3")
    
    def validate_model(self, model: str) -> bool:
        """
        Validate that model is available.
        
        Args:
            model: Model identifier
            
        Returns:
            True if model is available
        """
        provider = self._get_provider(model)
        
        if provider == "openai" and not settings.openai_api_key:
            return False
        elif provider == "anthropic" and not settings.anthropic_api_key:
            return False
        elif provider == "groq" and not settings.groq_api_key:
            return False
        elif provider == "google" and not settings.google_api_key:
            return False
        elif provider == "ollama":
            # Try to connect to Ollama
            try:
                import requests
                response = requests.get(f"{settings.ollama_base_url}/api/tags", timeout=5)
                return response.status_code == 200
            except:
                return False
        
        return True
