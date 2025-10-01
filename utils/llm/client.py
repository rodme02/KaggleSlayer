"""
LLM client for API interactions.
Compatible with OpenAI SDK v1.0+
"""

import json
from typing import Dict, Any, Optional
from pathlib import Path
import os
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from openai import OpenAI
    OPENAI_V1 = True
except ImportError:
    import openai
    OPENAI_V1 = False


class LLMClient:
    """Simple LLM client for API interactions with OpenRouter."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or "https://openrouter.ai/api/v1"

        if OPENAI_V1:
            # OpenAI SDK v1.0+
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            ) if self.api_key else None
        else:
            # OpenAI SDK v0.x (legacy)
            if self.api_key:
                openai.api_key = self.api_key
                openai.api_base = self.base_url
            self.client = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def complete(self, prompt: str, model: str = "x-ai/grok-4-fast:free",
                max_tokens: int = 2000, temperature: float = 0.7) -> Optional[str]:
        """Complete a prompt using the LLM."""
        try:
            if OPENAI_V1:
                # OpenAI SDK v1.0+
                if not self.client:
                    raise ValueError("LLM client not initialized. Check API key.")

                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
            else:
                # OpenAI SDK v0.x (legacy)
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content
        except Exception as e:
            print(f"LLM API error: {e}")
            return None

    def extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from LLM response."""
        try:
            # Try to find JSON in the response
            start_idx = text.find('{')
            end_idx = text.rfind('}') + 1

            if start_idx != -1 and end_idx > start_idx:
                json_str = text[start_idx:end_idx]
                return json.loads(json_str)
            else:
                return None
        except json.JSONDecodeError:
            return None