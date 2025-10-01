#!/usr/bin/env python3
"""
LLM Coordinator - Interface to OpenRouter Free LLMs for KaggleSlayer

Uses free OpenRouter models for autonomous Kaggle competition intelligence:
- Primary: x-ai/grok-4-fast:free (general intelligence and all tasks)
"""

import json
import os
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional


@dataclass
class LLMCall:
    """Track LLM API calls for monitoring and debugging"""
    timestamp: str
    model: str
    agent: str
    prompt_hash: str
    prompt_preview: str  # First 100 chars
    response_preview: str  # First 200 chars
    tokens_used: int
    success: bool
    error_message: Optional[str] = None
    execution_time: float = 0.0


class LLMCoordinator:
    """
    Central coordinator for all LLM interactions in KaggleSlayer
    """

    # Free OpenRouter model configurations
    MODELS = {
        "primary": {
            "name": "x-ai/grok-4-fast:free",
            "description": "Fast, general intelligence model",
            "use_case": "general analysis, insights, recommendations"
        },
        "code": {
            "name": "deepseek/deepseek-coder-6.7b-instruct",
            "description": "Specialized code generation model",
            "use_case": "feature engineering, code generation"
        },
        "fallback": {
            "name": "meta-llama/llama-3.2-3b-instruct",
            "description": "Reliable backup model",
            "use_case": "backup when other models fail"
        }
    }

    def __init__(self, log_dir: Path = None):
        """Initialize LLM coordinator with OpenRouter client"""
        self.log_dir = log_dir or Path("llm_logs")
        self.log_dir.mkdir(exist_ok=True)

        # Initialize OpenAI client with OpenRouter
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY not found in environment. "
                "Get your free API key from https://openrouter.ai/"
            )

        self.client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key,
        )

        # Cache for identical prompts
        self.cache = {}
        self.cache_file = self.log_dir / "llm_cache.json"
        self._load_cache()

        # Call tracking
        self.calls_log = []
        self.total_tokens = 0

        print(f"LLM Coordinator initialized with OpenRouter")
        print(f"Available models: {', '.join([m['name'] for m in self.MODELS.values()])}")

    def _load_cache(self):
        """Load cached responses to avoid duplicate API calls"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    self.cache = json.load(f)
                print(f"Loaded {len(self.cache)} cached responses")
            except Exception as e:
                print(f"Warning: Could not load cache: {e}")
                self.cache = {}

    def _save_cache(self):
        """Save response cache to file"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cache: {e}")

    def _get_cache_key(self, model: str, messages: List[Dict], temperature: float) -> str:
        """Generate cache key for identical prompts"""
        content = f"{model}:{temperature}:{json.dumps(messages, sort_keys=True)}"
        return hashlib.md5(content.encode()).hexdigest()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError))
    )
    def _make_api_call(self, model: str, messages: List[Dict], temperature: float = 0.7,
                      max_tokens: int = 2000) -> Dict:
        """Make API call with retry logic"""
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                extra_headers={
                    "HTTP-Referer": "https://github.com/KaggleSlayer",
                    "X-Title": "KaggleSlayer Autonomous Agent"
                }
            )

            execution_time = time.time() - start_time

            return {
                "content": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "execution_time": execution_time,
                "success": True
            }

        except Exception as e:
            execution_time = time.time() - start_time
            return {
                "content": None,
                "tokens_used": 0,
                "execution_time": execution_time,
                "success": False,
                "error": str(e)
            }

    def chat(self, prompt: str, agent: str = "unknown", model_type: str = "primary",
             temperature: float = 0.7, max_tokens: int = 2000) -> Optional[str]:
        """
        Send a chat message to LLM and get response

        Args:
            prompt: The prompt to send
            agent: Name of calling agent (for logging)
            model_type: Type of model to use (primary/code/fallback)
            temperature: Randomness (0=deterministic, 1=creative)
            max_tokens: Maximum response length

        Returns:
            LLM response text or None if failed
        """
        if model_type not in self.MODELS:
            print(f"Warning: Unknown model type '{model_type}', using primary")
            model_type = "primary"

        model = self.MODELS[model_type]["name"]
        messages = [{"role": "user", "content": prompt}]

        # Check cache first
        cache_key = self._get_cache_key(model, messages, temperature)
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            print(f"Using cached response for {agent}")

            # Still log the call for tracking
            call = LLMCall(
                timestamp=datetime.now().isoformat(),
                model=model,
                agent=agent,
                prompt_hash=cache_key[:8],
                prompt_preview=prompt[:100],
                response_preview=cached["content"][:200] if cached["content"] else "",
                tokens_used=cached.get("tokens_used", 0),
                success=True,
                execution_time=0.0
            )
            self.calls_log.append(call)
            return cached["content"]

        print(f"Making LLM call: {agent} -> {model}")

        # Try primary model first, fallback if needed
        models_to_try = [model]
        if model_type != "fallback":
            models_to_try.append(self.MODELS["fallback"]["name"])

        for try_model in models_to_try:
            result = self._make_api_call(try_model, messages, temperature, max_tokens)

            # Log the call
            call = LLMCall(
                timestamp=datetime.now().isoformat(),
                model=try_model,
                agent=agent,
                prompt_hash=cache_key[:8],
                prompt_preview=prompt[:100],
                response_preview=result.get("content", "")[:200] if result.get("content") else "",
                tokens_used=result.get("tokens_used", 0),
                success=result["success"],
                error_message=result.get("error"),
                execution_time=result.get("execution_time", 0.0)
            )

            self.calls_log.append(call)
            self.total_tokens += result.get("tokens_used", 0)

            if result["success"] and result["content"]:
                # Cache successful response
                self.cache[cache_key] = {
                    "content": result["content"],
                    "tokens_used": result["tokens_used"],
                    "model": try_model
                }
                self._save_cache()

                print(f"LLM response received: {result['tokens_used']} tokens")
                return result["content"]
            else:
                print(f"LLM call failed with {try_model}: {result.get('error', 'Unknown error')}")
                if try_model == models_to_try[-1]:  # Last attempt failed
                    break

        print(f"All LLM models failed for {agent}")
        return None

    def structured_output(self, prompt: str, agent: str = "unknown",
                         model_type: str = "primary", **kwargs) -> Optional[Dict]:
        """
        Get structured JSON output from LLM

        Args:
            prompt: The prompt requesting JSON output
            agent: Name of calling agent
            model_type: Type of model to use

        Returns:
            Parsed JSON dict or None if failed
        """
        # Add JSON instruction to prompt
        json_prompt = f"{prompt}\n\nReturn your response as valid JSON only, no other text."

        response = self.chat(json_prompt, agent, model_type, **kwargs)
        if not response:
            return None

        # Try to parse JSON from response
        try:
            # Extract JSON if wrapped in markdown code blocks
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                if json_end != -1:
                    response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                if json_end != -1:
                    response = response[json_start:json_end].strip()

            return json.loads(response)

        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse JSON response from {agent}: {e}")
            print(f"Response was: {response[:200]}...")
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        return {
            "models": self.MODELS,
            "total_calls": len(self.calls_log),
            "total_tokens": self.total_tokens,
            "cache_size": len(self.cache)
        }

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        if not self.calls_log:
            return {"total_calls": 0, "total_tokens": 0, "agents": {}}

        agent_stats = {}
        for call in self.calls_log:
            if call.agent not in agent_stats:
                agent_stats[call.agent] = {
                    "calls": 0,
                    "tokens": 0,
                    "successes": 0,
                    "failures": 0
                }

            agent_stats[call.agent]["calls"] += 1
            agent_stats[call.agent]["tokens"] += call.tokens_used
            if call.success:
                agent_stats[call.agent]["successes"] += 1
            else:
                agent_stats[call.agent]["failures"] += 1

        return {
            "total_calls": len(self.calls_log),
            "total_tokens": self.total_tokens,
            "estimated_cost": 0.00,  # All free models
            "agents": agent_stats,
            "cache_hits": len(self.cache)
        }

    def export_logs(self) -> None:
        """Export all LLM calls to JSON file"""
        log_file = self.log_dir / f"llm_calls_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(log_file, 'w') as f:
            json.dump([asdict(call) for call in self.calls_log], f, indent=2)

        print(f"Exported {len(self.calls_log)} LLM calls to {log_file}")

    def test_connection(self) -> bool:
        """Test connection to OpenRouter with all available models"""
        print("Testing OpenRouter connection...")

        test_prompt = "Hello! Respond with exactly: 'Connection successful'"

        for model_type, model_info in self.MODELS.items():
            print(f"Testing {model_type} model: {model_info['name']}")

            response = self.chat(
                test_prompt,
                agent="connection_test",
                model_type=model_type,
                temperature=0.0,
                max_tokens=50
            )

            if response and "Connection successful" in response:
                print(f"OK {model_type} model working correctly")
            else:
                print(f"X {model_type} model failed: {response}")
                return False

        print("All OpenRouter models are working!")
        return True


def main():
    """Test script for LLM Coordinator"""
    try:
        llm = LLMCoordinator()

        if llm.test_connection():
            print("\n=== Testing structured output ===")

            test_prompt = """
            Analyze this simple dataset for a classification problem:
            - 1000 rows, 5 features
            - Target: binary (0/1)
            - Features: age, income, education, location, score
            - Missing values: 5% in income, 2% in education

            Provide analysis as JSON with keys:
            - problem_type: string
            - key_features: array of strings
            - preprocessing_steps: array of strings
            - model_recommendations: array of strings
            """

            result = llm.structured_output(test_prompt, agent="test")
            if result:
                print("Structured output test successful:")
                print(json.dumps(result, indent=2))
            else:
                print("Structured output test failed")

            print("\n=== Usage Statistics ===")
            stats = llm.get_usage_stats()
            print(json.dumps(stats, indent=2))

        else:
            print("Connection test failed")

    except Exception as e:
        print(f"Error testing LLM coordinator: {e}")


if __name__ == "__main__":
    main()