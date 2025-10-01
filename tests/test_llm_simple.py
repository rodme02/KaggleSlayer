#!/usr/bin/env python3
"""
Simple LLM connection test for OpenRouter.
Run this to verify your LLM setup is working.
"""

import os
import sys

# Load .env manually
def load_env():
    env_file = ".env"
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()

# Check API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("ERROR: OPENAI_API_KEY not found in .env file")
    print("\nTo fix:")
    print("1. Get a free API key from https://openrouter.ai/")
    print("2. Add to .env file: OPENAI_API_KEY=your_key_here")
    sys.exit(1)

print("=" * 60)
print("OPENROUTER LLM CONNECTION TEST")
print("=" * 60)
print(f"\n✓ API Key found: {api_key[:20]}...")
print(f"✓ Base URL: https://openrouter.ai/api/v1")

# Try to import and test
print("\n" + "=" * 60)
print("Testing LLM client...")
print("=" * 60)

try:
    from utils.llm import LLMClient

    client = LLMClient()
    print("✓ LLMClient imported successfully")

    print("\n🔄 Sending test request to OpenRouter...")
    print("   Model: x-ai/grok-4-fast:free")
    print("   This may take a few seconds...\n")

    response = client.complete(
        prompt="Say 'Hello from KaggleSlayer!' and nothing else.",
        model="deepseek/deepseek-r1:free",
        max_tokens=50,
        temperature=0.3
    )

    if response:
        print("=" * 60)
        print("✅ SUCCESS! LLM IS WORKING")
        print("=" * 60)
        print(f"Response from LLM:\n{response}")
        print("\n" + "=" * 60)
        print("Your OpenRouter connection is ready!")
        print("The pipeline can now use LLM for intelligent insights.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("=" * 60)
        print("❌ FAILED: No response from LLM")
        print("=" * 60)
        print("\nPossible issues:")
        print("  - Invalid API key")
        print("  - Network/firewall blocking OpenRouter")
        print("  - OpenRouter service temporarily unavailable")
        print("  - Rate limit exceeded")
        print("\nTry:")
        print("  1. Verify API key at https://openrouter.ai/keys")
        print("  2. Check network connection")
        print("  3. Wait a few minutes and try again")
        sys.exit(1)

except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nMissing dependency. Install with:")
    print("  pip install openai tenacity")
    sys.exit(1)

except Exception as e:
    print(f"❌ Error: {e}")
    print("\nUnexpected error occurred.")
    print("Check that:")
    print("  1. openai package is installed (pip install openai)")
    print("  2. API key is correct")
    print("  3. Network allows connections to openrouter.ai")
    sys.exit(1)
