#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script to verify OpenRouter LLM connection.
"""

import os
import sys
import io
from pathlib import Path

# Fix encoding issues on Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Manually load .env file
def load_env():
    """Load environment variables from .env file."""
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()

load_env()

def test_env_variables():
    """Check if API keys are configured."""
    print("="*60)
    print("CHECKING ENVIRONMENT VARIABLES")
    print("="*60)

    openai_key = os.getenv("OPENAI_API_KEY")
    openrouter_key = os.getenv("OPENROUTER_API_KEY")

    if openrouter_key:
        print(f"✓ OPENROUTER_API_KEY found: {openrouter_key[:20]}...")
    else:
        print("✗ OPENROUTER_API_KEY not found")

    if openai_key:
        print(f"✓ OPENAI_API_KEY found: {openai_key[:20]}...")
    else:
        print("✗ OPENAI_API_KEY not found")

    print()

    # The code uses OPENAI_API_KEY, so we need to set it from OPENROUTER_API_KEY
    if openrouter_key and not openai_key:
        print("⚠ Setting OPENAI_API_KEY from OPENROUTER_API_KEY...")
        os.environ["OPENAI_API_KEY"] = openrouter_key
        print("✓ Environment variable synchronized")

    return bool(openrouter_key or openai_key)


def test_llm_import():
    """Test importing LLM client."""
    print("\n" + "="*60)
    print("TESTING LLM CLIENT IMPORT")
    print("="*60)

    try:
        from utils.llm import LLMClient
        print("✓ LLMClient imported successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to import LLMClient: {e}")
        return False


def test_llm_connection():
    """Test actual LLM API call."""
    print("\n" + "="*60)
    print("TESTING LLM API CONNECTION")
    print("="*60)

    try:
        from utils.llm import LLMClient

        # Synchronize API key
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if openrouter_key and not openai_key:
            os.environ["OPENAI_API_KEY"] = openrouter_key

        # Create client
        client = LLMClient()

        if not client.api_key:
            print("✗ No API key configured")
            print("\nTo fix this:")
            print("1. Get a free API key from https://openrouter.ai/")
            print("2. Add to .env file: OPENAI_API_KEY=your_key_here")
            return False

        print(f"✓ API Key configured: {client.api_key[:20]}...")
        print(f"✓ Base URL: {client.base_url}")
        print("\n🔄 Testing API call (this may take a few seconds)...")

        # Simple test prompt
        response = client.complete(
            prompt="Say 'Hello from KaggleSlayer!' and nothing else.",
            model="x-ai/grok-4-fast:free",
            max_tokens=50,
            temperature=0.3
        )

        if response:
            print(f"\n✅ LLM API WORKING!")
            print(f"Response: {response}")
            return True
        else:
            print("\n✗ LLM returned no response")
            print("This could mean:")
            print("  - API key is invalid")
            print("  - Network connectivity issue")
            print("  - OpenRouter service is down")
            print("  - Rate limit exceeded")
            return False

    except Exception as e:
        print(f"\n✗ LLM API call failed: {e}")
        print("\nPossible issues:")
        print("  1. Invalid API key")
        print("  2. Network/firewall blocking OpenRouter")
        print("  3. Missing dependencies (pip install openai)")
        return False


def test_llm_coordinator():
    """Test LLM coordinator."""
    print("\n" + "="*60)
    print("TESTING LLM COORDINATOR")
    print("="*60)

    try:
        from utils.llm import LLMCoordinator

        # Synchronize API key
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")

        if openrouter_key and not openai_key:
            os.environ["OPENAI_API_KEY"] = openrouter_key

        coordinator = LLMCoordinator()
        print("✓ LLMCoordinator imported successfully")
        print("✓ Ready to generate insights for pipeline")
        return True
    except Exception as e:
        print(f"✗ Failed to import LLMCoordinator: {e}")
        return False


def main():
    """Run all tests."""
    print("\n")
    print("█"*60)
    print(" "*15 + "LLM CONNECTION TEST")
    print("█"*60)

    results = []

    # Test 1: Environment variables
    results.append(("Environment Variables", test_env_variables()))

    # Test 2: Import
    results.append(("LLM Import", test_llm_import()))

    # Test 3: Connection
    results.append(("LLM Connection", test_llm_connection()))

    # Test 4: Coordinator
    results.append(("LLM Coordinator", test_llm_coordinator()))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:12} {test_name}")

    print("\n" + "="*60)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("="*60)

    if passed == total:
        print("\n🎉 All LLM tests passed! OpenRouter is ready to use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
