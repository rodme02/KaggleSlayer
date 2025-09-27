#!/usr/bin/env python3
"""
Test LLM Setup - Verify LLM components work correctly

This script tests the LLM infrastructure without requiring actual API keys.
For real API testing, set OPENROUTER_API_KEY in your environment.
"""

import os
import sys
import json
from pathlib import Path

def test_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    try:
        import openai
        import tenacity
        from dotenv import load_dotenv

        # Test our custom modules
        sys.path.append(str(Path.cwd()))
        from agents.llm_coordinator import LLMCoordinator, LLMCall
        from utils.llm_utils import PromptTemplates, LLMUtils

        print("All imports successful")
        return True
    except ImportError as e:
        print(f"Import error: {e}")
        return False

def test_prompt_templates():
    """Test that prompt templates work correctly"""
    print("\nTesting prompt templates...")

    try:
        from utils.llm_utils import PromptTemplates, LLMUtils

        # Test competition analysis template
        comp_prompt = PromptTemplates.competition_analysis(
            title="Test Competition",
            description="Predict house prices using features",
            evaluation_metric="RMSE",
            dataset_files=["train.csv", "test.csv"]
        )

        assert "Test Competition" in comp_prompt
        assert "RMSE" in comp_prompt
        assert "JSON" in comp_prompt
        print("Competition analysis template works")

        # Test dataset insights template
        dataset_prompt = PromptTemplates.dataset_insights(
            eda_summary="Dataset has 1000 rows, 10 features, 5% missing values",
            competition_context="Regression problem with RMSE evaluation"
        )

        assert "1000 rows" in dataset_prompt
        assert "JSON" in dataset_prompt
        print("Dataset insights template works")

        # Test feature engineering template
        feature_prompt = PromptTemplates.feature_engineering_code(
            dataset_info={"total_rows": 1000, "total_columns": 10, "target_type": "regression"},
            competition_insights={"key_strategies": ["feature engineering"], "model_recommendations": ["XGBoost"]},
            feature_descriptions="age, income, location features"
        )

        assert "feature engineering" in feature_prompt
        assert "JSON" in feature_prompt
        print("Feature engineering template works")

        return True

    except Exception as e:
        print(f"Prompt template error: {e}")
        return False

def test_llm_utils():
    """Test LLM utility functions"""
    print("\nTesting LLM utilities...")

    try:
        from utils.llm_utils import LLMUtils
        # Test dataset summary formatting
        dataset_info = {
            'total_rows': 1000,
            'total_columns': 10,
            'target_column': 'price',
            'target_type': 'regression',
            'memory_usage_mb': 5.2,
            'duplicates_count': 0,
            'feature_types': {'age': 'numerical', 'name': 'categorical', 'income': 'numerical'},
            'missing_percentages': {'age': 0.0, 'name': 5.0, 'income': 2.0}
        }

        summary = LLMUtils.format_dataset_summary(dataset_info)
        assert "1,000 rows" in summary
        assert "price" in summary
        assert "regression" in summary
        print("Dataset summary formatting works")

        # Test text truncation
        long_text = "A" * 5000
        truncated = LLMUtils.truncate_text(long_text, 100)
        assert len(truncated) <= 103  # 100 + "..."
        print("Text truncation works")

        # Test code block extraction
        text_with_code = """
        Here's some code:
        ```python
        df['new_feature'] = df['old_feature'] * 2
        ```
        And more text.
        """

        code_blocks = LLMUtils.extract_code_blocks(text_with_code)
        assert len(code_blocks) == 1
        assert "new_feature" in code_blocks[0]
        print("Code block extraction works")

        return True

    except Exception as e:
        print(f"LLM utils error: {e}")
        return False

def test_llm_coordinator_without_api():
    """Test LLM coordinator functionality without actual API calls"""
    print("\nTesting LLM coordinator (without API)...")

    try:
        from agents.llm_coordinator import LLMCoordinator, LLMCall
        # Test that we can create the coordinator structure
        # (This will fail without API key, but we can test the structure)

        models = LLMCoordinator.MODELS
        assert "primary" in models
        assert "code" in models
        assert "fallback" in models
        print("Model configurations look correct")

        # Test LLMCall dataclass
        call = LLMCall(
            timestamp="2023-01-01",
            model="test-model",
            agent="test-agent",
            prompt_hash="abc123",
            prompt_preview="Test prompt",
            response_preview="Test response",
            tokens_used=100,
            success=True
        )

        call_dict = call.__dict__
        assert call_dict["tokens_used"] == 100
        assert call_dict["success"] == True
        print("LLMCall dataclass works")

        return True

    except Exception as e:
        print(f"LLM coordinator structure error: {e}")
        return False

def test_with_real_api():
    """Test with real API if credentials are available"""
    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        print("\nNo OPENROUTER_API_KEY found - skipping real API test")
        print("To test with real API:")
        print("1. Get free API key from https://openrouter.ai/")
        print("2. Set OPENROUTER_API_KEY environment variable")
        print("3. Run: python agents/llm_coordinator.py")
        return True

    print(f"\nTesting with real OpenRouter API...")

    try:
        from agents.llm_coordinator import LLMCoordinator

        llm = LLMCoordinator()

        # Test connection with all models
        success = llm.test_connection()

        if success:
            print("Real API connection successful!")

            # Show usage stats
            stats = llm.get_usage_stats()
            print(f"API calls made: {stats['total_calls']}")
            print(f"Tokens used: {stats['total_tokens']}")
            print(f"Estimated cost: ${stats['estimated_cost']}")

        return success

    except Exception as e:
        print(f"Real API test error: {e}")
        return False

def main():
    """Run all tests"""
    print("KaggleSlayer LLM Setup Test")
    print("=" * 30)

    tests = [
        test_imports,
        test_prompt_templates,
        test_llm_utils,
        test_llm_coordinator_without_api,
        test_with_real_api
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1
        else:
            break  # Stop on first failure

    print(f"\n" + "=" * 30)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("All LLM components are ready!")
        print("\nNext steps:")
        print("1. Get OpenRouter API key from https://openrouter.ai/")
        print("2. Copy .env.example to .env and add your key")
        print("3. Test with: python agents/llm_coordinator.py")
        return True
    else:
        print("Some tests failed - check the errors above")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)