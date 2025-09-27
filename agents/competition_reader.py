#!/usr/bin/env python3
"""
Competition Reader Agent - Analyzes Kaggle competitions using LLM intelligence

This agent fetches competition information and uses LLM to generate strategic insights
that guide all other agents in the pipeline for optimal performance.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False

from agents.llm_coordinator import LLMCoordinator
from utils.llm_utils import PromptTemplates, LLMUtils


@dataclass
class CompetitionInsights:
    """Structure to hold LLM-generated competition insights"""
    competition_name: str
    problem_type: str
    difficulty_level: str
    key_strategies: List[str]
    evaluation_focus: str
    common_pitfalls: List[str]
    feature_opportunities: List[str]
    model_recommendations: List[str]
    time_complexity: str
    data_concerns: List[str]
    success_factors: List[str]
    analysis_timestamp: str
    confidence_score: float = 0.0


class CompetitionReader:
    """
    Competition Intelligence Agent that analyzes competitions for strategic insights
    """

    def __init__(self, competition_path: Path, llm_coordinator: LLMCoordinator = None):
        self.competition_path = Path(competition_path)
        self.competition_name = self.competition_path.name
        self.insights_file = self.competition_path / "competition_understanding.json"

        # Initialize LLM coordinator
        self.llm = llm_coordinator or LLMCoordinator(log_dir=self.competition_path / "llm_logs")

        # Initialize Kaggle API if available
        self.kaggle_api = None
        if KAGGLE_AVAILABLE:
            try:
                self.kaggle_api = KaggleApi()
                self.kaggle_api.authenticate()
                print(f"Kaggle API authenticated for competition analysis")
            except Exception as e:
                print(f"Warning: Kaggle API authentication failed: {e}")

    def fetch_competition_info(self) -> Dict[str, Any]:
        """Fetch competition information from Kaggle API or local files"""
        competition_info = {
            "title": self.competition_name.replace('-', ' ').title(),
            "description": "",
            "evaluation_metric": "Unknown",
            "dataset_files": [],
            "data_source": "local"
        }

        # Try to get info from Kaggle API
        if self.kaggle_api:
            try:
                print(f"Fetching competition info from Kaggle API: {self.competition_name}")

                # Get competition details
                comp_details = self.kaggle_api.competition_view(self.competition_name)

                if comp_details:
                    competition_info.update({
                        "title": comp_details.title or self.competition_name,
                        "description": comp_details.description or "No description available",
                        "evaluation_metric": comp_details.evaluationMetric or "Unknown",
                        "data_source": "kaggle_api"
                    })

                    print(f"Retrieved competition details from Kaggle API")

            except Exception as e:
                print(f"Could not fetch from Kaggle API: {e}, using local analysis")

        # Get dataset files from local directory
        dataset_files = []
        for file_pattern in ["*.csv", "*.json", "*.txt"]:
            dataset_files.extend([f.name for f in self.competition_path.glob(file_pattern)])

        competition_info["dataset_files"] = dataset_files

        # Infer information from local files if API failed
        if not competition_info["description"] or competition_info["description"] == "":
            competition_info["description"] = self._infer_description_from_files()

        if competition_info["evaluation_metric"] == "Unknown":
            competition_info["evaluation_metric"] = self._infer_metric_from_files()

        return competition_info

    def _infer_description_from_files(self) -> str:
        """Infer competition description from local files and structure"""
        description_parts = []

        # Check for README or description files
        for readme_file in ["README.md", "readme.txt", "description.txt"]:
            readme_path = self.competition_path / readme_file
            if readme_path.exists():
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        content = f.read()[:1000]  # First 1000 chars
                        description_parts.append(content)
                except Exception:
                    pass

        # Analyze dataset structure
        train_path = self.competition_path / "train.csv"
        if train_path.exists():
            try:
                import pandas as pd
                df = pd.read_csv(train_path, nrows=5)  # Sample first few rows

                description_parts.append(f"Training dataset with {len(df.columns)} columns: {', '.join(df.columns[:5])}")

                # Check for common target patterns
                for col in df.columns:
                    if col.lower() in ['target', 'survived', 'price', 'saleprice', 'class', 'label']:
                        description_parts.append(f"Prediction target appears to be: {col}")
                        break

            except Exception as e:
                print(f"Could not analyze training data: {e}")

        if description_parts:
            return " | ".join(description_parts)
        else:
            return f"Competition to predict outcomes using provided dataset features."

    def _infer_metric_from_files(self) -> str:
        """Infer evaluation metric from competition structure"""
        # Common patterns in competition names and files
        name_lower = self.competition_name.lower()

        if any(word in name_lower for word in ['price', 'sales', 'regression']):
            return "RMSE or MAE (regression)"
        elif any(word in name_lower for word in ['classification', 'survived', 'class']):
            return "Accuracy or AUC (classification)"
        elif 'forecast' in name_lower or 'time' in name_lower:
            return "Time series accuracy"
        else:
            return "Competition-specific metric"

    def analyze_competition_with_llm(self, competition_info: Dict[str, Any]) -> Optional[CompetitionInsights]:
        """Use LLM to analyze competition and generate strategic insights"""
        print("Analyzing competition with LLM...")

        # Create prompt for competition analysis
        prompt = PromptTemplates.competition_analysis(
            title=competition_info["title"],
            description=LLMUtils.truncate_text(competition_info["description"], 1500),
            evaluation_metric=competition_info["evaluation_metric"],
            dataset_files=competition_info["dataset_files"]
        )

        # Get structured insights from LLM
        llm_response = self.llm.structured_output(
            prompt,
            agent="competition_reader",
            model_type="primary",
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=2500
        )

        if not llm_response:
            print("Failed to get LLM analysis for competition")
            return None

        # Validate and structure the response
        required_keys = [
            "problem_type", "difficulty_level", "key_strategies", "evaluation_focus",
            "common_pitfalls", "feature_opportunities", "model_recommendations"
        ]

        if not LLMUtils.validate_json_structure(llm_response, required_keys):
            print("LLM response missing required fields")
            return None

        # Calculate confidence score based on response quality
        confidence_score = self._calculate_confidence(llm_response, competition_info)

        # Create structured insights
        insights = CompetitionInsights(
            competition_name=self.competition_name,
            problem_type=llm_response.get("problem_type", "unknown"),
            difficulty_level=str(llm_response.get("difficulty_level", "unknown")),
            key_strategies=llm_response.get("key_strategies", [])[:5],  # Top 5
            evaluation_focus=llm_response.get("evaluation_focus", "unknown"),
            common_pitfalls=llm_response.get("common_pitfalls", [])[:3],  # Top 3
            feature_opportunities=llm_response.get("feature_opportunities", [])[:5],
            model_recommendations=llm_response.get("model_recommendations", [])[:5],
            time_complexity=str(llm_response.get("time_complexity", "unknown")),
            data_concerns=llm_response.get("data_concerns", [])[:3],
            success_factors=llm_response.get("success_factors", [])[:3],
            analysis_timestamp=datetime.now().isoformat(),
            confidence_score=confidence_score
        )

        return insights

    def _calculate_confidence(self, llm_response: Dict, competition_info: Dict) -> float:
        """Calculate confidence score for LLM analysis quality"""
        confidence = 0.0
        max_confidence = 1.0

        # Check if we have actual competition data vs inferred
        if competition_info["data_source"] == "kaggle_api":
            confidence += 0.3
        else:
            confidence += 0.1

        # Check response completeness
        response_completeness = sum(1 for value in llm_response.values() if value)
        confidence += min(0.3, response_completeness * 0.03)

        # Check if strategies and recommendations seem reasonable
        if len(llm_response.get("key_strategies", [])) >= 3:
            confidence += 0.2

        if len(llm_response.get("model_recommendations", [])) >= 3:
            confidence += 0.2

        return min(confidence, max_confidence)

    def save_insights(self, insights: CompetitionInsights) -> None:
        """Save competition insights to JSON file"""
        insights_dict = asdict(insights)

        with open(self.insights_file, 'w') as f:
            json.dump(insights_dict, f, indent=2)

        print(f"Competition insights saved to: {self.insights_file}")

    def load_existing_insights(self) -> Optional[CompetitionInsights]:
        """Load existing competition insights if available"""
        if not self.insights_file.exists():
            return None

        try:
            with open(self.insights_file, 'r') as f:
                data = json.load(f)

            # Check if insights are recent (within 7 days)
            if 'analysis_timestamp' in data:
                analysis_time = datetime.fromisoformat(data['analysis_timestamp'])
                days_old = (datetime.now() - analysis_time).days

                if days_old > 7:
                    print(f"Existing insights are {days_old} days old, will refresh")
                    return None

            return CompetitionInsights(**data)

        except Exception as e:
            print(f"Could not load existing insights: {e}")
            return None

    def print_insights_summary(self, insights: CompetitionInsights) -> None:
        """Print a human-readable summary of competition insights"""
        print(f"\nCompetition Intelligence Summary: {insights.competition_name}")
        print("=" * 60)
        print(f"Problem Type: {insights.problem_type}")
        print(f"Difficulty: {insights.difficulty_level}")
        print(f"Evaluation Focus: {insights.evaluation_focus}")
        print(f"Time Estimate: {insights.time_complexity}")
        print(f"Confidence Score: {insights.confidence_score:.2f}")

        print(f"\nKey Strategies:")
        for i, strategy in enumerate(insights.key_strategies, 1):
            print(f"  {i}. {strategy}")

        print(f"\nRecommended Models:")
        for i, model in enumerate(insights.model_recommendations, 1):
            print(f"  {i}. {model}")

        print(f"\nFeature Opportunities:")
        for i, feature in enumerate(insights.feature_opportunities, 1):
            print(f"  {i}. {feature}")

        print(f"\nCommon Pitfalls to Avoid:")
        for i, pitfall in enumerate(insights.common_pitfalls, 1):
            print(f"  {i}. {pitfall}")

        if insights.data_concerns:
            print(f"\nData Concerns:")
            for i, concern in enumerate(insights.data_concerns, 1):
                print(f"  {i}. {concern}")

    def run_competition_analysis(self, force_refresh: bool = False) -> Optional[CompetitionInsights]:
        """Run complete competition analysis pipeline"""
        print(f"Competition Intelligence Agent: {self.competition_name}")
        print("=" * 50)

        # Check for existing insights
        if not force_refresh:
            existing_insights = self.load_existing_insights()
            if existing_insights:
                print("Using existing competition insights (use --force to refresh)")
                self.print_insights_summary(existing_insights)
                return existing_insights

        # Fetch competition information
        print("Step 1: Fetching competition information...")
        competition_info = self.fetch_competition_info()

        print(f"Competition: {competition_info['title']}")
        print(f"Metric: {competition_info['evaluation_metric']}")
        print(f"Files: {', '.join(competition_info['dataset_files'])}")

        # Analyze with LLM
        print("\nStep 2: Analyzing with LLM...")
        insights = self.analyze_competition_with_llm(competition_info)

        if not insights:
            print("Failed to generate competition insights")
            return None

        # Save insights
        print("\nStep 3: Saving insights...")
        self.save_insights(insights)

        # Print summary
        self.print_insights_summary(insights)

        print(f"\nCompetition analysis complete!")
        return insights


def main():
    """Main entry point for Competition Reader Agent"""
    parser = argparse.ArgumentParser(description="Competition Intelligence Agent")
    parser.add_argument("competition_path", type=Path,
                       help="Path to competition directory")
    parser.add_argument("--force", action="store_true",
                       help="Force refresh of existing insights")

    args = parser.parse_args()

    # Validate competition path
    if not args.competition_path.exists():
        print(f"ERROR: Competition path does not exist: {args.competition_path}")
        return 1

    train_csv = args.competition_path / "train.csv"
    if not train_csv.exists():
        print(f"ERROR: Training data not found: {train_csv}")
        return 1

    try:
        # Initialize and run competition reader
        reader = CompetitionReader(args.competition_path)
        insights = reader.run_competition_analysis(force_refresh=args.force)

        if insights:
            print(f"\nSuccess! Competition insights available for other agents.")
            print(f"Confidence: {insights.confidence_score:.1%}")
            return 0
        else:
            print(f"\nFailed to analyze competition")
            return 1

    except Exception as e:
        print(f"ERROR: Competition analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())