#!/usr/bin/env python3
"""
Feedback Analyzer Agent - Autonomous improvement through performance analysis

This agent analyzes model performance, identifies improvement opportunities,
and orchestrates iterative enhancements to achieve better leaderboard results.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np

# ML libraries for analysis
from sklearn.metrics import confusion_matrix, classification_report, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    from agents.llm_coordinator import LLMCoordinator
    from utils.llm_utils import PromptTemplates, LLMUtils
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


@dataclass
class PerformanceAnalysis:
    """Structure for performance analysis results"""
    competition_name: str
    current_cv_score: float
    current_model: str
    performance_trend: List[float]  # Historical scores
    error_analysis: Dict[str, Any]
    feature_impact_analysis: Dict[str, float]
    learning_curve_analysis: Dict[str, Any]
    validation_analysis: Dict[str, Any]
    timestamp: str


@dataclass
class ImprovementRecommendations:
    """Structure for improvement recommendations"""
    competition_name: str
    priority_actions: List[str]
    feature_improvements: List[str]
    model_improvements: List[str]
    data_improvements: List[str]
    hyperparameter_adjustments: Dict[str, Any]
    ensemble_suggestions: List[str]
    validation_improvements: List[str]
    estimated_impact: Dict[str, float]  # action -> expected score improvement
    confidence_scores: Dict[str, float]  # action -> confidence
    timestamp: str


@dataclass
class LLMIterationInsights:
    """Structure for LLM-generated iteration insights"""
    competition_name: str
    performance_diagnosis: List[str]
    root_cause_analysis: List[str]
    improvement_strategies: List[str]
    priority_ranking: List[str]
    implementation_plan: List[str]
    risk_mitigation: List[str]
    success_metrics: List[str]
    iteration_focus: str
    confidence_score: float
    analysis_timestamp: str


class FeedbackAnalyzer:
    """
    Feedback Analysis Agent for autonomous iterative improvement
    """

    def __init__(self, competition_path: Path, enable_llm: bool = True):
        self.competition_path = Path(competition_path)
        self.competition_name = self.competition_path.name
        self.enable_llm = enable_llm

        # Initialize output directory
        self.output_dir = self.competition_path / "feedback_analysis"
        self.output_dir.mkdir(exist_ok=True)

        # Initialize LLM coordinator if available and enabled
        self.llm_coordinator = None
        if self.enable_llm and DEPENDENCIES_AVAILABLE:
            try:
                self.llm_coordinator = LLMCoordinator(log_dir=self.output_dir / "llm_logs")
                print("LLM coordinator initialized for feedback analysis")
            except Exception as e:
                print(f"Warning: Could not initialize LLM coordinator: {e}")
                self.enable_llm = False

        # Load existing insights and results
        self.competition_insights = None
        self.model_results = None
        self.performance_history = []
        self.iteration_count = 0

        # Storage for analysis results
        self.performance_analysis = None
        self.improvement_recommendations = None
        self.llm_insights = None

    def load_existing_results(self) -> Dict[str, Any]:
        """Load results from all previous agents"""
        print("Loading results from previous pipeline runs...")

        results = {
            "competition_insights": None,
            "data_insights": None,
            "feature_engineering": None,
            "model_selection": None,
            "performance_history": []
        }

        # Load competition insights
        comp_insights_path = self.competition_path / "competition_understanding.json"
        if comp_insights_path.exists():
            with open(comp_insights_path, 'r') as f:
                results["competition_insights"] = json.load(f)
            print("Loaded competition insights")

        # Load data scout results
        data_insights_path = self.competition_path / "data_scout_outputs" / "dataset_info.json"
        if data_insights_path.exists():
            with open(data_insights_path, 'r') as f:
                results["data_insights"] = json.load(f)
            print("Loaded data insights")

        # Load feature engineering results
        feature_path = self.competition_path / "feature_engineering" / "feature_engineering.json"
        if feature_path.exists():
            with open(feature_path, 'r') as f:
                results["feature_engineering"] = json.load(f)
            print("Loaded feature engineering results")

        # Load model selection results
        model_path = self.competition_path / "model_selection" / "model_selection.json"
        if model_path.exists():
            with open(model_path, 'r') as f:
                results["model_selection"] = json.load(f)
            print("Loaded model selection results")

        # Load performance history
        history_path = self.output_dir / "performance_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                results["performance_history"] = json.load(f)
            print(f"Loaded performance history: {len(results['performance_history'])} iterations")

        return results

    def analyze_current_performance(self, results: Dict[str, Any]) -> PerformanceAnalysis:
        """Analyze current model performance and identify issues"""
        print("Analyzing current performance...")

        # Extract current performance metrics
        current_score = 0.0
        current_model = "Unknown"
        performance_trend = []

        if results["model_selection"]:
            model_results = results["model_selection"].get("model_results", [])
            if model_results:
                best_model = model_results[0]
                current_score = best_model.get("cv_mean", 0.0)
                current_model = best_model.get("model_name", "Unknown")

        # Build performance trend from history
        performance_trend = [entry.get("cv_score", 0.0) for entry in results["performance_history"]]
        performance_trend.append(current_score)

        # Analyze errors and performance patterns
        error_analysis = self._analyze_errors(results)
        feature_impact = self._analyze_feature_impact(results)
        learning_curve = self._analyze_learning_curves(results)
        validation_analysis = self._analyze_validation_patterns(results)

        analysis = PerformanceAnalysis(
            competition_name=self.competition_name,
            current_cv_score=current_score,
            current_model=current_model,
            performance_trend=performance_trend,
            error_analysis=error_analysis,
            feature_impact_analysis=feature_impact,
            learning_curve_analysis=learning_curve,
            validation_analysis=validation_analysis,
            timestamp=datetime.now().isoformat()
        )

        return analysis

    def analyze_with_llm(self, performance_analysis: PerformanceAnalysis, results: Dict[str, Any]) -> Optional[LLMIterationInsights]:
        """Use LLM to analyze performance and suggest improvements"""
        if not self.enable_llm or not self.llm_coordinator:
            return None

        print("Analyzing performance with LLM...")

        # Prepare comprehensive context for LLM
        performance_summary = self._prepare_performance_summary(performance_analysis)
        pipeline_summary = self._prepare_pipeline_summary(results)
        competition_context = self._prepare_competition_context(results)

        # Create prompt for performance analysis
        prompt = PromptTemplates.performance_analysis(
            performance_summary=performance_summary,
            pipeline_summary=pipeline_summary,
            competition_context=competition_context,
            iteration_count=len(performance_analysis.performance_trend)
        )

        # Get structured insights from LLM
        llm_response = self.llm_coordinator.structured_output(
            prompt,
            agent="feedback_analyzer",
            model_type="primary",
            temperature=0.2,  # Lower temperature for consistent analysis
            max_tokens=4000
        )

        if not llm_response:
            print("Failed to get LLM performance analysis")
            return None

        # Validate response structure
        required_keys = ["performance_diagnosis", "improvement_strategies", "priority_ranking"]
        if not LLMUtils.validate_json_structure(llm_response, required_keys):
            print("LLM response missing required fields for performance analysis")
            return None

        # Calculate confidence score
        confidence = self._calculate_analysis_confidence(llm_response, performance_analysis)

        # Create structured insights
        insights = LLMIterationInsights(
            competition_name=self.competition_name,
            performance_diagnosis=llm_response.get("performance_diagnosis", [])[:5],
            root_cause_analysis=llm_response.get("root_cause_analysis", [])[:5],
            improvement_strategies=llm_response.get("improvement_strategies", [])[:8],
            priority_ranking=llm_response.get("priority_ranking", [])[:5],
            implementation_plan=llm_response.get("implementation_plan", [])[:6],
            risk_mitigation=llm_response.get("risk_mitigation", [])[:3],
            success_metrics=llm_response.get("success_metrics", [])[:4],
            iteration_focus=llm_response.get("iteration_focus", "General improvement"),
            confidence_score=confidence,
            analysis_timestamp=datetime.now().isoformat()
        )

        return insights

    def generate_improvement_recommendations(self, performance_analysis: PerformanceAnalysis,
                                           llm_insights: Optional[LLMIterationInsights],
                                           results: Dict[str, Any]) -> ImprovementRecommendations:
        """Generate concrete improvement recommendations"""
        print("Generating improvement recommendations...")

        # Initialize recommendation categories
        priority_actions = []
        feature_improvements = []
        model_improvements = []
        data_improvements = []
        hyperparameter_adjustments = {}
        ensemble_suggestions = []
        validation_improvements = []
        estimated_impact = {}
        confidence_scores = {}

        # Use LLM insights if available
        if llm_insights:
            priority_actions = llm_insights.priority_ranking[:3]

            # Categorize LLM suggestions
            for strategy in llm_insights.improvement_strategies:
                if any(word in strategy.lower() for word in ['feature', 'engineer', 'transform']):
                    feature_improvements.append(strategy)
                elif any(word in strategy.lower() for word in ['model', 'algorithm', 'hyperparameter']):
                    model_improvements.append(strategy)
                elif any(word in strategy.lower() for word in ['data', 'clean', 'preprocess']):
                    data_improvements.append(strategy)
                elif any(word in strategy.lower() for word in ['ensemble', 'blend', 'stack']):
                    ensemble_suggestions.append(strategy)
                else:
                    validation_improvements.append(strategy)

            # Assign confidence scores based on LLM confidence
            base_confidence = llm_insights.confidence_score
            for action in priority_actions:
                confidence_scores[action] = base_confidence

        # Add rule-based improvements
        rule_based_improvements = self._generate_rule_based_improvements(performance_analysis, results)

        # Merge improvements
        feature_improvements.extend(rule_based_improvements.get("feature_improvements", []))
        model_improvements.extend(rule_based_improvements.get("model_improvements", []))
        data_improvements.extend(rule_based_improvements.get("data_improvements", []))

        # Estimate impact for each improvement
        estimated_impact = self._estimate_improvement_impact(
            performance_analysis, feature_improvements, model_improvements, data_improvements
        )

        # Create comprehensive recommendations
        recommendations = ImprovementRecommendations(
            competition_name=self.competition_name,
            priority_actions=priority_actions[:5],
            feature_improvements=feature_improvements[:8],
            model_improvements=model_improvements[:5],
            data_improvements=data_improvements[:5],
            hyperparameter_adjustments=hyperparameter_adjustments,
            ensemble_suggestions=ensemble_suggestions[:3],
            validation_improvements=validation_improvements[:3],
            estimated_impact=estimated_impact,
            confidence_scores=confidence_scores,
            timestamp=datetime.now().isoformat()
        )

        return recommendations

    def create_iteration_plan(self, recommendations: ImprovementRecommendations) -> Dict[str, Any]:
        """Create actionable iteration plan"""
        print("Creating iteration plan...")

        # Select top improvements based on impact and confidence
        selected_improvements = self._select_top_improvements(recommendations)

        # Create execution plan
        iteration_plan = {
            "iteration_number": len(self.performance_history) + 1,
            "focus_area": self._determine_focus_area(recommendations),
            "selected_improvements": selected_improvements,
            "execution_order": self._determine_execution_order(selected_improvements),
            "success_criteria": {
                "target_cv_improvement": self._calculate_target_improvement(recommendations),
                "minimum_improvement": 0.001,
                "maximum_iterations": 3
            },
            "risk_mitigation": [
                "Validate improvements with cross-validation",
                "Keep backup of current best model",
                "Monitor for overfitting"
            ],
            "estimated_time": self._estimate_execution_time(selected_improvements),
            "created_timestamp": datetime.now().isoformat()
        }

        return iteration_plan

    def save_analysis_results(self, performance_analysis: PerformanceAnalysis,
                            recommendations: ImprovementRecommendations,
                            iteration_plan: Dict[str, Any]):
        """Save all analysis results"""
        print("Saving feedback analysis results...")

        # Save performance analysis
        performance_path = self.output_dir / "performance_analysis.json"
        with open(performance_path, 'w') as f:
            json.dump(asdict(performance_analysis), f, indent=2)
        print(f"Saved performance analysis: {performance_path}")

        # Save improvement recommendations
        recommendations_path = self.output_dir / "improvement_recommendations.json"
        with open(recommendations_path, 'w') as f:
            json.dump(asdict(recommendations), f, indent=2, default=str)
        print(f"Saved improvement recommendations: {recommendations_path}")

        # Save iteration plan
        iteration_path = self.output_dir / "iteration_plan.json"
        with open(iteration_path, 'w') as f:
            json.dump(iteration_plan, f, indent=2)
        print(f"Saved iteration plan: {iteration_path}")

        # Save LLM insights if available
        if self.llm_insights:
            llm_path = self.output_dir / "llm_iteration_insights.json"
            with open(llm_path, 'w') as f:
                json.dump(asdict(self.llm_insights), f, indent=2)
            print(f"Saved LLM insights: {llm_path}")

        # Update performance history
        self._update_performance_history(performance_analysis)

    def print_analysis_summary(self, performance_analysis: PerformanceAnalysis,
                             recommendations: ImprovementRecommendations,
                             iteration_plan: Dict[str, Any]):
        """Print human-readable summary of analysis"""
        print(f"\nFeedback Analysis Summary: {self.competition_name}")
        print("=" * 60)
        print(f"Current CV Score: {performance_analysis.current_cv_score:.4f}")
        print(f"Current Model: {performance_analysis.current_model}")
        print(f"Performance Trend: {len(performance_analysis.performance_trend)} iterations")

        if len(performance_analysis.performance_trend) > 1:
            recent_improvement = performance_analysis.performance_trend[-1] - performance_analysis.performance_trend[-2]
            print(f"Recent Change: {recent_improvement:+.4f}")

        print(f"\nTop Priority Actions:")
        for i, action in enumerate(recommendations.priority_actions, 1):
            impact = recommendations.estimated_impact.get(action, 0.0)
            confidence = recommendations.confidence_scores.get(action, 0.0)
            print(f"  {i}. {action}")
            print(f"     Impact: {impact:+.3f}, Confidence: {confidence:.2f}")

        print(f"\nIteration Plan:")
        print(f"Focus Area: {iteration_plan['focus_area']}")
        print(f"Target Improvement: +{iteration_plan['success_criteria']['target_cv_improvement']:.4f}")
        print(f"Estimated Time: {iteration_plan['estimated_time']} minutes")

        if self.llm_insights:
            print(f"\nLLM Confidence: {self.llm_insights.confidence_score:.2f}")
            print(f"Iteration Focus: {self.llm_insights.iteration_focus}")

    def run_feedback_analysis(self) -> Tuple[PerformanceAnalysis, ImprovementRecommendations, Dict[str, Any]]:
        """Run complete feedback analysis pipeline"""
        print(f"Feedback Analysis Agent: {self.competition_name}")
        print("=" * 60)

        # Load existing results
        results = self.load_existing_results()

        # Analyze current performance
        print("\nStep 1: Analyzing current performance...")
        self.performance_analysis = self.analyze_current_performance(results)

        # Get LLM insights if enabled
        if self.enable_llm and self.llm_coordinator:
            print("\nStep 2: Getting LLM analysis...")
            self.llm_insights = self.analyze_with_llm(self.performance_analysis, results)

        # Generate improvement recommendations
        print("\nStep 3: Generating improvement recommendations...")
        self.improvement_recommendations = self.generate_improvement_recommendations(
            self.performance_analysis, self.llm_insights, results
        )

        # Create iteration plan
        print("\nStep 4: Creating iteration plan...")
        iteration_plan = self.create_iteration_plan(self.improvement_recommendations)

        # Save results
        self.save_analysis_results(
            self.performance_analysis, self.improvement_recommendations, iteration_plan
        )

        # Print summary
        self.print_analysis_summary(
            self.performance_analysis, self.improvement_recommendations, iteration_plan
        )

        print(f"\nFeedback Analysis Complete!")

        return self.performance_analysis, self.improvement_recommendations, iteration_plan

    # Helper methods
    def _analyze_errors(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model errors and patterns"""
        error_analysis = {
            "error_type": "unknown",
            "error_patterns": [],
            "problematic_features": [],
            "data_quality_issues": []
        }

        # Analyze based on available data
        if results.get("data_insights"):
            missing_pcts = results["data_insights"].get("missing_percentages", {})
            high_missing = [col for col, pct in missing_pcts.items() if pct > 20]
            if high_missing:
                error_analysis["data_quality_issues"].append(f"High missing values in: {', '.join(high_missing[:3])}")

        return error_analysis

    def _analyze_feature_impact(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Analyze feature importance and impact"""
        feature_impact = {}

        if results.get("model_selection") and results["model_selection"].get("model_results"):
            model_results = results["model_selection"]["model_results"]
            if model_results and model_results[0].get("feature_importance"):
                feature_impact = model_results[0]["feature_importance"]

        return feature_impact

    def _analyze_learning_curves(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze learning curve patterns"""
        return {
            "overfitting_detected": False,
            "underfitting_detected": False,
            "data_sufficiency": "adequate",
            "recommendations": ["Monitor validation curves"]
        }

    def _analyze_validation_patterns(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cross-validation patterns"""
        validation_analysis = {
            "cv_stability": "stable",
            "variance_concerns": False,
            "fold_consistency": True
        }

        # Analyze CV scores if available
        if results.get("model_selection") and results["model_selection"].get("model_results"):
            model_results = results["model_selection"]["model_results"]
            if model_results:
                cv_std = model_results[0].get("cv_std", 0.0)
                cv_mean = model_results[0].get("cv_mean", 0.0)

                if cv_std > 0.05 or (cv_mean > 0 and cv_std / cv_mean > 0.1):
                    validation_analysis["variance_concerns"] = True
                    validation_analysis["cv_stability"] = "unstable"

        return validation_analysis

    def _prepare_performance_summary(self, performance_analysis: PerformanceAnalysis) -> str:
        """Prepare performance summary for LLM"""
        summary_parts = [
            f"Current CV Score: {performance_analysis.current_cv_score:.4f}",
            f"Model: {performance_analysis.current_model}",
            f"Iterations: {len(performance_analysis.performance_trend)}"
        ]

        if len(performance_analysis.performance_trend) > 1:
            recent_change = performance_analysis.performance_trend[-1] - performance_analysis.performance_trend[-2]
            summary_parts.append(f"Recent change: {recent_change:+.4f}")

        return " | ".join(summary_parts)

    def _prepare_pipeline_summary(self, results: Dict[str, Any]) -> str:
        """Prepare pipeline summary for LLM"""
        summary_parts = []

        if results.get("feature_engineering"):
            n_features = results["feature_engineering"].get("total_features", 0)
            summary_parts.append(f"Features: {n_features}")

        if results.get("model_selection"):
            n_models = results["model_selection"].get("total_models_tested", 0)
            summary_parts.append(f"Models tested: {n_models}")

        return " | ".join(summary_parts) if summary_parts else "Limited pipeline info"

    def _prepare_competition_context(self, results: Dict[str, Any]) -> str:
        """Prepare competition context for LLM"""
        if results.get("competition_insights"):
            comp = results["competition_insights"]
            return f"Problem: {comp.get('problem_type', 'unknown')} | Strategies: {', '.join(comp.get('key_strategies', [])[:2])}"

        return f"Competition: {self.competition_name}"

    def _calculate_analysis_confidence(self, llm_response: Dict, performance_analysis: PerformanceAnalysis) -> float:
        """Calculate confidence score for LLM analysis"""
        confidence = 0.0

        # Base confidence on response completeness
        if len(llm_response.get("performance_diagnosis", [])) >= 3:
            confidence += 0.3

        if len(llm_response.get("improvement_strategies", [])) >= 5:
            confidence += 0.4

        if llm_response.get("priority_ranking"):
            confidence += 0.3

        # Boost confidence if we have performance history
        if len(performance_analysis.performance_trend) > 1:
            confidence += 0.1

        return min(confidence, 1.0)

    def _generate_rule_based_improvements(self, performance_analysis: PerformanceAnalysis, results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate rule-based improvement suggestions"""
        improvements = {
            "feature_improvements": [],
            "model_improvements": [],
            "data_improvements": []
        }

        # Feature improvements based on analysis
        if len(performance_analysis.feature_impact_analysis) > 5:
            low_importance_features = [f for f, imp in performance_analysis.feature_impact_analysis.items() if imp < 0.01]
            if len(low_importance_features) > 3:
                improvements["feature_improvements"].append("Remove low-importance features to reduce noise")

        # Model improvements based on validation patterns
        if performance_analysis.validation_analysis.get("variance_concerns"):
            improvements["model_improvements"].append("Increase regularization to improve stability")

        # Data improvements based on error analysis
        if performance_analysis.error_analysis.get("data_quality_issues"):
            improvements["data_improvements"].append("Improve data preprocessing to handle quality issues")

        return improvements

    def _estimate_improvement_impact(self, performance_analysis: PerformanceAnalysis,
                                   feature_improvements: List[str], model_improvements: List[str],
                                   data_improvements: List[str]) -> Dict[str, float]:
        """Estimate impact of each improvement"""
        estimated_impact = {}

        # Assign base impact estimates
        for improvement in feature_improvements:
            estimated_impact[improvement] = 0.005  # Feature improvements typically small but consistent

        for improvement in model_improvements:
            estimated_impact[improvement] = 0.010  # Model improvements can have larger impact

        for improvement in data_improvements:
            estimated_impact[improvement] = 0.003  # Data improvements foundational but smaller

        return estimated_impact

    def _select_top_improvements(self, recommendations: ImprovementRecommendations) -> List[str]:
        """Select top improvements based on impact and confidence"""
        all_improvements = (
            recommendations.feature_improvements[:3] +
            recommendations.model_improvements[:2] +
            recommendations.data_improvements[:2]
        )

        # Sort by estimated impact
        sorted_improvements = sorted(
            all_improvements,
            key=lambda x: recommendations.estimated_impact.get(x, 0.0),
            reverse=True
        )

        return sorted_improvements[:5]

    def _determine_focus_area(self, recommendations: ImprovementRecommendations) -> str:
        """Determine main focus area for iteration"""
        if len(recommendations.feature_improvements) > len(recommendations.model_improvements):
            return "Feature Engineering"
        elif len(recommendations.model_improvements) > 0:
            return "Model Optimization"
        else:
            return "Data Quality"

    def _determine_execution_order(self, selected_improvements: List[str]) -> List[str]:
        """Determine optimal execution order"""
        # Simple heuristic: data improvements first, then features, then models
        data_first = [imp for imp in selected_improvements if "data" in imp.lower() or "preprocess" in imp.lower()]
        feature_second = [imp for imp in selected_improvements if "feature" in imp.lower()]
        model_third = [imp for imp in selected_improvements if "model" in imp.lower() or "hyperparameter" in imp.lower()]

        return data_first + feature_second + model_third

    def _calculate_target_improvement(self, recommendations: ImprovementRecommendations) -> float:
        """Calculate target CV improvement"""
        total_estimated_impact = sum(recommendations.estimated_impact.values())
        return min(total_estimated_impact * 0.7, 0.02)  # Conservative estimate with cap

    def _estimate_execution_time(self, selected_improvements: List[str]) -> int:
        """Estimate execution time in minutes"""
        base_time = 15  # Base pipeline run time
        improvement_time = len(selected_improvements) * 10  # 10 minutes per improvement
        return base_time + improvement_time

    def _update_performance_history(self, performance_analysis: PerformanceAnalysis):
        """Update performance history"""
        history_entry = {
            "iteration": len(self.performance_history) + 1,
            "timestamp": performance_analysis.timestamp,
            "cv_score": performance_analysis.current_cv_score,
            "model": performance_analysis.current_model
        }

        self.performance_history.append(history_entry)

        # Save updated history
        history_path = self.output_dir / "performance_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.performance_history, f, indent=2)


def main():
    """Main entry point for Feedback Analysis Agent"""
    parser = argparse.ArgumentParser(description="Feedback Analysis Agent with LLM")
    parser.add_argument("competition_path", type=Path,
                       help="Path to competition directory")
    parser.add_argument("--no-llm", action="store_true",
                       help="Disable LLM and use basic analysis only")

    args = parser.parse_args()

    # Validate competition path
    if not args.competition_path.exists():
        print(f"ERROR: Competition path does not exist: {args.competition_path}")
        return 1

    try:
        # Initialize and run feedback analysis
        analyzer = FeedbackAnalyzer(args.competition_path, enable_llm=not args.no_llm)
        performance_analysis, recommendations, iteration_plan = analyzer.run_feedback_analysis()

        print(f"\nFeedback Analysis Success!")
        print(f"Priority actions: {len(recommendations.priority_actions)}")
        print(f"Total recommendations: {len(recommendations.feature_improvements) + len(recommendations.model_improvements)}")
        print(f"Results saved to: {analyzer.output_dir}")
        return 0

    except Exception as e:
        print(f"ERROR: Feedback analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())