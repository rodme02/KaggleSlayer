"""
LLM-driven insight generation for competition strategy and model improvement.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import json
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CompetitionInsight:
    """Insight about competition strategy."""
    insight_type: str  # "strategy", "feature", "model", "data"
    title: str
    description: str
    actionable_steps: List[str]
    priority: str  # "high", "medium", "low"
    confidence: float  # 0.0 to 1.0
    timestamp: str


class InsightGenerator:
    """Generates insights using LLM analysis and data patterns."""

    def __init__(self):
        self.insights_history = []

    def generate_competition_insights(self, competition_data: Dict[str, Any],
                                    dataset_info: Dict[str, Any],
                                    performance_history: List[Dict[str, Any]]) -> List[CompetitionInsight]:
        """Generate strategic insights for the competition."""
        insights = []

        # Data-driven insights
        insights.extend(self._analyze_dataset_patterns(dataset_info))

        # Performance-driven insights
        if performance_history:
            insights.extend(self._analyze_performance_patterns(performance_history))

        # Competition-specific insights
        insights.extend(self._analyze_competition_characteristics(competition_data))

        # Store insights
        self.insights_history.extend(insights)

        return insights

    def _analyze_dataset_patterns(self, dataset_info: Dict[str, Any]) -> List[CompetitionInsight]:
        """Analyze dataset characteristics for insights."""
        insights = []

        # Check for imbalanced data
        if 'target_distribution' in dataset_info:
            dist = dataset_info['target_distribution']
            if isinstance(dist, dict):
                values = list(dist.values())
                if max(values) / min(values) > 10:  # Highly imbalanced
                    insights.append(CompetitionInsight(
                        insight_type="data",
                        title="Highly Imbalanced Dataset Detected",
                        description="The target variable shows significant class imbalance which may affect model performance.",
                        actionable_steps=[
                            "Apply SMOTE or other oversampling techniques",
                            "Use stratified sampling for train/validation splits",
                            "Consider class weights in model training",
                            "Focus on precision/recall rather than accuracy"
                        ],
                        priority="high",
                        confidence=0.9,
                        timestamp=datetime.now().isoformat()
                    ))

        # Check for high missing data
        if 'missing_percentages' in dataset_info:
            missing_data = dataset_info['missing_percentages']
            high_missing_cols = [col for col, pct in missing_data.items() if pct > 0.3]
            if len(high_missing_cols) > 3:
                insights.append(CompetitionInsight(
                    insight_type="data",
                    title="Extensive Missing Data Pattern",
                    description=f"Multiple columns ({len(high_missing_cols)}) have >30% missing values.",
                    actionable_steps=[
                        "Investigate if missing data is informative (MNAR)",
                        "Create 'missingness' indicator features",
                        "Consider advanced imputation techniques like KNN or iterative",
                        "Evaluate impact of dropping high-missing columns"
                    ],
                    priority="medium",
                    confidence=0.8,
                    timestamp=datetime.now().isoformat()
                ))

        # Check feature count vs sample size
        if 'total_rows' in dataset_info and 'total_columns' in dataset_info:
            ratio = dataset_info['total_columns'] / dataset_info['total_rows']
            if ratio > 0.1:  # High-dimensional data
                insights.append(CompetitionInsight(
                    insight_type="feature",
                    title="High-Dimensional Dataset",
                    description="Feature count is high relative to sample size, risk of overfitting.",
                    actionable_steps=[
                        "Apply dimensionality reduction (PCA, t-SNE)",
                        "Use feature selection techniques aggressively",
                        "Consider regularized models (Lasso, Ridge)",
                        "Implement cross-validation carefully"
                    ],
                    priority="high",
                    confidence=0.85,
                    timestamp=datetime.now().isoformat()
                ))

        return insights

    def _analyze_performance_patterns(self, performance_history: List[Dict[str, Any]]) -> List[CompetitionInsight]:
        """Analyze performance trends for insights."""
        insights = []

        if len(performance_history) < 2:
            return insights

        # Check for performance stagnation
        recent_scores = [p.get('validation_score', 0) for p in performance_history[-5:]]
        if len(recent_scores) >= 3:
            score_variance = max(recent_scores) - min(recent_scores)
            if score_variance < 0.01:  # Very little improvement
                insights.append(CompetitionInsight(
                    insight_type="strategy",
                    title="Performance Plateau Detected",
                    description="Model performance has plateaued in recent iterations.",
                    actionable_steps=[
                        "Try different model architectures",
                        "Implement ensemble methods",
                        "Explore feature engineering approaches",
                        "Consider hyperparameter optimization",
                        "Investigate data quality issues"
                    ],
                    priority="high",
                    confidence=0.9,
                    timestamp=datetime.now().isoformat()
                ))

        # Check for overfitting patterns
        overfitting_cases = 0
        for perf in performance_history[-3:]:
            train_score = perf.get('train_score', 0)
            val_score = perf.get('validation_score', 0)
            if train_score - val_score > 0.1:
                overfitting_cases += 1

        if overfitting_cases >= 2:
            insights.append(CompetitionInsight(
                insight_type="model",
                title="Consistent Overfitting Pattern",
                description="Models consistently show large train-validation score gaps.",
                actionable_steps=[
                    "Increase regularization strength",
                    "Reduce model complexity",
                    "Add more training data if possible",
                    "Implement early stopping",
                    "Use dropout or other regularization techniques"
                ],
                priority="high",
                confidence=0.85,
                timestamp=datetime.now().isoformat()
            ))

        return insights

    def _analyze_competition_characteristics(self, competition_data: Dict[str, Any]) -> List[CompetitionInsight]:
        """Analyze competition-specific characteristics."""
        insights = []

        # Time series detection
        if 'dataset_files' in competition_data:
            files = competition_data['dataset_files']
            if any('time' in file.lower() or 'date' in file.lower() for file in files):
                insights.append(CompetitionInsight(
                    insight_type="strategy",
                    title="Potential Time Series Competition",
                    description="Dataset suggests temporal components that require special handling.",
                    actionable_steps=[
                        "Implement time-aware cross-validation",
                        "Create lag features and rolling statistics",
                        "Consider time series models (ARIMA, Prophet)",
                        "Avoid data leakage from future information",
                        "Analyze seasonal patterns"
                    ],
                    priority="high",
                    confidence=0.7,
                    timestamp=datetime.now().isoformat()
                ))

        # Text/NLP detection
        if 'feature_types' in competition_data:
            text_features = [f for f, t in competition_data['feature_types'].items() if t == 'text']
            if len(text_features) > 0:
                insights.append(CompetitionInsight(
                    insight_type="feature",
                    title="Text Features Detected",
                    description=f"Found {len(text_features)} text features requiring NLP processing.",
                    actionable_steps=[
                        "Implement TF-IDF or word embeddings",
                        "Try sentiment analysis features",
                        "Consider text length and complexity metrics",
                        "Explore topic modeling (LDA)",
                        "Use pre-trained language models if appropriate"
                    ],
                    priority="medium",
                    confidence=0.8,
                    timestamp=datetime.now().isoformat()
                ))

        return insights

    def prioritize_insights(self, insights: List[CompetitionInsight]) -> List[CompetitionInsight]:
        """Prioritize insights by importance and confidence."""
        priority_scores = {"high": 3, "medium": 2, "low": 1}

        def score_insight(insight):
            return priority_scores[insight.priority] * insight.confidence

        return sorted(insights, key=score_insight, reverse=True)

    def generate_insight_report(self, insights: List[CompetitionInsight]) -> Dict[str, Any]:
        """Generate a comprehensive insight report."""
        if not insights:
            return {'message': 'No insights available'}

        prioritized_insights = self.prioritize_insights(insights)

        # Group by type
        by_type = {}
        for insight in prioritized_insights:
            if insight.insight_type not in by_type:
                by_type[insight.insight_type] = []
            by_type[insight.insight_type].append(insight)

        # Create summary
        summary = {
            'total_insights': len(insights),
            'high_priority_count': len([i for i in insights if i.priority == "high"]),
            'by_type_count': {t: len(insights_list) for t, insights_list in by_type.items()},
            'avg_confidence': sum(i.confidence for i in insights) / len(insights),
            'top_priorities': [
                {
                    'title': insight.title,
                    'type': insight.insight_type,
                    'priority': insight.priority,
                    'confidence': insight.confidence,
                    'actionable_steps': insight.actionable_steps[:3]  # Top 3 steps
                }
                for insight in prioritized_insights[:5]
            ],
            'actionable_summary': self._create_actionable_summary(prioritized_insights[:10])
        }

        return summary

    def _create_actionable_summary(self, top_insights: List[CompetitionInsight]) -> List[str]:
        """Create a summary of actionable steps."""
        all_steps = []
        for insight in top_insights:
            all_steps.extend(insight.actionable_steps)

        # Remove duplicates while preserving order
        unique_steps = []
        seen = set()
        for step in all_steps:
            if step not in seen:
                unique_steps.append(step)
                seen.add(step)

        return unique_steps[:15]  # Top 15 actionable steps

    def get_insights_history(self) -> List[CompetitionInsight]:
        """Get all historical insights."""
        return self.insights_history