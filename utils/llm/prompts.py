#!/usr/bin/env python3
"""
LLM Utilities and Prompt Templates for KaggleSlayer

Reusable prompt templates and utility functions for consistent LLM interactions
across all agents in the KaggleSlayer pipeline.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime


class PromptTemplates:
    """Centralized prompt templates for all KaggleSlayer agents"""

    @staticmethod
    def competition_analysis(title: str, description: str, evaluation_metric: str,
                           dataset_files: List[str]) -> str:
        """Template for competition understanding and strategy"""
        return f"""
You are an expert Kaggle competitor analyzing a new competition. Provide strategic insights.

**COMPETITION DETAILS:**
Title: {title}
Description: {description}
Evaluation Metric: {evaluation_metric}
Dataset Files: {', '.join(dataset_files)}

**ANALYSIS REQUIRED:**
Analyze this competition and provide insights as JSON with these keys:

1. "problem_type": classification/regression/other
2. "difficulty_level": beginner/intermediate/advanced (1-10 scale)
3. "key_strategies": Array of 5 most important strategies for this competition
4. "evaluation_focus": What the metric emphasizes (precision/recall/speed/etc)
5. "common_pitfalls": Array of 3 mistakes competitors often make
6. "feature_opportunities": Array of 5 potential feature engineering directions
7. "model_recommendations": Array of 5 models ranked by expected performance
8. "time_complexity": estimated hours needed for top 20% solution
9. "data_concerns": potential data quality or leakage issues to watch for
10. "success_factors": top 3 factors that determine winning solutions

Focus on actionable insights that will guide an autonomous agent to achieve top 20% ranking.
"""

    @staticmethod
    def dataset_insights(eda_summary: str, competition_context: str) -> str:
        """Template for data scout LLM insights"""
        return f"""
You are a senior data scientist analyzing a dataset for a Kaggle competition.

**COMPETITION CONTEXT:**
{competition_context}

**DATASET EDA SUMMARY:**
{eda_summary}

**INSIGHTS REQUIRED:**
Based on this EDA, provide insights as JSON with these keys:

1. "data_quality_score": 1-10 rating of overall data quality
2. "critical_issues": Array of serious data problems that must be addressed
3. "preprocessing_priorities": Array of 5 preprocessing steps in order of importance
4. "feature_engineering_ideas": Array of 8 specific feature engineering opportunities
5. "potential_challenges": Array of challenges this dataset presents
6. "target_insights": Analysis of target variable and its characteristics
7. "feature_interactions": Potentially valuable feature combinations to explore
8. "missing_data_strategy": Recommended approach for handling missing values
9. "outlier_strategy": How to handle outliers for this specific problem
10. "validation_strategy": Recommended cross-validation approach and why

Focus on actionable recommendations that will improve model performance.
"""

    @staticmethod
    def feature_engineering_analysis(dataset_summary: str, competition_context: str,
                                   existing_features: List[str]) -> str:
        """Template for analyzing feature engineering opportunities"""
        return f"""
You are an expert data scientist analyzing a Kaggle competition dataset for feature engineering opportunities.

**DATASET SUMMARY:**
{dataset_summary}

**COMPETITION CONTEXT:**
{competition_context}

**EXISTING FEATURES:**
{', '.join(existing_features[:20])}{'...' if len(existing_features) > 20 else ''}

**ANALYSIS TASK:**
Analyze this dataset and provide strategic feature engineering recommendations. Return as JSON:

{{
  "recommended_features": [
    "Feature 1: Description of what to create and why",
    "Feature 2: Mathematical transformation or interaction",
    "Feature 3: Domain-specific insight",
    ...
  ],
  "feature_engineering_strategies": [
    "Strategy 1: Overall approach (e.g., 'Focus on polynomial interactions')",
    "Strategy 2: Data-driven insight (e.g., 'Address missing value patterns')",
    "Strategy 3: Domain knowledge (e.g., 'Create time-based features')",
    ...
  ],
  "interaction_opportunities": [
    "feature1 * feature2: Reasoning why this interaction matters",
    "feature3 / feature4: Ratio that captures important relationship",
    ...
  ],
  "transformation_suggestions": [
    "log(feature): To handle skewed distribution",
    "sqrt(feature): To reduce outlier impact",
    "feature^2: To capture non-linear relationships",
    ...
  ],
  "domain_specific_features": [
    "Feature based on domain knowledge for this problem type",
    "Industry-specific calculation or insight",
    ...
  ],
  "code_templates": [
    "df['new_feature'] = df['old_feature'].apply(lambda x: transformation)",
    "df['interaction'] = df['feature1'] * df['feature2']",
    ...
  ],
  "risk_assessment": [
    "Risk 1: Potential overfitting concern",
    "Risk 2: Data leakage warning",
    "Risk 3: Computational complexity issue"
  ]
}}

Focus on creating features that will genuinely improve model performance for this specific competition type.
"""

    @staticmethod
    def model_selection_analysis(dataset_summary: str, competition_context: str,
                                problem_context: str, available_models: List[str]) -> str:
        """Template for analyzing optimal model selection and hyperparameters"""
        return f"""
You are an expert machine learning engineer selecting optimal models for a Kaggle competition.

**DATASET SUMMARY:**
{dataset_summary}

**COMPETITION CONTEXT:**
{competition_context}

**PROBLEM CONTEXT:**
{problem_context}

**AVAILABLE MODELS:**
{', '.join(available_models)}

**MODEL SELECTION TASK:**
Analyze this competition and recommend the best models and hyperparameters. Return as JSON:

{{
  "recommended_models": [
    "Model1: Primary recommendation with reasoning",
    "Model2: Strong alternative with specific strengths",
    "Model3: Ensemble component or specialized use case",
    ...
  ],
  "model_rationale": {{
    "Model1": "Why this model fits the problem characteristics",
    "Model2": "Specific advantages for this dataset/competition",
    "Model3": "Role in ensemble or specific scenario"
  }},
  "hyperparameter_suggestions": {{
    "Model1": {{
      "param1": {{"type": "int", "range": [min, max], "reasoning": "why this range"}},
      "param2": {{"type": "float", "range": [min, max], "reasoning": "optimization focus"}},
      "param3": ["option1", "option2", "option3"]
    }},
    "Model2": {{
      "param1": value,
      "param2": {{"type": "int", "range": [min, max]}}
    }}
  }},
  "ensemble_strategy": "Specific ensemble approach (stacking, blending, simple average, etc.)",
  "validation_approach": "Optimal CV strategy for this problem",
  "feature_selection_advice": [
    "Feature selection method 1: Why it fits this problem",
    "Feature selection method 2: Specific technique for this data",
    ...
  ],
  "optimization_priorities": [
    "Priority 1: Most important metric to optimize",
    "Priority 2: Secondary consideration",
    "Priority 3: Efficiency or interpretability concern"
  ],
  "risk_assessment": [
    "Risk 1: Overfitting concern and mitigation",
    "Risk 2: Model complexity vs performance tradeoff",
    "Risk 3: Computational or practical limitation"
  ]
}}

Focus on model choices that maximize leaderboard performance for this specific competition type and dataset characteristics.
"""

    @staticmethod
    def performance_analysis(performance_summary: str, pipeline_summary: str,
                           competition_context: str, iteration_count: int) -> str:
        """Template for analyzing performance and suggesting improvements"""
        return f"""
You are an expert machine learning consultant analyzing Kaggle competition performance for iterative improvement.

**CURRENT PERFORMANCE:**
{performance_summary}

**PIPELINE STATUS:**
{pipeline_summary}

**COMPETITION CONTEXT:**
{competition_context}

**ITERATION CONTEXT:**
This is iteration #{iteration_count} of the improvement process.

**PERFORMANCE ANALYSIS TASK:**
Analyze the current performance and provide strategic improvement recommendations. Return as JSON:

{{
  "performance_diagnosis": [
    "Diagnosis 1: Specific performance issue identified",
    "Diagnosis 2: Pattern or bottleneck in current approach",
    "Diagnosis 3: Opportunity area for improvement",
    ...
  ],
  "root_cause_analysis": [
    "Root cause 1: Fundamental issue causing performance limitation",
    "Root cause 2: Data or model architecture problem",
    "Root cause 3: Feature or preprocessing gap",
    ...
  ],
  "improvement_strategies": [
    "Strategy 1: Specific actionable improvement with reasoning",
    "Strategy 2: Technical enhancement with expected impact",
    "Strategy 3: Data or feature engineering improvement",
    "Strategy 4: Model selection or hyperparameter adjustment",
    "Strategy 5: Ensemble or validation strategy change",
    ...
  ],
  "priority_ranking": [
    "Top priority action with highest expected impact",
    "Second priority action with good impact/effort ratio",
    "Third priority action for foundational improvement",
    ...
  ],
  "implementation_plan": [
    "Step 1: First action to take with specific details",
    "Step 2: Second action building on first",
    "Step 3: Third action for comprehensive improvement",
    "Step 4: Validation and testing approach",
    "Step 5: Ensemble or final optimization step",
    ...
  ],
  "risk_mitigation": [
    "Risk 1: Overfitting concern and prevention strategy",
    "Risk 2: Data leakage prevention",
    "Risk 3: Computational or time constraint management"
  ],
  "success_metrics": [
    "Primary metric: Expected CV improvement target",
    "Secondary metric: Validation stability improvement",
    "Quality metric: Feature importance or model interpretability",
    "Efficiency metric: Training time or complexity consideration"
  ],
  "iteration_focus": "Primary focus area for this iteration (e.g., 'Feature Engineering', 'Model Architecture', 'Data Quality')"
}}

Focus on actionable improvements that will measurably increase leaderboard performance in the next iteration.
"""

    @staticmethod
    def feature_engineering_code(dataset_info: Dict, competition_insights: Dict,
                                feature_descriptions: str) -> str:
        """Template for generating feature engineering code"""
        return f"""
You are an expert feature engineer writing Python code for a Kaggle competition.

**DATASET INFO:**
- Rows: {dataset_info.get('total_rows', 'unknown')}
- Columns: {dataset_info.get('total_columns', 'unknown')}
- Problem Type: {dataset_info.get('target_type', 'unknown')}
- Target: {dataset_info.get('target_column', 'unknown')}

**COMPETITION STRATEGY:**
Key strategies: {competition_insights.get('key_strategies', [])}
Model recommendations: {competition_insights.get('model_recommendations', [])}

**FEATURE DESCRIPTIONS:**
{feature_descriptions}

**CODE GENERATION TASK:**
Generate Python code for 15-20 feature engineering transformations. Return as JSON:

{{
  "features": [
    {{
      "name": "feature_name",
      "description": "What this feature captures",
      "code": "# Python code to create the feature\\ndf['new_feature'] = ...",
      "rationale": "Why this feature should help",
      "complexity": "simple/medium/complex"
    }}
  ]
}}

**REQUIREMENTS:**
1. Use pandas and numpy operations
2. Handle missing values appropriately
3. Create features that are likely to be predictive
4. Include both simple and complex transformations
5. Consider feature interactions and domain knowledge
6. Avoid data leakage (no future information)
7. Make code robust and error-resistant

**FEATURE CATEGORIES TO INCLUDE:**
- Numerical transformations (log, sqrt, polynomial)
- Categorical encoding and grouping
- Interaction features
- Statistical aggregations
- Binning and discretization
- Text features (if applicable)
- Date/time features (if applicable)

Generate diverse, high-quality features that will improve model performance.
"""

    @staticmethod
    def model_selection(dataset_characteristics: Dict, competition_insights: Dict,
                       feature_count: int) -> str:
        """Template for model selection recommendations"""
        return f"""
You are a machine learning expert selecting models for a Kaggle competition.

**DATASET CHARACTERISTICS:**
- Problem Type: {dataset_characteristics.get('target_type', 'unknown')}
- Dataset Size: {dataset_characteristics.get('total_rows', 0)} rows
- Feature Count: {feature_count} (after engineering)
- Target Distribution: {dataset_characteristics.get('target_insights', 'unknown')}

**COMPETITION INSIGHTS:**
- Difficulty: {competition_insights.get('difficulty_level', 'unknown')}
- Evaluation Metric: {competition_insights.get('evaluation_focus', 'unknown')}
- Recommended Models: {competition_insights.get('model_recommendations', [])}

**MODEL SELECTION TASK:**
Recommend 5 models for this competition, ranked by expected performance. Return as JSON:

{{
  "models": [
    {{
      "name": "model_name",
      "library": "sklearn/xgboost/lightgbm/catboost",
      "priority": 1-5,
      "expected_performance": "high/medium/low",
      "hyperparameters": {{
        "param1": ["value1", "value2", "value3"],
        "param2": [1, 10, 100]
      }},
      "rationale": "Why this model is good for this problem",
      "training_time": "fast/medium/slow",
      "pros": ["advantage1", "advantage2"],
      "cons": ["limitation1", "limitation2"]
    }}
  ],
  "ensemble_strategy": "stacking/voting/blending recommendation",
  "ensemble_rationale": "Why this ensemble approach",
  "hyperparameter_strategy": "Optuna optimization focus areas"
}}

**AVAILABLE MODELS:**
- XGBoost, LightGBM, CatBoost (gradient boosting)
- RandomForest, ExtraTrees (ensemble)
- Ridge, Lasso, ElasticNet (linear)
- SVM, KNN (instance-based)
- Neural Networks (if appropriate)

Consider dataset size, problem type, evaluation metric, and competition difficulty.
"""

    @staticmethod
    def feedback_analysis(cv_score: float, public_score: Optional[float],
                         model_info: Dict, competition_context: str) -> str:
        """Template for analyzing model performance and suggesting improvements"""
        score_comparison = ""
        if public_score is not None:
            diff = abs(cv_score - public_score)
            if diff > 0.05:  # Significant difference
                score_comparison = f"""
**SCORE ANALYSIS:**
- CV Score: {cv_score:.4f}
- Public Score: {public_score:.4f}
- Difference: {diff:.4f} ({'overfitting' if cv_score > public_score else 'underfitting/distribution shift'})
"""
            else:
                score_comparison = f"CV and public scores are well-aligned ({cv_score:.4f} vs {public_score:.4f})"

        return f"""
You are a machine learning expert analyzing model performance to suggest improvements.

**CURRENT PERFORMANCE:**
{score_comparison}
Model Type: {model_info.get('model_type', 'unknown')}
Features Used: {model_info.get('features_count', 'unknown')}

**COMPETITION CONTEXT:**
{competition_context}

**ANALYSIS TASK:**
Analyze the current model performance and provide improvement recommendations as JSON:

{{
  "performance_diagnosis": "overfitting/underfitting/good_fit/distribution_shift",
  "confidence_level": "high/medium/low confidence in diagnosis",
  "primary_issues": ["issue1", "issue2", "issue3"],
  "improvement_strategies": [
    {{
      "strategy": "strategy_name",
      "priority": "high/medium/low",
      "description": "What to do",
      "expected_impact": "high/medium/low",
      "implementation_effort": "easy/medium/hard"
    }}
  ],
  "feature_recommendations": ["specific feature engineering suggestions"],
  "model_recommendations": ["model architecture or hyperparameter changes"],
  "validation_recommendations": ["cross-validation or evaluation changes"],
  "next_experiments": ["ranked list of next experiments to try"]
}}

**IMPROVEMENT CATEGORIES TO CONSIDER:**
1. Feature Engineering (new features, transformations)
2. Model Selection (different algorithms, ensembles)
3. Hyperparameter Tuning (optimization focus)
4. Data Preprocessing (cleaning, outlier handling)
5. Validation Strategy (CV scheme, early stopping)
6. Regularization (reducing overfitting)

Provide specific, actionable recommendations for the next iteration.
"""

    @staticmethod
    def execution_strategy(competition_info: Dict, dataset_size: int, time_budget: str) -> str:
        """Template for creating execution strategy"""
        return f"""
You are a project manager for autonomous Kaggle competitions, creating an execution plan.

**COMPETITION INFO:**
- Difficulty: {competition_info.get('difficulty_level', 'unknown')}
- Problem Type: {competition_info.get('problem_type', 'unknown')}
- Dataset Size: {dataset_size} rows
- Time Budget: {time_budget}

**STRATEGY TASK:**
Create an execution strategy as JSON:

{{
  "execution_phases": [
    {{
      "phase": "phase_name",
      "duration": "estimated time",
      "tasks": ["task1", "task2"],
      "success_criteria": "how to measure success",
      "risk_level": "low/medium/high"
    }}
  ],
  "feature_engineering_depth": "light/moderate/intensive",
  "model_training_approach": "simple/comprehensive/ensemble",
  "hyperparameter_optimization": "minimal/moderate/extensive",
  "validation_strategy": "simple_cv/stratified/time_based",
  "iteration_plan": "number of improvement cycles",
  "risk_mitigation": ["potential risks and mitigation strategies"],
  "resource_allocation": "how to distribute computational resources"
}}

Balance thorough analysis with time constraints to achieve top 20% ranking.
"""


class LLMUtils:
    """Utility functions for LLM interactions"""

    @staticmethod
    def format_dataset_summary(dataset_info: Dict) -> str:
        """Format dataset info for LLM consumption"""
        summary = f"""
DATASET SUMMARY:
- Shape: {dataset_info.get('total_rows', 0):,} rows × {dataset_info.get('total_columns', 0)} columns
- Target: {dataset_info.get('target_column', 'unknown')} ({dataset_info.get('target_type', 'unknown')})
- Memory: {dataset_info.get('memory_usage_mb', 0):.1f} MB
- Duplicates: {dataset_info.get('duplicates_count', 0):,}

FEATURE TYPES:
"""
        feature_types = dataset_info.get('feature_types', {})
        type_counts = {}
        for ftype in feature_types.values():
            type_counts[ftype] = type_counts.get(ftype, 0) + 1

        for ftype, count in sorted(type_counts.items()):
            summary += f"- {ftype}: {count}\n"

        summary += "\nMISSING VALUES (top issues):\n"
        missing_pct = dataset_info.get('missing_percentages', {})
        sorted_missing = sorted(missing_pct.items(), key=lambda x: x[1], reverse=True)[:5]
        for col, pct in sorted_missing:
            if pct > 0:
                summary += f"- {col}: {pct:.1f}%\n"

        return summary

    @staticmethod
    def format_competition_context(competition_insights: Dict) -> str:
        """Format competition insights for other agents"""
        if not competition_insights:
            return "No competition context available"

        context = f"""
COMPETITION CONTEXT:
- Problem: {competition_insights.get('problem_type', 'unknown')}
- Difficulty: {competition_insights.get('difficulty_level', 'unknown')}
- Evaluation Focus: {competition_insights.get('evaluation_focus', 'unknown')}

KEY STRATEGIES: {', '.join(competition_insights.get('key_strategies', [])[:3])}
RECOMMENDED MODELS: {', '.join(competition_insights.get('model_recommendations', [])[:3])}
COMMON PITFALLS: {', '.join(competition_insights.get('common_pitfalls', [])[:2])}
"""
        return context

    @staticmethod
    def truncate_text(text: str, max_length: int = 2000) -> str:
        """Truncate text for LLM context limits"""
        if len(text) <= max_length:
            return text

        # Try to truncate at sentence boundaries
        truncated = text[:max_length]
        last_period = truncated.rfind('.')
        if last_period > max_length * 0.8:  # If we can keep 80% and end at sentence
            return truncated[:last_period + 1]

        return truncated + "..."

    @staticmethod
    def extract_code_blocks(text: str) -> List[str]:
        """Extract Python code blocks from LLM response"""
        code_blocks = []
        lines = text.split('\n')
        in_code_block = False
        current_block = []

        for line in lines:
            if line.strip().startswith('```python') or line.strip().startswith('```'):
                if in_code_block:
                    # End of code block
                    if current_block:
                        code_blocks.append('\n'.join(current_block))
                    current_block = []
                    in_code_block = False
                else:
                    # Start of code block
                    in_code_block = True
            elif in_code_block:
                current_block.append(line)

        # Handle case where code block doesn't end properly
        if in_code_block and current_block:
            code_blocks.append('\n'.join(current_block))

        return code_blocks

    @staticmethod
    def validate_json_structure(data: Dict, required_keys: List[str]) -> bool:
        """Validate that JSON response has required structure"""
        if not isinstance(data, dict):
            return False

        return all(key in data for key in required_keys)

    @staticmethod
    def safe_get_nested(data: Dict, keys: List[str], default=None) -> Any:
        """Safely get nested dictionary values"""
        current = data
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current


# Predefined prompt combinations for common use cases
PROMPT_COMBINATIONS = {
    "competition_setup": [
        "competition_analysis",
        "dataset_insights",
        "execution_strategy"
    ],
    "feature_development": [
        "dataset_insights",
        "feature_engineering_code"
    ],
    "model_optimization": [
        "model_selection",
        "feedback_analysis"
    ]
}


def get_prompt_template(template_name: str, **kwargs) -> str:
    """Get a formatted prompt template by name"""
    if not hasattr(PromptTemplates, template_name):
        raise ValueError(f"Unknown template: {template_name}")

    template_method = getattr(PromptTemplates, template_name)
    return template_method(**kwargs)