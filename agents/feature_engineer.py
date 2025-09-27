#!/usr/bin/env python3
"""
Feature Engineering Agent - Advanced feature creation with LLM intelligence

This agent generates sophisticated features using LLM insights and code generation.
It analyzes dataset characteristics and competition requirements to create optimal features.
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    from agents.llm_coordinator import LLMCoordinator
    from utils.llm_utils import PromptTemplates, LLMUtils
    from agents.data_scout import DataScout
    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False


@dataclass
class FeatureEngineering:
    """Structure to hold generated features and metadata"""
    competition_name: str
    features_created: List[str]
    feature_types: Dict[str, str]  # feature_name -> type (numerical, categorical, etc.)
    creation_methods: Dict[str, str]  # feature_name -> method description
    feature_importance_scores: Dict[str, float]
    code_generated: List[str]  # List of code blocks generated
    validation_scores: Dict[str, float]  # Cross-validation scores with new features
    timestamp: str
    total_features: int


@dataclass
class LLMFeatureInsights:
    """Structure for LLM-generated feature engineering insights"""
    competition_name: str
    recommended_features: List[str]
    feature_engineering_strategies: List[str]
    interaction_opportunities: List[str]
    transformation_suggestions: List[str]
    domain_specific_features: List[str]
    code_templates: List[str]
    risk_assessment: List[str]
    confidence_score: float
    analysis_timestamp: str


class FeatureEngineer:
    """
    Feature Engineering Agent with LLM-powered code generation
    """

    def __init__(self, competition_path: Path, enable_llm: bool = True):
        self.competition_path = Path(competition_path)
        self.competition_name = self.competition_path.name
        self.enable_llm = enable_llm

        # Initialize output directory
        self.output_dir = self.competition_path / "feature_engineering"
        self.output_dir.mkdir(exist_ok=True)

        # Initialize LLM coordinator if available and enabled
        self.llm_coordinator = None
        if self.enable_llm and DEPENDENCIES_AVAILABLE:
            try:
                self.llm_coordinator = LLMCoordinator(log_dir=self.output_dir / "llm_logs")
                print("LLM coordinator initialized for feature engineering")
            except Exception as e:
                print(f"Warning: Could not initialize LLM coordinator: {e}")
                self.enable_llm = False

        # Initialize data scout for dataset insights
        self.data_scout = None
        self.dataset_info = None
        self.competition_insights = None

        # Storage for generated features
        self.feature_engineering = None
        self.llm_insights = None
        self.original_features = []
        self.engineered_features = []

    def load_data_and_insights(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data and existing insights from data scout and competition reader"""
        print("Loading data and existing insights...")

        # Load raw data
        train_path = self.competition_path / "train.csv"
        test_path = self.competition_path / "test.csv"

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path) if test_path.exists() else pd.DataFrame()

        print(f"Loaded train: {train_df.shape}, test: {test_df.shape}")

        # Load data scout insights if available
        scout_outputs = self.competition_path / "data_scout_outputs"
        if scout_outputs.exists():
            dataset_info_path = scout_outputs / "dataset_info.json"
            if dataset_info_path.exists():
                with open(dataset_info_path, 'r') as f:
                    self.dataset_info = json.load(f)
                print("Loaded data scout insights")

        # Load competition insights if available
        comp_insights_path = self.competition_path / "competition_understanding.json"
        if comp_insights_path.exists():
            with open(comp_insights_path, 'r') as f:
                self.competition_insights = json.load(f)
            print("Loaded competition insights")

        return train_df, test_df

    def analyze_features_with_llm(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Optional[LLMFeatureInsights]:
        """Use LLM to analyze dataset and suggest advanced feature engineering strategies"""
        if not self.enable_llm or not self.llm_coordinator:
            return None

        print("Analyzing features with LLM...")

        # Prepare dataset summary for LLM
        dataset_summary = self._prepare_dataset_summary(train_df, test_df)
        competition_context = self._prepare_competition_context()

        # Create prompt for feature engineering analysis
        prompt = PromptTemplates.feature_engineering_analysis(
            dataset_summary=dataset_summary,
            competition_context=competition_context,
            existing_features=list(train_df.columns)
        )

        # Get structured insights from LLM
        llm_response = self.llm_coordinator.structured_output(
            prompt,
            agent="feature_engineer",
            model_type="primary",
            temperature=0.4,  # Moderate creativity for feature ideas
            max_tokens=3000
        )

        if not llm_response:
            print("Failed to get LLM feature analysis")
            return None

        # Validate response structure
        required_keys = [
            "recommended_features", "feature_engineering_strategies",
            "interaction_opportunities", "transformation_suggestions"
        ]

        if not LLMUtils.validate_json_structure(llm_response, required_keys):
            print("LLM response missing required fields for feature analysis")
            return None

        # Calculate confidence score
        confidence = self._calculate_feature_confidence(llm_response)

        # Create structured insights
        insights = LLMFeatureInsights(
            competition_name=self.competition_name,
            recommended_features=llm_response.get("recommended_features", [])[:10],
            feature_engineering_strategies=llm_response.get("feature_engineering_strategies", [])[:5],
            interaction_opportunities=llm_response.get("interaction_opportunities", [])[:5],
            transformation_suggestions=llm_response.get("transformation_suggestions", [])[:5],
            domain_specific_features=llm_response.get("domain_specific_features", [])[:5],
            code_templates=llm_response.get("code_templates", [])[:3],
            risk_assessment=llm_response.get("risk_assessment", [])[:3],
            confidence_score=confidence,
            analysis_timestamp=datetime.now().isoformat()
        )

        return insights

    def generate_features_with_llm(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> List[str]:
        """Generate feature engineering code using LLM"""
        if not self.enable_llm or not self.llm_coordinator:
            return []

        print("Generating feature engineering code with LLM...")

        # Prepare context for code generation
        dataset_info = {
            "total_rows": len(train_df),
            "total_columns": len(train_df.columns),
            "target_column": self._identify_target_column(train_df),
            "target_type": "classification" if self._is_classification_problem() else "regression",
            "feature_types": self._analyze_feature_types(train_df)
        }

        feature_descriptions = self._create_feature_descriptions(train_df)

        # Create prompt for code generation
        prompt = PromptTemplates.feature_engineering_code(
            dataset_info=dataset_info,
            competition_insights=self.competition_insights or {},
            feature_descriptions=feature_descriptions
        )

        # Get code from LLM
        llm_response = self.llm_coordinator.structured_output(
            prompt,
            agent="feature_engineer",
            model_type="code",  # Use code-specialized model
            temperature=0.2,  # Lower temperature for more reliable code
            max_tokens=4000
        )

        if not llm_response:
            print("Failed to generate feature engineering code")
            return []

        # Extract code blocks from response
        code_blocks = []
        if "feature_engineering_code" in llm_response:
            code_blocks.extend(llm_response["feature_engineering_code"])
        if "transformation_code" in llm_response:
            code_blocks.extend(llm_response["transformation_code"])
        if "interaction_code" in llm_response:
            code_blocks.extend(llm_response["interaction_code"])

        print(f"Generated {len(code_blocks)} code blocks for feature engineering")
        return code_blocks

    def execute_feature_engineering(self, train_df: pd.DataFrame, test_df: pd.DataFrame, code_blocks: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Safely execute generated feature engineering code"""
        print("Executing feature engineering code...")

        # Store original features
        self.original_features = list(train_df.columns)

        # Create copies for feature engineering
        train_engineered = train_df.copy()
        test_engineered = test_df.copy() if not test_df.empty else pd.DataFrame()

        executed_code = []
        features_created = []
        creation_methods = {}

        for i, code_block in enumerate(code_blocks):
            try:
                print(f"Executing code block {i+1}/{len(code_blocks)}...")

                # Create safe execution environment
                safe_globals = {
                    'pd': pd, 'np': np,
                    'train_df': train_engineered,
                    'test_df': test_engineered,
                    'StandardScaler': StandardScaler,
                    'LabelEncoder': LabelEncoder
                }

                # Extract and clean code
                clean_code = LLMUtils.extract_code_blocks(code_block)
                if clean_code:
                    code_to_execute = clean_code[0]
                else:
                    code_to_execute = code_block

                # Store features before execution
                features_before = set(train_engineered.columns)

                # Execute code
                exec(code_to_execute, safe_globals)

                # Get updated dataframes
                train_engineered = safe_globals.get('train_df', train_engineered)
                test_engineered = safe_globals.get('test_df', test_engineered)

                # Identify new features
                features_after = set(train_engineered.columns)
                new_features = list(features_after - features_before)

                if new_features:
                    features_created.extend(new_features)
                    for feature in new_features:
                        creation_methods[feature] = f"LLM generated code block {i+1}"
                    print(f"Created features: {', '.join(new_features)}")

                executed_code.append(code_to_execute)

            except Exception as e:
                print(f"Error executing code block {i+1}: {e}")
                # Continue with other code blocks
                continue

        # Store engineering results
        self.engineered_features = features_created

        print(f"Feature engineering complete. Created {len(features_created)} new features")

        return train_engineered, test_engineered

    def apply_basic_feature_engineering(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply basic feature engineering techniques as fallback"""
        print("Applying basic feature engineering...")

        train_engineered = train_df.copy()
        test_engineered = test_df.copy() if not test_df.empty else pd.DataFrame()

        features_created = []
        creation_methods = {}

        # Numeric features (excluding target and ID columns)
        target_col = self._identify_target_column(train_df)
        numeric_cols = train_df.select_dtypes(include=[np.number]).columns
        # Only process columns that exist in both train and test (or when test is empty)
        if not test_engineered.empty:
            numeric_cols = [col for col in numeric_cols if col in test_df.columns]

        for col in numeric_cols:
            if col not in ['id', 'Id', 'ID', target_col]:  # Skip ID and target columns
                # Log transformation for skewed features
                if train_df[col].min() > 0 and train_df[col].skew() > 1:
                    new_col = f"{col}_log"
                    train_engineered[new_col] = np.log1p(train_df[col])
                    if not test_engineered.empty:
                        test_engineered[new_col] = np.log1p(test_df[col])
                    features_created.append(new_col)
                    creation_methods[new_col] = "Log transformation"

                # Square features for potential non-linear relationships
                if train_df[col].std() > 0:
                    new_col = f"{col}_squared"
                    train_engineered[new_col] = train_df[col] ** 2
                    if not test_engineered.empty:
                        test_engineered[new_col] = test_df[col] ** 2
                    features_created.append(new_col)
                    creation_methods[new_col] = "Square transformation"

        # Pairwise interactions for top numeric features (limit to avoid explosion)
        if len(numeric_cols) > 1:
            important_numeric = numeric_cols[:5]  # Top 5 numeric features
            for i, col1 in enumerate(important_numeric):
                for col2 in important_numeric[i+1:]:
                    if col1 != col2:
                        new_col = f"{col1}_x_{col2}"
                        train_engineered[new_col] = train_df[col1] * train_df[col2]
                        if not test_engineered.empty:
                            test_engineered[new_col] = test_df[col1] * test_df[col2]
                        features_created.append(new_col)
                        creation_methods[new_col] = "Feature interaction"

        # Categorical features encoding
        categorical_cols = train_df.select_dtypes(include=['object']).columns
        # Only process columns that exist in both train and test (or when test is empty)
        if not test_engineered.empty:
            categorical_cols = [col for col in categorical_cols if col in test_df.columns]

        for col in categorical_cols:
            if col not in ['id', 'Id', 'ID']:
                # Frequency encoding
                freq_map = train_df[col].value_counts().to_dict()
                new_col = f"{col}_frequency"
                train_engineered[new_col] = train_df[col].map(freq_map)
                if not test_engineered.empty:
                    test_engineered[new_col] = test_df[col].map(freq_map).fillna(0)
                features_created.append(new_col)
                creation_methods[new_col] = "Frequency encoding"

        self.engineered_features = features_created

        print(f"Basic feature engineering complete. Created {len(features_created)} features")

        return train_engineered, test_engineered

    def save_results(self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_engineering: FeatureEngineering):
        """Save feature engineering results"""
        print("Saving feature engineering results...")

        # Save engineered datasets
        train_path = self.output_dir / "train_engineered.csv"
        train_df.to_csv(train_path, index=False)
        print(f"Saved engineered training data: {train_path}")

        if not test_df.empty:
            test_path = self.output_dir / "test_engineered.csv"
            test_df.to_csv(test_path, index=False)
            print(f"Saved engineered test data: {test_path}")

        # Save feature engineering metadata
        engineering_path = self.output_dir / "feature_engineering.json"
        with open(engineering_path, 'w') as f:
            json.dump(asdict(feature_engineering), f, indent=2)
        print(f"Saved feature engineering metadata: {engineering_path}")

        # Save LLM insights if available
        if self.llm_insights:
            insights_path = self.output_dir / "llm_feature_insights.json"
            with open(insights_path, 'w') as f:
                json.dump(asdict(self.llm_insights), f, indent=2)
            print(f"Saved LLM feature insights: {insights_path}")

        # Create feature mapping
        feature_mapping = {
            "original_features": self.original_features,
            "engineered_features": self.engineered_features,
            "total_features_before": len(self.original_features),
            "total_features_after": len(train_df.columns),
            "features_added": len(self.engineered_features)
        }

        mapping_path = self.output_dir / "feature_mapping.json"
        with open(mapping_path, 'w') as f:
            json.dump(feature_mapping, f, indent=2)
        print(f"Saved feature mapping: {mapping_path}")

    def run_feature_engineering(self) -> Tuple[pd.DataFrame, pd.DataFrame, FeatureEngineering]:
        """Run complete feature engineering pipeline"""
        print(f"Feature Engineering Agent: {self.competition_name}")
        print("=" * 60)

        # Load data and insights
        train_df, test_df = self.load_data_and_insights()

        # Analyze features with LLM if enabled
        if self.enable_llm and self.llm_coordinator:
            print("\nStep 1: Analyzing features with LLM...")
            self.llm_insights = self.analyze_features_with_llm(train_df, test_df)

            # Generate feature engineering code
            print("\nStep 2: Generating feature engineering code...")
            code_blocks = self.generate_features_with_llm(train_df, test_df)

            if code_blocks:
                # Execute generated code
                print("\nStep 3: Executing feature engineering code...")
                train_engineered, test_engineered = self.execute_feature_engineering(train_df, test_df, code_blocks)
            else:
                print("No code generated, falling back to basic feature engineering")
                train_engineered, test_engineered = self.apply_basic_feature_engineering(train_df, test_df)
        else:
            print("\nStep 1: Applying basic feature engineering...")
            train_engineered, test_engineered = self.apply_basic_feature_engineering(train_df, test_df)

        # Create feature engineering summary
        feature_engineering = FeatureEngineering(
            competition_name=self.competition_name,
            features_created=self.engineered_features,
            feature_types=self._analyze_feature_types(train_engineered),
            creation_methods={f: "Generated" for f in self.engineered_features},
            feature_importance_scores={},  # Will be filled by model selection agent
            code_generated=[],  # Code blocks would be stored here
            validation_scores={},  # Cross-validation scores
            timestamp=datetime.now().isoformat(),
            total_features=len(train_engineered.columns)
        )

        # Save results
        self.save_results(train_engineered, test_engineered, feature_engineering)

        print(f"\nFeature Engineering Complete!")
        print(f"Original features: {len(self.original_features)}")
        print(f"New features: {len(self.engineered_features)}")
        print(f"Total features: {len(train_engineered.columns)}")

        return train_engineered, test_engineered, feature_engineering

    # Helper methods
    def _prepare_dataset_summary(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> str:
        """Prepare dataset summary for LLM"""
        summary_parts = [
            f"Training data: {train_df.shape[0]} rows, {train_df.shape[1]} columns",
            f"Test data: {test_df.shape[0]} rows, {test_df.shape[1]} columns" if not test_df.empty else "No test data",
            f"Features: {', '.join(train_df.columns[:10])}" + ("..." if len(train_df.columns) > 10 else "")
        ]

        if self.dataset_info:
            summary_parts.append(f"Missing values: {self.dataset_info.get('missing_percentages', {})}")
            summary_parts.append(f"Data types: {self.dataset_info.get('feature_types', {})}")

        return " | ".join(summary_parts)

    def _prepare_competition_context(self) -> str:
        """Prepare competition context for LLM"""
        if not self.competition_insights:
            return f"Competition: {self.competition_name}"

        context_parts = [
            f"Competition: {self.competition_insights.get('competition_name', self.competition_name)}",
            f"Problem type: {self.competition_insights.get('problem_type', 'unknown')}",
            f"Key strategies: {', '.join(self.competition_insights.get('key_strategies', [])[:3])}"
        ]

        return " | ".join(context_parts)

    def _calculate_feature_confidence(self, llm_response: Dict) -> float:
        """Calculate confidence score for feature engineering recommendations"""
        confidence = 0.0

        # Check completeness of recommendations
        if len(llm_response.get("recommended_features", [])) >= 5:
            confidence += 0.3

        # Check strategy variety
        if len(llm_response.get("feature_engineering_strategies", [])) >= 3:
            confidence += 0.3

        # Check for code templates
        if llm_response.get("code_templates"):
            confidence += 0.2

        # Check for domain insights
        if llm_response.get("domain_specific_features"):
            confidence += 0.2

        return min(confidence, 1.0)

    def _identify_target_column(self, train_df: pd.DataFrame) -> str:
        """Identify the target column in training data"""
        common_targets = ['target', 'label', 'y', 'class', 'survived', 'price', 'saleprice']

        for col in train_df.columns:
            if col.lower() in common_targets:
                return col

        # Return last column as default
        return train_df.columns[-1]

    def _is_classification_problem(self) -> bool:
        """Determine if this is a classification problem"""
        if self.competition_insights:
            problem_type = self.competition_insights.get('problem_type', '').lower()
            return 'classification' in problem_type

        # Default heuristic based on competition name
        name_lower = self.competition_name.lower()
        return any(word in name_lower for word in ['classification', 'class', 'survived', 'category'])

    def _analyze_feature_types(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analyze feature types in dataframe"""
        feature_types = {}

        for col in df.columns:
            if df[col].dtype in ['int64', 'float64']:
                feature_types[col] = 'numerical'
            elif df[col].dtype == 'object':
                feature_types[col] = 'categorical'
            elif df[col].dtype == 'datetime64[ns]':
                feature_types[col] = 'datetime'
            else:
                feature_types[col] = 'other'

        return feature_types

    def _create_feature_descriptions(self, train_df: pd.DataFrame) -> str:
        """Create feature descriptions for LLM"""
        descriptions = []

        for col in train_df.columns[:20]:  # Limit to first 20 columns
            col_type = 'numerical' if train_df[col].dtype in ['int64', 'float64'] else 'categorical'
            unique_count = train_df[col].nunique()
            descriptions.append(f"{col} ({col_type}, {unique_count} unique values)")

        return ", ".join(descriptions)


def main():
    """Main entry point for Feature Engineering Agent"""
    parser = argparse.ArgumentParser(description="Feature Engineering Agent with LLM")
    parser.add_argument("competition_path", type=Path,
                       help="Path to competition directory")
    parser.add_argument("--no-llm", action="store_true",
                       help="Disable LLM and use basic feature engineering only")

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
        # Initialize and run feature engineering
        engineer = FeatureEngineer(args.competition_path, enable_llm=not args.no_llm)
        train_engineered, test_engineered, feature_engineering = engineer.run_feature_engineering()

        print(f"\nFeature Engineering Success!")
        print(f"Features created: {len(feature_engineering.features_created)}")
        print(f"Total features: {feature_engineering.total_features}")
        print(f"Results saved to: {engineer.output_dir}")
        return 0

    except Exception as e:
        print(f"ERROR: Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())