#!/usr/bin/env python3
"""
Data Scout Agent - Initial EDA and Data Cleaning for Kaggle Competitions

Primary Functions:
1. Load downloaded competition data (train/test CSVs)
2. Perform initial EDA to understand the dataset
3. Basic data cleaning and preprocessing
4. Output cleaned training set and analysis report
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd
import numpy as np

# LLM integration
try:
    from agents.llm_coordinator import LLMCoordinator
    from utils.llm_utils import PromptTemplates, LLMUtils
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


@dataclass
class DatasetInfo:
    """Structure to hold dataset information"""
    competition_name: str
    total_rows: int
    total_columns: int
    train_rows: int
    test_rows: int
    feature_types: Dict[str, str]
    missing_values: Dict[str, int]
    missing_percentages: Dict[str, float]
    target_column: Optional[str]
    target_type: Optional[str]
    duplicates_count: int
    memory_usage_mb: float
    analysis_timestamp: str


@dataclass
class DataQualityReport:
    """Structure to hold data quality analysis"""
    competition_name: str
    outliers_detected: Dict[str, int]
    data_types_converted: List[str]
    rows_removed: int
    columns_removed: List[str]
    cleaning_actions: List[str]
    recommendations: List[str]


@dataclass
class LLMDataInsights:
    """Structure to hold LLM-generated data insights"""
    competition_name: str
    data_quality_score: int
    critical_issues: List[str]
    preprocessing_priorities: List[str]
    feature_engineering_ideas: List[str]
    potential_challenges: List[str]
    target_insights: str
    feature_interactions: List[str]
    missing_data_strategy: str
    outlier_strategy: str
    validation_strategy: str
    confidence_score: float
    analysis_timestamp: str


class DataScout:
    """
    Data Scout Agent for initial EDA and data cleaning of Kaggle competition datasets
    """

    def __init__(self, competition_path: Path, output_dir: Optional[Path] = None,
                 enable_llm: bool = True):
        self.competition_path = Path(competition_path)
        self.competition_name = self.competition_path.name
        self.output_dir = output_dir or self.competition_path / "scout_output"
        self.output_dir.mkdir(exist_ok=True)

        # Data containers
        self.train_df = None
        self.test_df = None
        self.dataset_info = None
        self.quality_report = None
        self.llm_insights = None

        # Configuration
        self.outlier_threshold = 3.0  # Standard deviations for outlier detection
        self.missing_threshold = 0.8  # Drop columns with >80% missing values

        # LLM integration
        self.enable_llm = enable_llm and LLM_AVAILABLE
        self.llm_coordinator = None

        if self.enable_llm:
            try:
                self.llm_coordinator = LLMCoordinator(log_dir=self.competition_path / "llm_logs")
                print("LLM-enhanced Data Scout initialized")
            except Exception as e:
                print(f"Warning: Could not initialize LLM coordinator: {e}")
                self.enable_llm = False

    def load_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load train and test datasets from competition directory"""
        train_path = self.competition_path / "train.csv"
        test_path = self.competition_path / "test.csv"

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found: {train_path}")

        print(f"Loading data for competition: {self.competition_name}")

        # Load training data
        self.train_df = pd.read_csv(train_path)
        print(f"Loaded training data: {self.train_df.shape}")

        # Load test data if available
        if test_path.exists():
            self.test_df = pd.read_csv(test_path)
            print(f"Loaded test data: {self.test_df.shape}")
        else:
            print("! No test data found")
            self.test_df = None

        return self.train_df, self.test_df

    def analyze_features(self) -> Dict[str, str]:
        """Analyze feature types and characteristics"""
        feature_types = {}

        for col in self.train_df.columns:
            dtype = str(self.train_df[col].dtype)

            if dtype in ['int64', 'int32', 'float64', 'float32']:
                unique_count = self.train_df[col].nunique()
                total_count = len(self.train_df)

                if unique_count == 2:
                    feature_types[col] = "binary"
                elif unique_count < 10 and unique_count < total_count * 0.05:
                    feature_types[col] = "categorical_numeric"
                else:
                    feature_types[col] = "numerical"
            elif dtype == 'object':
                unique_count = self.train_df[col].nunique()
                total_count = len(self.train_df)

                if unique_count < total_count * 0.5:
                    feature_types[col] = "categorical"
                else:
                    feature_types[col] = "text"
            elif dtype.startswith('datetime'):
                feature_types[col] = "datetime"
            else:
                feature_types[col] = "other"

        return feature_types

    def detect_target_column(self) -> Optional[str]:
        """Attempt to detect the target column"""
        common_targets = ['target', 'label', 'y', 'class', 'outcome', 'prediction']

        # Check for common target column names
        for col in self.train_df.columns:
            if col.lower() in common_targets:
                return col

        # Check if target column is not in test set
        if self.test_df is not None:
            train_cols = set(self.train_df.columns)
            test_cols = set(self.test_df.columns)
            diff_cols = train_cols - test_cols

            if len(diff_cols) == 1:
                return list(diff_cols)[0]

        # Check for columns with limited unique values that might be targets
        for col in self.train_df.columns:
            unique_ratio = self.train_df[col].nunique() / len(self.train_df)
            if 0.001 < unique_ratio < 0.1 and self.train_df[col].dtype in ['int64', 'float64']:
                return col

        return None

    def perform_eda(self) -> DatasetInfo:
        """Perform comprehensive exploratory data analysis"""
        print("\nPerforming EDA...")

        feature_types = self.analyze_features()
        target_col = self.detect_target_column()

        # Missing values analysis
        missing_counts = self.train_df.isnull().sum().to_dict()
        missing_percentages = (self.train_df.isnull().sum() / len(self.train_df) * 100).to_dict()

        # Duplicate analysis
        duplicates = self.train_df.duplicated().sum()

        # Memory usage
        memory_usage = self.train_df.memory_usage(deep=True).sum() / (1024 * 1024)

        # Target analysis
        target_type = None
        if target_col:
            if feature_types.get(target_col) in ['numerical']:
                target_type = "regression"
            else:
                target_type = "classification"

        self.dataset_info = DatasetInfo(
            competition_name=self.competition_name,
            total_rows=len(self.train_df),
            total_columns=len(self.train_df.columns),
            train_rows=len(self.train_df),
            test_rows=len(self.test_df) if self.test_df is not None else 0,
            feature_types=feature_types,
            missing_values=missing_counts,
            missing_percentages=missing_percentages,
            target_column=target_col,
            target_type=target_type,
            duplicates_count=duplicates,
            memory_usage_mb=round(memory_usage, 2),
            analysis_timestamp=datetime.now().isoformat()
        )

        self._print_eda_summary()
        return self.dataset_info

    def _print_eda_summary(self):
        """Print EDA summary to console"""
        info = self.dataset_info
        print(f"\nDataset Analysis Summary for {info.competition_name}")
        print("=" * 50)
        print(f"Dataset Size: {info.train_rows:,} rows × {info.total_columns} columns")
        print(f"Memory Usage: {info.memory_usage_mb} MB")
        print(f"Duplicates: {info.duplicates_count:,}")
        print(f"Target Column: {info.target_column or 'Not detected'}")
        print(f"Problem Type: {info.target_type or 'Unknown'}")

        print(f"\nFeature Types:")
        type_counts = {}
        for ftype in info.feature_types.values():
            type_counts[ftype] = type_counts.get(ftype, 0) + 1
        for ftype, count in sorted(type_counts.items()):
            print(f"  {ftype}: {count}")

        print(f"\nMissing Values (top 10):")
        missing_sorted = sorted(info.missing_percentages.items(),
                              key=lambda x: x[1], reverse=True)[:10]
        for col, pct in missing_sorted:
            if pct > 0:
                print(f"  {col}: {pct:.1f}% ({info.missing_values[col]:,} rows)")

    def detect_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        """Detect outliers using IQR method for numerical columns"""
        outliers = {}
        numerical_cols = [col for col, dtype in self.dataset_info.feature_types.items()
                         if dtype == 'numerical' and col in df.columns]

        for col in numerical_cols:
            if df[col].dtype in ['int64', 'float64'] and not df[col].isnull().all():
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outliers[col] = outlier_mask.sum()

        return outliers

    def basic_cleaning(self) -> pd.DataFrame:
        """Perform basic data cleaning operations"""
        print("\nPerforming basic data cleaning...")

        cleaned_df = self.train_df.copy()
        cleaning_actions = []
        outliers_detected = self.detect_outliers(cleaned_df)
        converted_types = []
        removed_columns = []
        initial_rows = len(cleaned_df)

        # 1. Remove columns with excessive missing values
        high_missing_cols = [col for col, pct in self.dataset_info.missing_percentages.items()
                           if pct > self.missing_threshold * 100]
        if high_missing_cols:
            cleaned_df = cleaned_df.drop(columns=high_missing_cols)
            removed_columns.extend(high_missing_cols)
            cleaning_actions.append(f"Removed {len(high_missing_cols)} columns with >{self.missing_threshold*100}% missing values")

        # 2. Handle duplicates
        if self.dataset_info.duplicates_count > 0:
            cleaned_df = cleaned_df.drop_duplicates()
            cleaning_actions.append(f"Removed {self.dataset_info.duplicates_count} duplicate rows")

        # 3. Basic type conversions
        for col in cleaned_df.columns:
            if col in self.dataset_info.feature_types:
                feature_type = self.dataset_info.feature_types[col]

                # Convert categorical numerics to category
                if feature_type == "categorical_numeric" and cleaned_df[col].dtype in ['int64', 'float64']:
                    cleaned_df[col] = cleaned_df[col].astype('category')
                    converted_types.append(f"{col}: {feature_type}")

                # Convert obvious categories to category type
                elif feature_type == "categorical" and cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].astype('category')
                    converted_types.append(f"{col}: {feature_type}")

        # 4. Fill obvious missing values
        for col in cleaned_df.columns:
            if cleaned_df[col].isnull().sum() > 0:
                feature_type = self.dataset_info.feature_types.get(col, "unknown")

                if feature_type == "numerical":
                    # Fill with median for numerical
                    median_val = cleaned_df[col].median()
                    if pd.notna(median_val):
                        cleaned_df[col] = cleaned_df[col].fillna(median_val)
                        cleaning_actions.append(f"Filled missing {col} with median ({median_val})")

                elif feature_type in ["categorical", "categorical_numeric"]:
                    # Fill with mode for categorical
                    mode_val = cleaned_df[col].mode()
                    if len(mode_val) > 0:
                        cleaned_df[col] = cleaned_df[col].fillna(mode_val[0])
                        cleaning_actions.append(f"Filled missing {col} with mode ({mode_val[0]})")

        final_rows = len(cleaned_df)
        rows_removed = initial_rows - final_rows

        # Create quality report
        recommendations = self._generate_recommendations()

        self.quality_report = DataQualityReport(
            competition_name=self.competition_name,
            outliers_detected=outliers_detected,
            data_types_converted=converted_types,
            rows_removed=rows_removed,
            columns_removed=removed_columns,
            cleaning_actions=cleaning_actions,
            recommendations=recommendations
        )

        self._print_cleaning_summary()
        return cleaned_df

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations for further analysis"""
        recommendations = []

        # High missing value columns
        high_missing = [col for col, pct in self.dataset_info.missing_percentages.items()
                       if 20 <= pct <= 80]
        if high_missing:
            recommendations.append(f"Consider advanced imputation for: {', '.join(high_missing[:3])}")

        # High cardinality categorical features
        high_card_cats = [col for col, ftype in self.dataset_info.feature_types.items()
                         if ftype == "categorical" and self.train_df[col].nunique() > 50]
        if high_card_cats:
            recommendations.append(f"Consider encoding/grouping high cardinality features: {', '.join(high_card_cats[:3])}")

        # Text features
        text_features = [col for col, ftype in self.dataset_info.feature_types.items()
                        if ftype == "text"]
        if text_features:
            recommendations.append(f"Apply NLP preprocessing to text features: {', '.join(text_features[:3])}")

        # Datetime features
        datetime_features = [col for col, ftype in self.dataset_info.feature_types.items()
                           if ftype == "datetime"]
        if datetime_features:
            recommendations.append(f"Extract datetime features from: {', '.join(datetime_features)}")

        # Skewed numerical features
        numerical_cols = [col for col, ftype in self.dataset_info.feature_types.items()
                         if ftype == "numerical"]
        if numerical_cols:
            recommendations.append("Check for skewed distributions and consider transformations")

        return recommendations

    def analyze_with_llm(self) -> Optional[LLMDataInsights]:
        """Use LLM to analyze dataset and generate advanced insights"""
        if not self.enable_llm or not self.llm_coordinator:
            return None

        if not self.dataset_info:
            print("Warning: No dataset info available for LLM analysis")
            return None

        print("\nGenerating LLM insights...")

        try:
            # Load competition context if available
            competition_context = "No competition context available"
            competition_insights_path = self.competition_path / "competition_understanding.json"

            if competition_insights_path.exists():
                try:
                    with open(competition_insights_path, 'r') as f:
                        comp_data = json.load(f)
                    competition_context = LLMUtils.format_competition_context(comp_data)
                except Exception:
                    pass

            # Create EDA summary for LLM
            eda_summary = LLMUtils.format_dataset_summary(asdict(self.dataset_info))

            # Get insights from LLM
            prompt = PromptTemplates.dataset_insights(
                eda_summary=eda_summary,
                competition_context=competition_context
            )

            llm_response = self.llm_coordinator.structured_output(
                prompt,
                agent="data_scout",
                model_type="primary",
                temperature=0.4,
                max_tokens=2500
            )

            if not llm_response:
                print("Failed to get LLM insights")
                return None

            # Validate response structure
            required_keys = [
                "data_quality_score", "critical_issues", "preprocessing_priorities",
                "feature_engineering_ideas", "potential_challenges"
            ]

            if not LLMUtils.validate_json_structure(llm_response, required_keys):
                print("LLM response missing required fields")
                return None

            # Calculate confidence score
            confidence = self._calculate_llm_confidence(llm_response)

            # Create structured insights
            insights = LLMDataInsights(
                competition_name=self.competition_name,
                data_quality_score=min(10, max(1, int(llm_response.get("data_quality_score", 5)))),
                critical_issues=llm_response.get("critical_issues", [])[:3],
                preprocessing_priorities=llm_response.get("preprocessing_priorities", [])[:5],
                feature_engineering_ideas=llm_response.get("feature_engineering_ideas", [])[:8],
                potential_challenges=llm_response.get("potential_challenges", [])[:3],
                target_insights=str(llm_response.get("target_insights", ""))[:500],
                feature_interactions=llm_response.get("feature_interactions", [])[:5],
                missing_data_strategy=str(llm_response.get("missing_data_strategy", ""))[:300],
                outlier_strategy=str(llm_response.get("outlier_strategy", ""))[:300],
                validation_strategy=str(llm_response.get("validation_strategy", ""))[:300],
                confidence_score=confidence,
                analysis_timestamp=datetime.now().isoformat()
            )

            self.llm_insights = insights
            return insights

        except Exception as e:
            print(f"Error in LLM analysis: {e}")
            return None

    def _calculate_llm_confidence(self, llm_response: Dict) -> float:
        """Calculate confidence score for LLM insights"""
        confidence = 0.0

        # Check completeness of response
        if llm_response.get("data_quality_score"):
            confidence += 0.2

        if len(llm_response.get("critical_issues", [])) >= 2:
            confidence += 0.2

        if len(llm_response.get("preprocessing_priorities", [])) >= 3:
            confidence += 0.2

        if len(llm_response.get("feature_engineering_ideas", [])) >= 5:
            confidence += 0.2

        if llm_response.get("validation_strategy"):
            confidence += 0.2

        return min(1.0, confidence)

    def _print_llm_insights(self):
        """Print LLM insights summary"""
        if not self.llm_insights:
            return

        insights = self.llm_insights
        print(f"\nLLM Data Insights (Confidence: {insights.confidence_score:.1%})")
        print("=" * 40)
        print(f"Data Quality Score: {insights.data_quality_score}/10")

        if insights.critical_issues:
            print(f"\nCritical Issues:")
            for i, issue in enumerate(insights.critical_issues, 1):
                print(f"  {i}. {issue}")

        if insights.preprocessing_priorities:
            print(f"\nPreprocessing Priorities:")
            for i, priority in enumerate(insights.preprocessing_priorities, 1):
                print(f"  {i}. {priority}")

        if insights.feature_engineering_ideas:
            print(f"\nFeature Engineering Ideas (Top 5):")
            for i, idea in enumerate(insights.feature_engineering_ideas[:5], 1):
                print(f"  {i}. {idea}")

        if insights.target_insights:
            print(f"\nTarget Analysis:")
            print(f"  {insights.target_insights}")

        if insights.validation_strategy:
            print(f"\nRecommended Validation:")
            print(f"  {insights.validation_strategy}")

    def _print_cleaning_summary(self):
        """Print cleaning summary"""
        report = self.quality_report
        print(f"\nData Cleaning Summary")
        print("=" * 30)
        print(f"Rows removed: {report.rows_removed}")
        print(f"Columns removed: {len(report.columns_removed)}")
        print(f"Data types converted: {len(report.data_types_converted)}")

        if report.cleaning_actions:
            print(f"\nCleaning Actions:")
            for action in report.cleaning_actions:
                print(f"  - {action}")

        if report.outliers_detected:
            print(f"\nOutliers Detected:")
            for col, count in sorted(report.outliers_detected.items(),
                                   key=lambda x: x[1], reverse=True)[:5]:
                if count > 0:
                    print(f"  - {col}: {count} outliers")

        if report.recommendations:
            print(f"\nRecommendations:")
            for rec in report.recommendations:
                print(f"  - {rec}")

    def save_outputs(self, cleaned_df: pd.DataFrame):
        """Save all outputs to files"""
        print(f"\nSaving outputs to {self.output_dir}")

        # Save cleaned training data
        cleaned_path = self.output_dir / "train_cleaned.csv"
        cleaned_df.to_csv(cleaned_path, index=False)
        print(f"Saved cleaned training data: {cleaned_path}")

        # Save dataset info as JSON
        info_path = self.output_dir / "dataset_info.json"
        info_dict = asdict(self.dataset_info)
        # Convert numpy types to native Python types for JSON serialization
        for key, value in info_dict.items():
            if isinstance(value, dict):
                info_dict[key] = {k: int(v) if isinstance(v, np.integer) else
                                 float(v) if isinstance(v, np.floating) else v
                                 for k, v in value.items()}
            elif isinstance(value, (np.integer, np.floating)):
                info_dict[key] = int(value) if isinstance(value, np.integer) else float(value)

        with open(info_path, 'w') as f:
            json.dump(info_dict, f, indent=2)
        print(f"Saved dataset info: {info_path}")

        # Save quality report as JSON
        report_path = self.output_dir / "quality_report.json"
        report_dict = asdict(self.quality_report)
        # Convert numpy types to native Python types for JSON serialization
        for key, value in report_dict.items():
            if isinstance(value, dict):
                report_dict[key] = {k: int(v) if isinstance(v, np.integer) else
                                   float(v) if isinstance(v, np.floating) else v
                                   for k, v in value.items()}
            elif isinstance(value, (np.integer, np.floating)):
                report_dict[key] = int(value) if isinstance(value, np.integer) else float(value)

        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        print(f"Saved quality report: {report_path}")

        # Save LLM insights if available
        if self.llm_insights:
            llm_insights_path = self.output_dir / "llm_insights.json"
            with open(llm_insights_path, 'w') as f:
                json.dump(asdict(self.llm_insights), f, indent=2)
            print(f"Saved LLM insights: {llm_insights_path}")

        # Save basic statistics
        stats_path = self.output_dir / "basic_statistics.csv"
        cleaned_df.describe(include='all').to_csv(stats_path)
        print(f"Saved basic statistics: {stats_path}")

    def run_full_analysis(self) -> Tuple[pd.DataFrame, DatasetInfo, DataQualityReport]:
        """Run the complete data scout analysis pipeline"""
        print(f"Starting Data Scout Analysis for {self.competition_name}")
        print("=" * 60)

        # Load data
        train_df, test_df = self.load_data()

        # Perform EDA
        dataset_info = self.perform_eda()

        # Clean data
        cleaned_df = self.basic_cleaning()

        # Generate LLM insights if enabled
        if self.enable_llm and self.llm_coordinator:
            print("\nStep 4: Generating LLM insights...")
            self.llm_insights = self.analyze_with_llm()
            if self.llm_insights:
                self._print_llm_insights()

        # Save outputs
        self.save_outputs(cleaned_df)

        print(f"\nData Scout Analysis Complete!")
        print(f"Outputs saved to: {self.output_dir}")

        return cleaned_df, dataset_info, self.quality_report


def main():
    """Main entry point for the Data Scout Agent"""
    parser = argparse.ArgumentParser(description="Data Scout Agent - EDA and Basic Cleaning")
    parser.add_argument("competition_path", type=Path,
                       help="Path to competition directory containing train.csv")
    parser.add_argument("--output-dir", type=Path, default=None,
                       help="Output directory (default: competition_path/scout_output)")
    parser.add_argument("--outlier-threshold", type=float, default=3.0,
                       help="Standard deviation threshold for outlier detection")
    parser.add_argument("--missing-threshold", type=float, default=0.8,
                       help="Drop columns with missing values above this threshold")

    args = parser.parse_args()

    # Validate competition path
    if not args.competition_path.exists():
        print(f"ERROR: Competition path does not exist: {args.competition_path}")
        sys.exit(1)

    train_csv = args.competition_path / "train.csv"
    if not train_csv.exists():
        print(f"ERROR: Training data not found: {train_csv}")
        sys.exit(1)

    try:
        # Initialize scout
        scout = DataScout(args.competition_path, args.output_dir)
        scout.outlier_threshold = args.outlier_threshold
        scout.missing_threshold = args.missing_threshold

        # Run analysis
        cleaned_df, dataset_info, quality_report = scout.run_full_analysis()

        print(f"\nAnalysis completed successfully!")
        return 0

    except Exception as e:
        print(f"ERROR: Error during analysis: {e}")
        return 1


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)
    exit(main())