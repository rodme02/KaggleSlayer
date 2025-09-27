#!/usr/bin/env python3
"""
Baseline Model Agent - Simple models for initial benchmarking

Creates baseline models using cleaned data from Data Scout:
- Classification: Logistic Regression
- Regression: Linear Regression
- Basic preprocessing with simple categorical encoding
- Cross-validation evaluation
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import joblib


@dataclass
class BaselineResults:
    """Structure to hold baseline model results"""
    competition_name: str
    model_type: str
    problem_type: str
    target_column: str
    features_used: List[str]
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    feature_importance: Optional[Dict[str, float]]
    preprocessing_steps: List[str]
    model_params: Dict[str, Any]
    training_timestamp: str


class BaselineModel:
    """
    Baseline Model Agent for creating simple benchmark models
    """

    def __init__(self, competition_path: Path, use_scout_output: bool = True):
        self.competition_path = Path(competition_path)
        self.competition_name = self.competition_path.name
        self.use_scout_output = use_scout_output

        # Paths
        self.scout_dir = self.competition_path / "scout_output"
        self.model_dir = self.competition_path / "baseline_model"
        self.model_dir.mkdir(exist_ok=True)

        # Data containers
        self.train_df = None
        self.test_df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.model = None
        self.scaler = None
        self.encoders = {}

        # Configuration
        self.random_state = 42
        self.cv_folds = 5

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training and test data"""
        print(f"Loading data for competition: {self.competition_name}")

        if self.use_scout_output and self.scout_dir.exists():
            # Use cleaned data from scout
            train_path = self.scout_dir / "train_cleaned.csv"
            if train_path.exists():
                self.train_df = pd.read_csv(train_path)
                print(f"Loaded cleaned training data: {self.train_df.shape}")
            else:
                # Fallback to original data
                self.train_df = pd.read_csv(self.competition_path / "train.csv")
                print(f"Loaded original training data: {self.train_df.shape}")
        else:
            # Use original data
            self.train_df = pd.read_csv(self.competition_path / "train.csv")
            print(f"Loaded original training data: {self.train_df.shape}")

        # Load test data
        test_path = self.competition_path / "test.csv"
        if test_path.exists():
            self.test_df = pd.read_csv(test_path)
            print(f"Loaded test data: {self.test_df.shape}")
        else:
            print("No test data found")
            self.test_df = None

        return self.train_df, self.test_df

    def load_scout_info(self) -> Dict[str, Any]:
        """Load dataset info from scout analysis"""
        info_path = self.scout_dir / "dataset_info.json"
        if info_path.exists():
            with open(info_path, 'r') as f:
                return json.load(f)
        return {}

    def detect_target_and_problem_type(self) -> Tuple[str, str]:
        """Detect target column and problem type"""
        # Try to get from scout info first
        scout_info = self.load_scout_info()
        if scout_info.get('target_column') and scout_info.get('target_type'):
            return scout_info['target_column'], scout_info['target_type']

        # Fallback detection logic
        common_targets = ['target', 'label', 'y', 'class', 'outcome', 'prediction',
                         'survived', 'saleprice', 'price']

        # Check for common target names
        for col in self.train_df.columns:
            if col.lower() in common_targets:
                unique_ratio = self.train_df[col].nunique() / len(self.train_df)
                if unique_ratio < 0.1:
                    return col, "classification"
                else:
                    return col, "regression"

        # Check if target column is not in test set
        if self.test_df is not None:
            train_cols = set(self.train_df.columns)
            test_cols = set(self.test_df.columns)
            diff_cols = train_cols - test_cols

            if len(diff_cols) == 1:
                target_col = list(diff_cols)[0]
                target_series = self.train_df[target_col]

                # Check data type first
                if target_series.dtype in ['int64', 'float64']:
                    # For numerical data, check if it's continuous or discrete
                    unique_count = target_series.nunique()
                    total_count = len(target_series)
                    unique_ratio = unique_count / total_count

                    # If more than 50 unique values OR unique ratio > 0.05, likely regression
                    if unique_count > 50 or unique_ratio > 0.05:
                        return target_col, "regression"
                    else:
                        # Check if values are integers (could be ordinal classification)
                        if target_series.dtype == 'int64' and all(target_series == target_series.astype(int)):
                            return target_col, "classification"
                        else:
                            return target_col, "regression"
                else:
                    # Non-numerical data is classification
                    return target_col, "classification"

        raise ValueError("Could not automatically detect target column")

    def preprocess_features(self, target_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Basic preprocessing of features"""
        print("\nPreprocessing features...")
        preprocessing_steps = []

        # Separate features from target and remove ID-like columns
        feature_cols = [col for col in self.train_df.columns if col != target_col]

        # Remove ID-like columns (common patterns)
        id_patterns = ['id', 'index', 'key', 'passengerid', 'customerid']
        feature_cols = [col for col in feature_cols
                       if not any(pattern in col.lower() for pattern in id_patterns)]

        if len(feature_cols) < len(self.train_df.columns) - 1:
            removed_cols = set(self.train_df.columns) - set(feature_cols) - {target_col}
            preprocessing_steps.append(f"Removed ID columns: {list(removed_cols)}")

        X_train = self.train_df[feature_cols].copy()
        y_train = self.train_df[target_col].copy()

        # Prepare test features if available
        if self.test_df is not None:
            X_test = self.test_df[feature_cols].copy()
        else:
            X_test = None

        # Handle missing values first - check both train and test
        all_columns = X_train.columns

        for col in all_columns:
            train_missing = X_train[col].isnull().sum() > 0
            test_missing = X_test is not None and X_test[col].isnull().sum() > 0

            if train_missing or test_missing:
                if X_train[col].dtype in ['int64', 'float64']:
                    # Fill numerical with median
                    median_val = X_train[col].median()
                    X_train[col] = X_train[col].fillna(median_val)
                    if X_test is not None:
                        X_test[col] = X_test[col].fillna(median_val)
                    preprocessing_steps.append(f"Filled missing {col} with median ({median_val})")
                else:
                    # Fill categorical with mode
                    mode_val = X_train[col].mode()
                    if len(mode_val) > 0:
                        fill_val = mode_val[0]
                        X_train[col] = X_train[col].fillna(fill_val)
                        X_train[col] = X_train[col].infer_objects(copy=False)
                        if X_test is not None:
                            X_test[col] = X_test[col].fillna(fill_val)
                            X_test[col] = X_test[col].infer_objects(copy=False)
                        preprocessing_steps.append(f"Filled missing {col} with mode ({fill_val})")
                    else:
                        # Fallback to 'Unknown' for categorical
                        X_train[col] = X_train[col].fillna('Unknown')
                        X_train[col] = X_train[col].infer_objects(copy=False)
                        if X_test is not None:
                            X_test[col] = X_test[col].fillna('Unknown')
                            X_test[col] = X_test[col].infer_objects(copy=False)
                        preprocessing_steps.append(f"Filled missing {col} with 'Unknown'")

        # Encode categorical features
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            # Skip categorical columns with too many unique values (likely text/IDs)
            n_unique = X_train[col].nunique()
            if n_unique > 50:
                X_train = X_train.drop(columns=[col])
                if X_test is not None:
                    X_test = X_test.drop(columns=[col])
                preprocessing_steps.append(f"Dropped high-cardinality column {col} ({n_unique} unique values)")
                continue

            encoder = LabelEncoder()

            # Fit on combined train+test data if test exists
            if X_test is not None:
                combined_values = pd.concat([X_train[col], X_test[col]]).astype(str)
                encoder.fit(combined_values)
                X_train[col] = encoder.transform(X_train[col].astype(str))
                X_test[col] = encoder.transform(X_test[col].astype(str))
            else:
                X_train[col] = encoder.fit_transform(X_train[col].astype(str))

            self.encoders[col] = encoder
            preprocessing_steps.append(f"Label encoded {col}")

        # Scale numerical features
        numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) > 0:
            self.scaler = StandardScaler()
            X_train[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
            if X_test is not None:
                X_test[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
            preprocessing_steps.append(f"Scaled {len(numerical_cols)} numerical features")

        self.preprocessing_steps = preprocessing_steps
        print(f"Preprocessing completed: {len(preprocessing_steps)} steps")

        return X_train, X_test, y_train

    def train_baseline_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                           problem_type: str) -> Any:
        """Train baseline model based on problem type"""
        print(f"\nTraining {problem_type} baseline model...")

        if problem_type == "classification":
            # Check if binary or multiclass
            n_classes = y_train.nunique()
            n_features = len(X_train.columns)

            # Use different models based on data dimensionality
            if n_features > 500:  # High-dimensional data (like images)
                from sklearn.linear_model import SGDClassifier
                print(f"High-dimensional data detected ({n_features} features), using SGDClassifier")
                self.model = SGDClassifier(
                    loss='log_loss',  # Equivalent to LogisticRegression
                    random_state=self.random_state,
                    max_iter=1000,
                    alpha=0.01  # Regularization
                )
            elif n_classes == 2:
                self.model = LogisticRegression(
                    random_state=self.random_state,
                    max_iter=1000
                )
            else:
                # For multiclass, use OneVsRestClassifier to avoid deprecation warnings
                from sklearn.multiclass import OneVsRestClassifier
                self.model = OneVsRestClassifier(
                    LogisticRegression(
                        random_state=self.random_state,
                        max_iter=1000
                    )
                )
        else:  # regression
            # Use Ridge regression for better generalization
            from sklearn.linear_model import Ridge
            self.model = Ridge(alpha=1.0, random_state=self.random_state)

        # Train the model
        self.model.fit(X_train, y_train)
        print(f"Model trained with {len(X_train.columns)} features")

        return self.model

    def evaluate_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                      problem_type: str) -> BaselineResults:
        """Evaluate model using cross-validation"""
        print(f"\nEvaluating model with {self.cv_folds}-fold cross-validation...")

        if problem_type == "classification":
            # Use stratified CV for classification
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                               random_state=self.random_state)
            scoring = 'accuracy'
        else:
            # Use regular CV for regression
            cv = KFold(n_splits=self.cv_folds, shuffle=True,
                      random_state=self.random_state)
            # Use negative RMSE for regression (more interpretable than R²)
            scoring = 'neg_root_mean_squared_error'

        # Perform cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train,
                                  cv=cv, scoring=scoring)

        if problem_type == "regression":
            # Convert negative RMSE back to positive
            cv_scores = -cv_scores
            print(f"CV RMSE: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        else:
            print(f"CV {scoring}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # Get feature importance if available
        feature_importance = None
        if hasattr(self.model, 'coef_'):
            if problem_type == "classification" and self.model.coef_.ndim > 1:
                # Multi-class case - use average absolute coefficients
                coefs = np.mean(np.abs(self.model.coef_), axis=0)
            else:
                # Binary classification or regression
                coefs = np.abs(self.model.coef_).flatten()

            feature_importance = dict(zip(X_train.columns, coefs))

            # Show top 5 features
            sorted_features = sorted(feature_importance.items(),
                                   key=lambda x: x[1], reverse=True)
            print(f"\nTop 5 most important features:")
            for feat, importance in sorted_features[:5]:
                print(f"  {feat}: {importance:.4f}")

        # Create results object
        results = BaselineResults(
            competition_name=self.competition_name,
            model_type=type(self.model).__name__,
            problem_type=problem_type,
            target_column=y_train.name,
            features_used=list(X_train.columns),
            cv_scores=cv_scores.tolist(),
            cv_mean=float(cv_scores.mean()),
            cv_std=float(cv_scores.std()),
            feature_importance=feature_importance,
            preprocessing_steps=self.preprocessing_steps,
            model_params=self.model.get_params(),
            training_timestamp=datetime.now().isoformat()
        )

        return results

    def generate_predictions(self, X_test: pd.DataFrame, problem_type: str) -> pd.DataFrame:
        """Generate predictions for test set"""
        if X_test is None:
            print("No test data available for predictions")
            return None

        print("\nGenerating predictions on test set...")

        if problem_type == "classification":
            predictions = self.model.predict(X_test)
            probabilities = None
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(X_test)
        else:
            predictions = self.model.predict(X_test)
            probabilities = None

        # Create predictions dataframe
        if 'Id' in self.test_df.columns:
            id_col = 'Id'
        elif 'id' in self.test_df.columns:
            id_col = 'id'
        elif 'PassengerId' in self.test_df.columns:
            id_col = 'PassengerId'
        else:
            # Create a simple index-based ID
            id_col = 'Id'
            pred_df = pd.DataFrame({'Id': range(len(predictions))})

        if id_col in self.test_df.columns:
            pred_df = pd.DataFrame({id_col: self.test_df[id_col]})
        else:
            pred_df = pd.DataFrame({'Id': range(len(predictions))})

        # Add predictions
        target_col = self.y_train.name if hasattr(self, 'y_train') else 'target'
        pred_df[target_col] = predictions

        # Add probabilities for binary classification
        if probabilities is not None and probabilities.shape[1] == 2:
            pred_df[f'{target_col}_proba'] = probabilities[:, 1]

        print(f"Generated {len(pred_df)} predictions")
        return pred_df

    def save_outputs(self, results: BaselineResults, predictions: pd.DataFrame = None):
        """Save model, results, and predictions"""
        print(f"\nSaving outputs to {self.model_dir}")

        # Save model and preprocessors
        model_path = self.model_dir / "baseline_model.joblib"
        joblib.dump(self.model, model_path)
        print(f"Saved model: {model_path}")

        if self.scaler:
            scaler_path = self.model_dir / "scaler.joblib"
            joblib.dump(self.scaler, scaler_path)
            print(f"Saved scaler: {scaler_path}")

        if self.encoders:
            encoders_path = self.model_dir / "encoders.joblib"
            joblib.dump(self.encoders, encoders_path)
            print(f"Saved encoders: {encoders_path}")

        # Save results
        results_path = self.model_dir / "baseline_results.json"
        with open(results_path, 'w') as f:
            json.dump(asdict(results), f, indent=2)
        print(f"Saved results: {results_path}")

        # Save predictions
        if predictions is not None:
            pred_path = self.model_dir / "predictions.csv"
            predictions.to_csv(pred_path, index=False)
            print(f"Saved predictions: {pred_path}")

    def run_baseline_pipeline(self) -> Tuple[BaselineResults, Optional[pd.DataFrame]]:
        """Run the complete baseline modeling pipeline"""
        print(f"Starting Baseline Model Pipeline for {self.competition_name}")
        print("=" * 60)

        # Load data
        train_df, test_df = self.load_data()

        # Detect target and problem type
        target_col, problem_type = self.detect_target_and_problem_type()
        print(f"Detected target: {target_col} ({problem_type})")

        # Preprocess features
        X_train, X_test, y_train = self.preprocess_features(target_col)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train

        # Train model
        model = self.train_baseline_model(X_train, y_train, problem_type)

        # Evaluate model
        results = self.evaluate_model(X_train, y_train, problem_type)

        # Generate predictions
        predictions = self.generate_predictions(X_test, problem_type) if X_test is not None else None

        # Save outputs
        self.save_outputs(results, predictions)

        print(f"\nBaseline Model Pipeline Complete!")
        print(f"CV Score: {results.cv_mean:.4f} (+/- {results.cv_std * 2:.4f})")
        print(f"Outputs saved to: {self.model_dir}")

        return results, predictions


def main():
    """Main entry point for the Baseline Model Agent"""
    parser = argparse.ArgumentParser(description="Baseline Model Agent")
    parser.add_argument("competition_path", type=Path,
                       help="Path to competition directory")
    parser.add_argument("--use-original", action="store_true",
                       help="Use original data instead of scout cleaned data")
    parser.add_argument("--cv-folds", type=int, default=5,
                       help="Number of cross-validation folds")

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
        # Initialize baseline model agent
        baseline = BaselineModel(
            args.competition_path,
            use_scout_output=not args.use_original
        )
        baseline.cv_folds = args.cv_folds

        # Run pipeline
        results, predictions = baseline.run_baseline_pipeline()

        print(f"\nBaseline modeling completed successfully!")
        return 0

    except Exception as e:
        print(f"ERROR: Error during baseline modeling: {e}")
        return 1


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    exit(main())