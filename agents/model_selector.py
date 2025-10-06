"""
Streamlined Model Selector Agent using modular components.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from pathlib import Path

from .base_agent import BaseAgent
from core.models import ModelFactory, ModelEvaluator, HyperparameterOptimizer, EnsembleBuilder
from utils.config import ConfigManager


class ModelSelectorAgent(BaseAgent):
    """Streamlined agent for model selection, training, and evaluation."""

    def __init__(self, competition_name: str, competition_path: Path,
                 config: Optional[ConfigManager] = None):
        super().__init__(competition_name, competition_path, config)

        # Initialize components
        self.model_factory = ModelFactory(random_state=self.get_config("pipeline.cv_random_state", 42))
        self.evaluator = ModelEvaluator(
            cv_folds=self.get_config("pipeline.cv_folds", 5),  # 5 folds for stable evaluation
            random_state=self.get_config("pipeline.cv_random_state", 42)
        )
        self.optimizer = HyperparameterOptimizer(
            n_trials=self.get_config("pipeline.optuna_trials", 20),  # Increased trials for better optimization
            timeout=self.get_config("pipeline.optuna_timeout", 300),  # 5 min timeout
            cv_folds=self.get_config("pipeline.cv_folds", 5),  # 5 folds for stable evaluation
            random_state=self.get_config("pipeline.cv_random_state", 42)
        )
        self.ensemble_builder = EnsembleBuilder(
            cv_folds=self.get_config("pipeline.cv_folds", 5),  # 5 folds for stable evaluation
            random_state=self.get_config("pipeline.cv_random_state", 42)
        )

    def run(self, optimize_hyperparameters: bool = True, create_ensemble: bool = True,
            use_feature_pipeline: bool = True) -> Dict[str, Any]:
        """Run the complete model selection and training process.

        Args:
            optimize_hyperparameters: Whether to optimize hyperparameters
            create_ensemble: Whether to create ensemble models
            use_feature_pipeline: If True, uses CLEANED data + feature pipeline (leak-free CV).
                                 If False, uses pre-engineered data (backward compatibility).
        """
        self.log_info("Starting model selection process...")

        try:
            # CRITICAL: Load CLEANED data (not engineered) to prevent leakage
            if use_feature_pipeline:
                self.log_info("Using leak-free CV: Feature engineering will happen inside each fold")
                train_df, _ = self._load_cleaned_data()
                # Get target column from data scout results
                data_results = self.load_results("data_scout_results.json")
                target_column = data_results.get('dataset_info', {}).get('target_column')
            else:
                self.log_warning("Using pre-engineered data (may have leakage)")
                train_df, _ = self._load_engineered_data()
                # Get target column from feature engineer results
                feature_results = self.load_results("feature_engineer_results.json")
                target_column = feature_results.get('target_column')

            if not target_column or target_column not in train_df.columns:
                self.log_error("Target column not found")
                raise ValueError("Target column not found")

            # Prepare data
            X_full = train_df.drop(columns=[target_column])
            y_full = train_df[target_column]

            # Encode target if it's categorical (string labels)
            from sklearn.preprocessing import LabelEncoder
            target_encoder = None
            if y_full.dtype == 'object' or y_full.dtype.name == 'category':
                target_encoder = LabelEncoder()
                y_full_encoded = pd.Series(target_encoder.fit_transform(y_full), index=y_full.index, name=y_full.name)
                print(f"Encoded target labels: {list(target_encoder.classes_)} -> {list(range(len(target_encoder.classes_)))}")
            else:
                y_full_encoded = y_full

            # Smart sampling for large datasets (>50K rows)
            # Uses STRATIFIED sampling to preserve class distribution
            # NOTE: We keep X_full and y_full_encoded for final training
            original_size = len(X_full)
            X = X_full  # Start with full dataset
            y = y_full_encoded  # Use encoded version

            if original_size > 50000:
                # For very large datasets, sample to make CV feasible
                # Use stratified sampling to preserve distribution
                from sklearn.model_selection import train_test_split

                # Adaptive sample size based on dataset size
                # For very large datasets (>200K), use smaller samples for speed
                import math
                if original_size > 200000:
                    sample_size = 15000  # 15K for very large datasets
                elif original_size > 100000:
                    sample_size = 20000  # 20K for large datasets
                else:
                    sample_size = min(30000, int(math.sqrt(original_size) * 100))
                sample_size = max(10000, sample_size)  # At least 10K

                self.log_info(f"Large dataset detected ({original_size} samples)")
                self.log_info(f"Using stratified sample of {sample_size} for model evaluation")

                try:
                    # Try stratified sampling for classification
                    if y.nunique() < 20:
                        X, _, y, _ = train_test_split(
                            X, y,
                            train_size=sample_size,
                            random_state=self.get_config("pipeline.cv_random_state", 42),
                            stratify=y
                        )
                    else:
                        # For regression, use simple random sampling
                        X, _, y, _ = train_test_split(
                            X, y,
                            train_size=sample_size,
                            random_state=self.get_config("pipeline.cv_random_state", 42)
                        )
                    self.log_info(f"Stratified sampling successful: {len(X)} samples")
                except Exception as e:
                    # Fallback to random sampling if stratification fails
                    self.log_warning(f"Stratified sampling failed ({e}), using random sample")
                    X, _, y, _ = train_test_split(
                        X, y,
                        train_size=sample_size,
                        random_state=self.get_config("pipeline.cv_random_state", 42)
                    )
            else:
                self.log_info(f"Using full dataset ({original_size} samples) for model evaluation")

            # Determine problem type
            problem_type = "classification" if y.nunique() < 20 else "regression"
            self.log_info(f"Detected problem type: {problem_type}")

            # Adaptive CV folds based on dataset size
            # Larger datasets can use fewer folds for faster evaluation
            if original_size > 100000:
                cv_folds = 3  # 3-fold CV for very large datasets
            elif original_size > 50000:
                cv_folds = 3  # 3-fold CV for large datasets
            else:
                cv_folds = self.get_config("pipeline.cv_folds", 5)  # Default 5-fold

            if cv_folds != self.evaluator.cv_folds:
                self.log_info(f"Using {cv_folds}-fold CV (adapted for dataset size)")
                self.evaluator.cv_folds = cv_folds

            # Create feature pipeline if using leak-free CV
            if use_feature_pipeline:
                from .feature_engineer import FeatureEngineerAgent
                feature_engineer = FeatureEngineerAgent(
                    self.competition_name,
                    self.competition_path,
                    self.config
                )
                feature_pipeline = feature_engineer.create_feature_pipeline()
                self.log_info("Using leak-free CV (feature engineering inside each fold)")
            else:
                feature_pipeline = None

            # Get available models
            available_models = self.model_factory.get_available_model_names(problem_type)

            # Print evaluation setup
            print(f"\n[>>] Model Evaluation Setup:")
            print(f"    Problem type: {problem_type}")
            print(f"    Dataset size: {len(X):,} samples x {len(X.columns)} features")
            if original_size > len(X):
                print(f"    (Sampled from {original_size:,} for faster CV)")
            print(f"    CV strategy: {cv_folds}-fold cross-validation")
            print(f"    Models to evaluate: {len(available_models)}")
            print(f"    Parallel jobs: {-1 if len(available_models) > 1 else 1} (all cores)")
            if use_feature_pipeline:
                print(f"    Feature engineering: Inside CV folds (leak-free)")
            print()

            # Evaluate models with parallel processing
            model_results = []

            # Parallel evaluation helper function
            def evaluate_single_model(i, model_name):
                """Evaluate a single model - used for parallel processing."""
                import time
                start_time = time.time()

                # Suppress logs at the start of this function for each thread
                if use_feature_pipeline:
                    from utils.logging import suppress_feature_logs
                    import contextlib

                    # Use context manager for entire evaluation
                    @contextlib.contextmanager
                    def evaluation_context():
                        with suppress_feature_logs():
                            yield
                else:
                    import contextlib
                    @contextlib.contextmanager
                    def evaluation_context():
                        yield

                try:
                    with evaluation_context():
                        # Create model with default parameters
                        model = self.model_factory.create_model(model_name, problem_type=problem_type)

                        # CRITICAL: If using feature pipeline, wrap model in a full pipeline
                        if feature_pipeline is not None:
                            from sklearn.pipeline import Pipeline
                            # Create a fresh feature pipeline for this model
                            from .feature_engineer import FeatureEngineerAgent
                            feature_engineer = FeatureEngineerAgent(
                                self.competition_name,
                                self.competition_path,
                                self.config
                            )
                            model_feature_pipeline = feature_engineer.create_feature_pipeline()
                            full_pipeline = Pipeline([
                                ('features', model_feature_pipeline),
                                ('model', model)
                            ])
                        else:
                            full_pipeline = model

                        # Evaluate model with timeout check
                        eval_result = self.evaluator.evaluate_model(
                            full_pipeline, X, y, problem_type=problem_type, model_name=model_name
                        )

                        elapsed = time.time() - start_time
                        return (model_name, full_pipeline, eval_result.cv_mean, None, elapsed)

                except KeyboardInterrupt:
                    # User interrupted - propagate it
                    raise
                except Exception as e:
                    elapsed = time.time() - start_time
                    return (model_name, None, None, str(e), elapsed)

            # Run evaluations in parallel (speeds up evaluation significantly)
            print(f"[>>] Evaluating {len(available_models)} models in parallel...")
            print(f"    (This may take a few minutes)\n")

            from joblib import Parallel, delayed
            import time
            eval_start = time.time()

            parallel_results = Parallel(n_jobs=-1, backend='threading', verbose=0)(
                delayed(evaluate_single_model)(i, model_name)
                for i, model_name in enumerate(available_models, 1)
            )

            eval_total_time = time.time() - eval_start

            # Process results and display
            print(f"[>>] Model Evaluation Results:")
            print(f"    {'-'*60}")
            for i, result in enumerate(parallel_results, 1):
                # Unpack result (may have 4 or 5 elements depending on success)
                if len(result) == 5:
                    model_name, full_pipeline, cv_mean, error, elapsed = result
                else:
                    # Fallback for old format
                    model_name, full_pipeline, cv_mean, error = result
                    elapsed = 0

                if error is None:
                    model_results.append((model_name, full_pipeline, cv_mean))
                    print(f"    [{i}/{len(available_models)}] {model_name:<25} CV = {cv_mean:.4f}  ({elapsed:.1f}s)")
                else:
                    print(f"    [{i}/{len(available_models)}] {model_name:<25} FAILED - {error}")
                    self.log_warning(f"  Error: {error}")

            print(f"    {'-'*60}")
            print(f"    Total evaluation time: {eval_total_time:.1f}s\n")

            if not model_results:
                raise ValueError("No models could be evaluated successfully")

            # Select best models FOR REFERENCE (these are fitted pipelines, not for ensemble)
            best_model_results = self.ensemble_builder.select_best_models(model_results, top_n=3, problem_type=problem_type)

            # Optimize hyperparameters for best model
            # For classification: higher is better, for regression: lower is better
            if problem_type == "classification":
                best_model_name, best_model, best_score = max(model_results, key=lambda x: x[2])
            else:
                best_model_name, best_model, best_score = min(model_results, key=lambda x: x[2])
            optimized_params = {}

            if optimize_hyperparameters:
                self.log_info(f"Optimizing hyperparameters for {best_model_name}...")
                try:
                    param_space = self.model_factory.get_parameter_space(best_model_name, problem_type)
                    if param_space:
                        # Suppress CV error tracebacks during optimization
                        from utils.logging import suppress_sklearn_warnings
                        with suppress_sklearn_warnings():
                            optimization_result = self.optimizer.optimize(
                                lambda params: self.model_factory.create_model(best_model_name, params, problem_type),
                                param_space, X, y, problem_type
                            )
                        optimized_params = optimization_result.best_params
                        self.log_info(f"Optimization completed. Best score: {optimization_result.best_score:.4f}")
                except Exception as e:
                    self.log_warning(f"Hyperparameter optimization failed: {e}")

            # Create ensemble - try both stacking and voting, use the best
            print()  # Empty line
            ensemble_score = None
            ensemble_type = None
            ensemble_model = None

            if use_feature_pipeline and create_ensemble and len(best_model_results) > 1:
                print("  Ensemble disabled (using best single model with leak-free CV)")
            elif create_ensemble and len(best_model_results) > 1:
                print("  Creating ensembles...", end=' ', flush=True)
                try:
                    # Create FRESH unfitted pipelines for each best model
                    if use_feature_pipeline:
                        from sklearn.pipeline import Pipeline
                        from .feature_engineer import FeatureEngineerAgent

                        # Get list of best model names
                        best_model_names = [name for name, _ in best_model_results]

                        # Create fresh pipelines for ensemble
                        ensemble_base_models = []
                        for model_name in best_model_names:
                            # Create fresh feature pipeline
                            feature_engineer = FeatureEngineerAgent(
                                self.competition_name,
                                self.competition_path,
                                self.config
                            )
                            fresh_feature_pipeline = feature_engineer.create_feature_pipeline()

                            # Create fresh model
                            fresh_model = self.model_factory.create_model(model_name, problem_type=problem_type)

                            # Combine into pipeline
                            fresh_pipeline = Pipeline([
                                ('features', fresh_feature_pipeline),
                                ('model', fresh_model)
                            ])
                            ensemble_base_models.append((model_name, fresh_pipeline))
                    else:
                        # For non-pipeline case, create fresh models
                        best_model_names = [name for name, _ in best_model_results]
                        ensemble_base_models = [
                            (name, self.model_factory.create_model(name, problem_type=problem_type))
                            for name in best_model_names
                        ]

                    # Suppress feature engineering logs (but not ensemble logs) during CV
                    from utils.logging_utils import suppress_feature_logs
                    with suppress_feature_logs():
                        # Try stacking ensemble
                        stacking_result = self.ensemble_builder.create_stacking_ensemble(
                            ensemble_base_models, X, y, problem_type
                        )
                        stacking_score = stacking_result.ensemble_score

                        # Try voting ensemble
                        voting_result = self.ensemble_builder.create_voting_ensemble(
                            ensemble_base_models, X, y, problem_type
                        )
                        voting_score = voting_result.ensemble_score

                    # Use the best ensemble
                    if stacking_score > voting_score:
                        ensemble_score = stacking_score
                        ensemble_type = "stacking"
                        ensemble_model = stacking_result.ensemble_model
                        print(f"Stacking: {stacking_score:.4f} (best)")
                    else:
                        ensemble_score = voting_score
                        ensemble_type = "voting"
                        ensemble_model = voting_result.ensemble_model
                        print(f"Voting: {voting_score:.4f} (best)")

                except Exception as e:
                    print(f"FAILED")
                    self.log_warning(f"  Error: {e}")

            # Decide whether to use ensemble or single model for final predictions
            final_model = best_model
            final_model_name = best_model_name
            final_score = best_score

            if ensemble_model is not None and ensemble_score is not None:
                # Use ensemble if it's better than the best single model
                if ensemble_score > best_score:
                    final_model = ensemble_model
                    final_model_name = f"ensemble_{ensemble_type}"
                    final_score = ensemble_score
                    self.log_info(f"Using ensemble for predictions (score: {ensemble_score:.4f} > {best_score:.4f})")
                else:
                    self.log_info(f"Using single model for predictions (score: {best_score:.4f} >= {ensemble_score:.4f})")

            # Fit final model on FULL training data (not sampled)
            if use_feature_pipeline and final_model is not None:
                if original_size > 50000:
                    self.log_info(f"Fitting final pipeline on FULL dataset ({original_size} samples)...")
                else:
                    self.log_info("Fitting final pipeline on full training data...")
                final_model.fit(X_full, y_full_encoded)
                self.log_info("Final pipeline fitted successfully on full dataset")

            # Save the final model (ensemble or best single)
            if final_model is not None:
                try:
                    import joblib
                    model_path = self.file_manager.get_file_path(f"best_model_{final_model_name}.pkl")
                    joblib.dump(final_model, model_path)
                    self.log_info(f"Saved final model ({final_model_name}) to {model_path}")

                    # Also save under a generic name for easy loading
                    generic_path = self.file_manager.get_file_path("best_model.pkl")
                    joblib.dump(final_model, generic_path)

                    # Save target encoder if we encoded the labels
                    if target_encoder is not None:
                        encoder_path = self.file_manager.get_file_path("target_encoder.pkl")
                        joblib.dump(target_encoder, encoder_path)
                        self.log_info(f"Saved target encoder to {encoder_path}")
                except Exception as e:
                    self.log_warning(f"Could not save model: {e}")

            # Create results summary
            results = {
                'competition_name': self.competition_name,
                'problem_type': problem_type,
                'target_column': target_column,
                'models_evaluated': len(model_results),
                'best_model_name': final_model_name,  # Use final model (ensemble or single)
                'best_model_score': final_score,  # Use final score
                'best_single_model': best_model_name,  # Keep track of best single model
                'best_single_score': best_score,
                'optimized_parameters': optimized_params,
                'ensemble_score': ensemble_score,
                'ensemble_type': ensemble_type,
                'model_comparison': [
                    {'name': name, 'score': score}
                    for name, _, score in model_results
                ],
                'data_shape': X.shape,
                'feature_count': len(X.columns)
            }

            # Save results
            self.save_results(results, "model_selector_results.json")

            self.log_info("Model selection completed successfully")
            return results

        except Exception as e:
            self.log_error(f"Model selection failed: {e}")

            # Try to provide partial results if any models were evaluated
            if len(model_results) > 0:
                self.log_warning(f"Partial success: {len(model_results)} models evaluated before failure")
                # Return partial results
                best_model_name, best_model, best_score = max(model_results, key=lambda x: x[2]) if problem_type == "classification" else min(model_results, key=lambda x: x[2])
                partial_results = {
                    'competition_name': self.competition_name,
                    'problem_type': problem_type,
                    'models_evaluated': len(model_results),
                    'best_model_name': best_model_name,
                    'best_model_score': best_score,
                    'partial_results': True,
                    'error': str(e)
                }
                self.save_results(partial_results, "model_selector_results.json")
                return partial_results

            raise

    def _load_engineered_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load engineered data from feature engineer."""
        try:
            train_df = self.file_manager.load_processed_data("train_engineered.csv")
            test_df = None

            if self.file_manager.file_exists(f"{self.file_manager.processed_dir}/test_engineered.csv"):
                test_df = self.file_manager.load_processed_data("test_engineered.csv")

            return train_df, test_df

        except FileNotFoundError:
            self.log_error("Engineered data not found. Run feature engineer first.")
            raise

    def _load_cleaned_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load CLEANED data from data scout (NOT engineered data).

        This is used for leak-free CV where feature engineering happens inside each fold.
        """
        try:
            train_df = self.file_manager.load_processed_data("train_cleaned.csv")
            test_df = None

            if self.file_manager.file_exists(f"{self.file_manager.processed_dir}/test_cleaned.csv"):
                test_df = self.file_manager.load_processed_data("test_cleaned.csv")

            return train_df, test_df

        except FileNotFoundError:
            self.log_error("Cleaned data not found. Run data scout first.")
            raise

    def predict_with_best_model(self, X_test: pd.DataFrame) -> List:
        """Make predictions using the best trained model (could be ensemble or single)."""
        try:
            # Load model results to get best model info
            results = self.load_results("model_selector_results.json")
            if not results:
                raise ValueError("No model results found")

            best_model_name = results.get('best_model_name')
            if not best_model_name:
                raise ValueError("No best model identified")

            # Try to load from generic path first (always has the final best model)
            generic_path = self.file_manager.get_file_path("best_model.pkl")
            if generic_path.exists():
                import joblib
                model = joblib.load(generic_path)
                predictions = model.predict(X_test)
                self.log_info(f"Generated predictions using {best_model_name}")
                return predictions.tolist()

            # Fallback: try specific model path
            model_path = self.file_manager.get_file_path(f"best_model_{best_model_name}.pkl")
            if not model_path.exists():
                # Try to retrain
                self.log_warning(f"Saved model not found, retraining {best_model_name}")
                best_single = results.get('best_single_model', best_model_name)
                return self._retrain_and_predict(best_single, X_test)

            # Load and use the saved model
            import joblib
            model = joblib.load(model_path)
            predictions = model.predict(X_test)

            self.log_info(f"Generated predictions using {best_model_name}")
            return predictions.tolist()

        except Exception as e:
            self.log_error(f"Prediction failed: {e}")
            raise

    def _retrain_and_predict(self, model_name: str, X_test: pd.DataFrame) -> List:
        """Retrain the best model and make predictions."""
        try:
            # Load training data
            train_df, _ = self._load_engineered_data()

            # Get target column
            feature_results = self.load_results("feature_engineer_results.json")
            target_column = feature_results.get('target_column')

            if not target_column or target_column not in train_df.columns:
                raise ValueError("Target column not found in training data")

            # Prepare data
            feature_cols = [col for col in train_df.columns if col != target_column]
            X_train = train_df[feature_cols]
            y_train = train_df[target_column]

            # Ensure test data has same features as training data
            common_features = [col for col in feature_cols if col in X_test.columns]
            X_train = X_train[common_features]
            X_test = X_test[common_features]

            # Get the model
            model = self.model_factory.create_model(model_name)

            # Train the model
            model.fit(X_train, y_train)

            # Save the trained model
            import joblib
            model_path = self.file_manager.get_file_path(f"best_model_{model_name}.pkl")
            joblib.dump(model, model_path)

            # Make predictions
            predictions = model.predict(X_test)

            self.log_info(f"Retrained and used {model_name} for predictions")
            return predictions.tolist()

        except Exception as e:
            self.log_error(f"Model retraining failed: {e}")
            # Return dummy predictions as fallback
            return [0] * len(X_test)