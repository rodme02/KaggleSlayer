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

    def run(self, optimize_hyperparameters: bool = True, create_ensemble: bool = True) -> Dict[str, Any]:
        """Run the complete model selection and training process."""
        self.log_info("Starting model selection process...")

        try:
            # Load engineered data
            train_df, _ = self._load_engineered_data()

            # Get target column
            feature_results = self.load_results("feature_engineer_results.json")
            target_column = feature_results.get('target_column')

            if not target_column or target_column not in train_df.columns:
                self.log_error("Target column not found")
                raise ValueError("Target column not found")

            # Prepare data
            X = train_df.drop(columns=[target_column])
            y = train_df[target_column]

            # Adaptive sampling based on dataset size (for faster initial evaluation)
            original_size = len(X)
            if original_size < 10000:
                sample_size = original_size  # No sampling for small datasets
            elif original_size < 100000:
                sample_size = min(10000, original_size)  # Sample to 10K
            else:
                sample_size = min(5000, original_size)  # Sample to 5K for very large datasets

            if len(X) > sample_size:
                self.log_info(f"Sampling {sample_size} rows from {original_size} for faster evaluation")
                from sklearn.model_selection import train_test_split
                try:
                    # Try stratified sampling for classification
                    if y.nunique() < 20:
                        X, _, y, _ = train_test_split(X, y, train_size=sample_size,
                                                    random_state=self.get_config("pipeline.cv_random_state", 42),
                                                    stratify=y)
                    else:
                        X, _, y, _ = train_test_split(X, y, train_size=sample_size,
                                                    random_state=self.get_config("pipeline.cv_random_state", 42))
                except Exception as e:
                    self.log_warning(f"Stratified sampling failed, using random sampling: {e}")
                    X, _, y, _ = train_test_split(X, y, train_size=sample_size,
                                                random_state=self.get_config("pipeline.cv_random_state", 42))

            # Determine problem type
            problem_type = "classification" if y.nunique() < 20 else "regression"
            self.log_info(f"Detected problem type: {problem_type}")

            # Get available models
            available_models = self.model_factory.get_available_model_names(problem_type)
            self.log_info(f"Available models: {available_models}")

            # Evaluate models
            model_results = []
            #for model_name in available_models[:5]:  # Limit to top 5 for speed
            for model_name in available_models:
                try:
                    self.log_info(f"Evaluating {model_name}...")

                    # Create model with default parameters
                    model = self.model_factory.create_model(model_name, problem_type=problem_type)

                    # Evaluate model
                    eval_result = self.evaluator.evaluate_model(
                        model, X, y, problem_type=problem_type, model_name=model_name
                    )

                    model_results.append((model_name, model, eval_result.cv_mean))
                    self.log_info(f"{model_name} CV score: {eval_result.cv_mean:.4f}")

                except Exception as e:
                    self.log_warning(f"Failed to evaluate {model_name}: {e}")
                    continue

            if not model_results:
                raise ValueError("No models could be evaluated successfully")

            # Select best models
            best_models = self.ensemble_builder.select_best_models(model_results, top_n=3)

            # Optimize hyperparameters for best model
            best_model_name, best_model, best_score = max(model_results, key=lambda x: x[2])
            optimized_params = {}

            if optimize_hyperparameters:
                self.log_info(f"Optimizing hyperparameters for {best_model_name}...")
                try:
                    param_space = self.model_factory.get_parameter_space(best_model_name, problem_type)
                    if param_space:
                        optimization_result = self.optimizer.optimize(
                            lambda params: self.model_factory.create_model(best_model_name, params, problem_type),
                            param_space, X, y, problem_type
                        )
                        optimized_params = optimization_result.best_params
                        self.log_info(f"Optimization completed. Best score: {optimization_result.best_score:.4f}")
                except Exception as e:
                    self.log_warning(f"Hyperparameter optimization failed: {e}")

            # Create ensemble - try both stacking and voting, use the best
            ensemble_score = None
            ensemble_type = None
            ensemble_model = None
            if create_ensemble and len(best_models) > 1:
                self.log_info("Creating ensembles...")
                try:
                    # Try stacking ensemble
                    stacking_result = self.ensemble_builder.create_stacking_ensemble(
                        best_models, X, y, problem_type
                    )
                    stacking_score = stacking_result.ensemble_score
                    self.log_info(f"Stacking ensemble score: {stacking_score:.4f}")

                    # Try voting ensemble
                    voting_result = self.ensemble_builder.create_voting_ensemble(
                        best_models, X, y, problem_type
                    )
                    voting_score = voting_result.ensemble_score
                    self.log_info(f"Voting ensemble score: {voting_score:.4f}")

                    # Use the best ensemble
                    if stacking_score > voting_score:
                        ensemble_score = stacking_score
                        ensemble_type = "stacking"
                        ensemble_model = stacking_result.ensemble_model
                        self.log_info(f"Using stacking ensemble (better score)")
                    else:
                        ensemble_score = voting_score
                        ensemble_type = "voting"
                        ensemble_model = voting_result.ensemble_model
                        self.log_info(f"Using voting ensemble (better score)")

                except Exception as e:
                    self.log_warning(f"Ensemble creation failed: {e}")

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
            raise

    def _load_engineered_data(self) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Load engineered data from feature engineer."""
        try:
            train_df = self.file_manager.load_dataframe("train_engineered.csv")
            test_df = None

            if self.file_manager.file_exists("test_engineered.csv"):
                test_df = self.file_manager.load_dataframe("test_engineered.csv")

            return train_df, test_df

        except FileNotFoundError:
            self.log_error("Engineered data not found. Run feature engineer first.")
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