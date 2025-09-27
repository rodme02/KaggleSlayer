#!/usr/bin/env python3
"""
Pipeline Coordinator - Autonomous orchestration of the complete KaggleSlayer pipeline

This coordinator manages the execution of all agents in the optimal sequence,
handles iterative improvements, and provides autonomous competition solving capabilities.
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd
import numpy as np

# Suppress warnings
warnings.filterwarnings('ignore')

try:
    from agents.competition_reader import CompetitionReader
    from agents.data_scout import DataScout
    from agents.feature_engineer import FeatureEngineer
    from agents.model_selector import ModelSelector
    from agents.feedback_analyzer import FeedbackAnalyzer
    from agents.llm_coordinator import LLMCoordinator
    from agents.submitter import Submitter
    AGENTS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import agents: {e}")
    AGENTS_AVAILABLE = False


@dataclass
class PipelineStage:
    """Structure for individual pipeline stage"""
    stage_name: str
    agent_class: str
    status: str  # pending, running, completed, failed, skipped
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: float = 0.0
    output_files: List[str] = None
    performance_metrics: Dict[str, Any] = None
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.output_files is None:
            self.output_files = []
        if self.performance_metrics is None:
            self.performance_metrics = {}


@dataclass
class PipelineExecution:
    """Structure for complete pipeline execution"""
    competition_name: str
    execution_id: str
    pipeline_version: str
    start_time: str
    end_time: Optional[str] = None
    total_duration_seconds: float = 0.0
    stages: List[PipelineStage] = None
    final_cv_score: float = 0.0
    best_model: str = ""
    iteration_count: int = 0
    success: bool = False
    error_summary: List[str] = None
    autonomous_mode: bool = True
    llm_enabled: bool = True

    def __post_init__(self):
        if self.stages is None:
            self.stages = []
        if self.error_summary is None:
            self.error_summary = []


class PipelineCoordinator:
    """
    Pipeline Coordinator for autonomous KaggleSlayer execution
    """

    def __init__(self, competition_path: Path, config_path: Optional[Path] = None,
                 enable_llm: bool = True, autonomous_mode: bool = True,
                 max_iterations: int = 3, auto_submit: bool = True):
        self.competition_path = Path(competition_path)
        self.competition_name = self.competition_path.name
        self.config_path = config_path or Path("config.yaml")
        self.enable_llm = enable_llm
        self.autonomous_mode = autonomous_mode
        self.max_iterations = max_iterations
        self.auto_submit = auto_submit

        # Initialize execution tracking
        self.execution_id = f"{self.competition_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = self.competition_path / "pipeline_execution"
        self.output_dir.mkdir(exist_ok=True)

        # Load configuration
        self.config = self._load_configuration()

        # Define pipeline stages
        self.stages = self._define_pipeline_stages()

        # Initialize execution tracking
        self.current_execution = None
        self.execution_history = []

        # Performance tracking
        self.best_cv_score = 0.0
        self.best_model = ""
        self.improvement_history = []

        print(f"Pipeline Coordinator initialized for {self.competition_name}")
        print(f"Execution ID: {self.execution_id}")
        print(f"LLM enabled: {self.enable_llm}")
        print(f"Autonomous mode: {self.autonomous_mode}")
        print(f"Max iterations: {self.max_iterations}")

    def run_full_pipeline(self) -> PipelineExecution:
        """Run the complete autonomous pipeline"""
        print(f"\nKaggleSlayer Autonomous Pipeline: {self.competition_name}")
        print("=" * 70)

        # Initialize execution tracking
        self.current_execution = PipelineExecution(
            competition_name=self.competition_name,
            execution_id=self.execution_id,
            pipeline_version="1.0.0",
            start_time=datetime.now().isoformat(),
            autonomous_mode=self.autonomous_mode,
            llm_enabled=self.enable_llm
        )

        try:
            # Run initial pipeline
            print("\n[ROCKET] PHASE 1: Initial Pipeline Execution")
            print("-" * 50)
            initial_success = self._run_initial_pipeline()

            if not initial_success:
                self._finalize_execution(success=False)
                return self.current_execution

            # Get initial performance
            initial_performance = self._extract_performance_metrics()
            self.best_cv_score = initial_performance.get("cv_score", 0.0)
            self.best_model = initial_performance.get("best_model", "")

            print(f"\n[DATA] Initial Performance: {self.best_cv_score:.4f} ({self.best_model})")

            # Iterative improvement if autonomous mode enabled
            if self.autonomous_mode and self.max_iterations > 0:
                print(f"\n[ITERATION] PHASE 2: Autonomous Improvement ({self.max_iterations} iterations max)")
                print("-" * 50)

                iteration_success = self._run_iterative_improvement()

            # Finalize execution
            final_performance = self._extract_performance_metrics()
            self.current_execution.final_cv_score = final_performance.get("cv_score", self.best_cv_score)
            self.current_execution.best_model = final_performance.get("best_model", self.best_model)

            # Auto-submit to Kaggle if enabled
            if self.auto_submit:
                print(f"\n[6] Kaggle Submission...")
                submission_success = self._run_kaggle_submission()
                if submission_success:
                    print("[OK] Kaggle submission completed successfully")
                else:
                    print("[WARNING] Kaggle submission failed, but pipeline was successful")

            self._finalize_execution(success=True)

            print(f"\n[SUCCESS] KaggleSlayer Pipeline Complete!")
            print(f"Final CV Score: {self.current_execution.final_cv_score:.4f}")
            print(f"Best Model: {self.current_execution.best_model}")
            print(f"Duration: {self.current_execution.total_duration_seconds:.1f}s")
            print(f"Stages: {len([s for s in self.current_execution.stages if s.status == 'completed'])}/{len(self.current_execution.stages)} completed")

            return self.current_execution

        except Exception as e:
            print(f"\n[FAILED] Pipeline failed: {e}")
            import traceback
            traceback.print_exc()

            self.current_execution.error_summary.append(str(e))
            self._finalize_execution(success=False)
            return self.current_execution

    def _run_initial_pipeline(self) -> bool:
        """Run the initial pipeline stages"""
        stage_success = True

        # Stage 1: Competition Intelligence
        print("\n[1] Competition Intelligence Analysis...")
        stage_success &= self._run_stage("competition_intelligence", CompetitionReader)

        # Stage 2: Data Scout Analysis
        print("\n[2] Data Scout Analysis...")
        stage_success &= self._run_stage("data_scout", DataScout)

        # Stage 3: Feature Engineering
        print("\n[3] Feature Engineering...")
        stage_success &= self._run_stage("feature_engineering", FeatureEngineer)

        # Stage 4: Model Selection
        print("\n[4] Model Selection & Optimization...")
        stage_success &= self._run_stage("model_selection", ModelSelector)

        return stage_success

    def _run_iterative_improvement(self) -> bool:
        """Run iterative improvement cycles"""
        improvement_success = True

        for iteration in range(1, self.max_iterations + 1):
            print(f"\n[ITER] Iteration {iteration}/{self.max_iterations}")
            print("-" * 30)

            # Stage 5: Feedback Analysis
            print(f"\n[5] Feedback Analysis (Iteration {iteration})...")
            feedback_success = self._run_stage("feedback_analysis", FeedbackAnalyzer)

            if not feedback_success:
                print(f"Feedback analysis failed in iteration {iteration}")
                break

            # Load improvement recommendations
            recommendations = self._load_improvement_recommendations()

            if not recommendations or not recommendations.get("priority_actions"):
                print(f"No actionable recommendations found in iteration {iteration}")
                break

            # Apply improvements based on recommendations
            improvement_applied = self._apply_improvements(recommendations, iteration)

            if improvement_applied:
                # Re-run model selection to evaluate improvements
                print(f"\n[4] Model Re-evaluation (Iteration {iteration})...")
                model_success = self._run_stage(f"model_selection_iter_{iteration}", ModelSelector)

                if model_success:
                    # Check if performance improved
                    new_performance = self._extract_performance_metrics()
                    new_cv_score = new_performance.get("cv_score", 0.0)

                    improvement = new_cv_score - self.best_cv_score
                    self.improvement_history.append({
                        "iteration": iteration,
                        "old_score": self.best_cv_score,
                        "new_score": new_cv_score,
                        "improvement": improvement
                    })

                    print(f"[METRICS] Performance Change: {improvement:+.4f} ({self.best_cv_score:.4f} -> {new_cv_score:.4f})")

                    if new_cv_score > self.best_cv_score:
                        self.best_cv_score = new_cv_score
                        self.best_model = new_performance.get("best_model", self.best_model)
                        print(f"[SUCCESS] New best performance achieved!")
                    elif improvement < -0.001:  # Significant degradation
                        print(f"[WARNING] Performance degraded, stopping iterations")
                        break
                    elif abs(improvement) < 0.001:  # Minimal change
                        print(f"[DATA] Minimal improvement, considering convergence")
                        if iteration > 1:  # Stop after second iteration if no improvement
                            print(f"[CONVERGED] Converged - stopping iterations")
                            break

            self.current_execution.iteration_count = iteration

        return improvement_success

    def _run_stage(self, stage_name: str, agent_class) -> bool:
        """Run a single pipeline stage"""
        stage = PipelineStage(
            stage_name=stage_name,
            agent_class=agent_class.__name__,
            status="running",
            start_time=datetime.now().isoformat()
        )

        try:
            start_time = time.time()

            # Initialize and run agent
            if agent_class == CompetitionReader:
                agent = CompetitionReader(self.competition_path)
                result = agent.run_competition_analysis()
                stage.performance_metrics["confidence_score"] = result.confidence_score if result else 0.0

            elif agent_class == DataScout:
                agent = DataScout(self.competition_path, enable_llm=self.enable_llm)
                result = agent.run_full_analysis()
                stage.performance_metrics["dataset_rows"] = result[0].shape[0] if result else 0
                stage.performance_metrics["dataset_cols"] = result[0].shape[1] if result else 0

            elif agent_class == FeatureEngineer:
                agent = FeatureEngineer(self.competition_path, enable_llm=self.enable_llm)
                result = agent.run_feature_engineering()
                stage.performance_metrics["features_created"] = len(result[2].features_created) if result else 0
                stage.performance_metrics["total_features"] = result[2].total_features if result else 0

            elif agent_class == ModelSelector:
                agent = ModelSelector(self.competition_path, enable_llm=self.enable_llm)
                result = agent.run_model_selection()
                if result and result[0] and result[0].model_results:
                    stage.performance_metrics["cv_score"] = result[0].model_results[0].cv_mean
                    stage.performance_metrics["best_model"] = result[0].best_model
                    stage.performance_metrics["models_tested"] = result[0].total_models_tested
                else:
                    stage.performance_metrics["cv_score"] = 0.0
                    stage.performance_metrics["best_model"] = ""
                    stage.performance_metrics["models_tested"] = 0

            elif agent_class == FeedbackAnalyzer:
                agent = FeedbackAnalyzer(self.competition_path, enable_llm=self.enable_llm)
                result = agent.run_feedback_analysis()
                stage.performance_metrics["recommendations_count"] = len(result[1].priority_actions) if result else 0

            end_time = time.time()
            stage.duration_seconds = end_time - start_time
            stage.end_time = datetime.now().isoformat()
            stage.status = "completed"

            # Collect output files
            stage.output_files = self._collect_output_files(agent.output_dir if hasattr(agent, 'output_dir') else None)

            print(f"[OK] {stage_name} completed in {stage.duration_seconds:.1f}s")

            # Add stage to execution
            self.current_execution.stages.append(stage)
            return True

        except Exception as e:
            stage.status = "failed"
            stage.error_message = str(e)
            stage.end_time = datetime.now().isoformat()
            stage.duration_seconds = time.time() - start_time

            print(f"[ERROR] {stage_name} failed: {e}")
            self.current_execution.stages.append(stage)
            self.current_execution.error_summary.append(f"{stage_name}: {e}")
            return False

    def _apply_improvements(self, recommendations: Dict[str, Any], iteration: int) -> bool:
        """Apply improvement recommendations"""
        print(f"\n[LIST] Applying improvements for iteration {iteration}...")

        applied_count = 0
        priority_actions = recommendations.get("priority_actions", [])

        for i, action in enumerate(priority_actions[:3], 1):  # Apply top 3 actions
            print(f"  {i}. {action}")

            # Apply improvement based on action type
            try:
                if "feature" in action.lower():
                    # Re-run feature engineering with improvements
                    print(f"    → Re-running feature engineering...")
                    success = self._run_stage(f"feature_engineering_iter_{iteration}", FeatureEngineer)
                    if success:
                        applied_count += 1

                elif "model" in action.lower() or "hyperparameter" in action.lower():
                    # Model improvements will be handled in model selection re-run
                    print(f"    → Marked for model re-evaluation")
                    applied_count += 1

                elif "data" in action.lower():
                    # Re-run data preprocessing
                    print(f"    → Re-running data preprocessing...")
                    success = self._run_stage(f"data_scout_iter_{iteration}", DataScout)
                    if success:
                        applied_count += 1

                else:
                    print(f"    → Action type not yet implemented")

            except Exception as e:
                print(f"    [ERROR] Failed to apply improvement: {e}")
                continue

        print(f"[DATA] Applied {applied_count}/{len(priority_actions[:3])} improvements")
        return applied_count > 0

    def _extract_performance_metrics(self) -> Dict[str, Any]:
        """Extract current performance metrics"""
        metrics = {"cv_score": 0.0, "best_model": ""}

        try:
            # Load latest model selection results
            model_selection_path = self.competition_path / "model_selection" / "model_selection.json"
            if model_selection_path.exists():
                with open(model_selection_path, 'r') as f:
                    model_data = json.load(f)

                if model_data.get("model_results"):
                    best_result = model_data["model_results"][0]
                    metrics["cv_score"] = best_result.get("cv_mean", 0.0)
                    metrics["best_model"] = best_result.get("model_name", "")

        except Exception as e:
            print(f"Warning: Could not extract performance metrics: {e}")

        return metrics

    def _load_improvement_recommendations(self) -> Dict[str, Any]:
        """Load improvement recommendations from feedback analyzer"""
        try:
            recommendations_path = self.competition_path / "feedback_analysis" / "improvement_recommendations.json"
            if recommendations_path.exists():
                with open(recommendations_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load recommendations: {e}")

        return {}

    def _collect_output_files(self, output_dir: Optional[Path]) -> List[str]:
        """Collect output files from a stage"""
        if not output_dir or not output_dir.exists():
            return []

        try:
            return [str(f.relative_to(self.competition_path)) for f in output_dir.glob("*") if f.is_file()]
        except Exception:
            return []

    def _finalize_execution(self, success: bool):
        """Finalize pipeline execution"""
        self.current_execution.end_time = datetime.now().isoformat()
        self.current_execution.success = success

        # Calculate total duration
        if self.current_execution.start_time and self.current_execution.end_time:
            start = datetime.fromisoformat(self.current_execution.start_time)
            end = datetime.fromisoformat(self.current_execution.end_time)
            self.current_execution.total_duration_seconds = (end - start).total_seconds()

        # Save execution results
        self._save_execution_results()

        # Add to execution history
        self.execution_history.append(self.current_execution)

    def _save_execution_results(self):
        """Save pipeline execution results"""
        execution_path = self.output_dir / f"execution_{self.execution_id}.json"
        with open(execution_path, 'w') as f:
            json.dump(asdict(self.current_execution), f, indent=2, default=str)

        print(f"\n[FOLDER] Execution results saved: {execution_path}")

        # Save execution summary
        summary = {
            "execution_id": self.execution_id,
            "competition_name": self.competition_name,
            "success": self.current_execution.success,
            "final_cv_score": self.current_execution.final_cv_score,
            "best_model": self.current_execution.best_model,
            "iteration_count": self.current_execution.iteration_count,
            "total_duration_minutes": self.current_execution.total_duration_seconds / 60,
            "improvement_history": self.improvement_history,
            "timestamp": self.current_execution.end_time
        }

        summary_path = self.output_dir / "latest_execution_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"[LIST] Execution summary saved: {summary_path}")

    def get_execution_status(self) -> Dict[str, Any]:
        """Get current execution status"""
        if not self.current_execution:
            return {"status": "not_started"}

        return {
            "execution_id": self.current_execution.execution_id,
            "status": "running" if not self.current_execution.end_time else ("completed" if self.current_execution.success else "failed"),
            "current_cv_score": self.best_cv_score,
            "best_model": self.best_model,
            "iteration_count": self.current_execution.iteration_count,
            "stages_completed": len([s for s in self.current_execution.stages if s.status == "completed"]),
            "total_stages": len(self.stages),
            "duration_seconds": self.current_execution.total_duration_seconds
        }

    def _define_pipeline_stages(self) -> List[Dict[str, Any]]:
        """Define the pipeline stages configuration"""
        return [
            {"name": "competition_intelligence", "agent": "CompetitionReader", "required": True},
            {"name": "data_scout", "agent": "DataScout", "required": True},
            {"name": "feature_engineering", "agent": "FeatureEngineer", "required": False},
            {"name": "model_selection", "agent": "ModelSelector", "required": True},
            {"name": "feedback_analysis", "agent": "FeedbackAnalyzer", "required": False}
        ]

    def _load_configuration(self) -> Dict[str, Any]:
        """Load pipeline configuration"""
        default_config = {
            "pipeline": {
                "max_iterations": 3,
                "convergence_threshold": 0.001,
                "timeout_minutes": 120
            },
            "agents": {
                "enable_llm": True,
                "llm_temperature": 0.3,
                "parallel_execution": False
            }
        }

        if self.config_path.exists():
            try:
                import yaml
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")

        return default_config

    def _run_kaggle_submission(self) -> bool:
        """Run automatic Kaggle submission"""
        try:
            # Initialize submitter
            submitter = Submitter(self.competition_path)

            # Find predictions from model selection stage
            model_dir = self.competition_path / "model_selection"
            predictions_file = None

            # Look for predictions file
            for pred_file in ["predictions.csv", "test_predictions.csv", "submission.csv"]:
                pred_path = model_dir / pred_file
                if pred_path.exists():
                    predictions_file = pred_path
                    break

            # Also check baseline model directory
            if not predictions_file:
                baseline_dir = self.competition_path / "baseline_model"
                for pred_file in ["predictions.csv", "test_predictions.csv", "submission.csv"]:
                    pred_path = baseline_dir / pred_file
                    if pred_path.exists():
                        predictions_file = pred_path
                        break

            if not predictions_file:
                print("[ERROR] No predictions file found for submission")
                return False

            # Create submission message
            cv_score = self.current_execution.final_cv_score
            model_name = self.current_execution.best_model
            message = f"KaggleSlayer: {model_name} CV:{cv_score:.4f} LLM-Enhanced"

            print(f"Submitting predictions from: {predictions_file}")
            print(f"Submission message: {message}")

            # Run submission
            result = submitter.run_submission_pipeline(
                predictions_path=predictions_file,
                message=message,
                competition_ref=self.competition_name,
                dry_run=False
            )

            return result.submission_status == "submitted"

        except Exception as e:
            print(f"[ERROR] Kaggle submission failed: {e}")
            return False


def main():
    """Main entry point for Pipeline Coordinator"""
    parser = argparse.ArgumentParser(description="KaggleSlayer Autonomous Pipeline Coordinator")
    parser.add_argument("competition_path", type=Path,
                       help="Path to competition directory")
    parser.add_argument("--config", type=Path,
                       help="Path to configuration file")
    parser.add_argument("--no-llm", action="store_true",
                       help="Disable LLM integration")
    parser.add_argument("--no-autonomous", action="store_true",
                       help="Disable autonomous improvement iterations")
    parser.add_argument("--max-iterations", type=int, default=3,
                       help="Maximum number of improvement iterations")
    parser.add_argument("--no-submit", action="store_true",
                       help="Disable automatic Kaggle submission")
    parser.add_argument("--message", type=str, default="KaggleSlayer autonomous pipeline",
                       help="Submission message for Kaggle")

    args = parser.parse_args()

    # Validate competition path
    if not args.competition_path.exists():
        print(f"ERROR: Competition path does not exist: {args.competition_path}")
        return 1

    train_csv = args.competition_path / "train.csv"
    if not train_csv.exists():
        print(f"ERROR: Training data not found: {train_csv}")
        return 1

    if not AGENTS_AVAILABLE:
        print("ERROR: Agent modules not available")
        return 1

    try:
        # Initialize and run pipeline coordinator
        coordinator = PipelineCoordinator(
            competition_path=args.competition_path,
            config_path=args.config,
            enable_llm=not args.no_llm,
            autonomous_mode=not args.no_autonomous,
            max_iterations=args.max_iterations,
            auto_submit=not args.no_submit
        )

        execution = coordinator.run_full_pipeline()

        if execution.success:
            print(f"\n🎉 KaggleSlayer Pipeline Success!")
            print(f"📊 Final CV Score: {execution.final_cv_score:.4f}")
            print(f"🏆 Best Model: {execution.best_model}")
            print(f"🔄 Iterations: {execution.iteration_count}")
            print(f"⏱️ Total Time: {execution.total_duration_seconds/60:.1f} minutes")
            return 0
        else:
            print(f"\n💥 KaggleSlayer Pipeline Failed!")
            print(f"❌ Errors: {len(execution.error_summary)}")
            for error in execution.error_summary[:3]:
                print(f"   • {error}")
            return 1

    except Exception as e:
        print(f"ERROR: Pipeline coordination failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())