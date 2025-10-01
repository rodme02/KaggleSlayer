#!/usr/bin/env python3
"""
Streamlined KaggleSlayer Dashboard using the refactored architecture.
"""

import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents import PipelineCoordinator
from utils.config import ConfigManager


def main():
    st.set_page_config(
        page_title="KaggleSlayer - Autonomous ML Pipeline",
        page_icon="🎯",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎯 KaggleSlayer - Autonomous ML Pipeline")
    st.markdown("**Clean, Modular, and Maintainable Kaggle Competition Framework**")

    # Sidebar configuration
    st.sidebar.header("Pipeline Configuration")

    competition_name = st.sidebar.text_input("Competition Name", value="titanic")
    competition_path = st.sidebar.text_input("Competition Data Path", value="competition_data/titanic")

    # Pipeline controls
    st.sidebar.header("Pipeline Steps")
    skip_data_scout = st.sidebar.checkbox("Skip Data Scout")
    skip_feature_engineer = st.sidebar.checkbox("Skip Feature Engineer")
    skip_model_selector = st.sidebar.checkbox("Skip Model Selector")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Pipeline Execution")

        if st.button("🚀 Run Complete Pipeline", type="primary"):
            if not Path(competition_path).exists():
                st.error(f"Competition path does not exist: {competition_path}")
                return

            try:
                config = ConfigManager()
                coordinator = PipelineCoordinator(competition_name, Path(competition_path), config)

                skip_steps = []
                if skip_data_scout:
                    skip_steps.append("data_scout")
                if skip_feature_engineer:
                    skip_steps.append("feature_engineer")
                if skip_model_selector:
                    skip_steps.append("model_selector")

                with st.spinner("Running pipeline..."):
                    results = coordinator.run(skip_steps=skip_steps)

                st.success("Pipeline completed successfully!")

                # Display results
                st.subheader("Pipeline Results")
                st.json(results)

                # Key metrics
                if "final_score" in results:
                    st.metric("Final Score", f"{results['final_score']:.4f}")
                if "best_model" in results:
                    st.metric("Best Model", results['best_model'])

            except Exception as e:
                st.error(f"Pipeline failed: {e}")

    with col2:
        st.header("Pipeline Status")

        # Check if results exist
        results_path = Path(competition_path) / "pipeline_results.json"
        if results_path.exists():
            import json
            with open(results_path) as f:
                results = json.load(f)

            st.success("Previous results found")
            st.write(f"**Status**: {results.get('pipeline_status', 'Unknown')}")
            st.write(f"**Steps Completed**: {len(results.get('steps_completed', []))}")

            if "final_score" in results:
                st.metric("Last Score", f"{results['final_score']:.4f}")
        else:
            st.info("No previous results found")

    # Architecture overview
    st.header("🏗️ New Modular Architecture")

    architecture_col1, architecture_col2 = st.columns(2)

    with architecture_col1:
        st.subheader("Core Components")
        st.markdown("""
        - **Data**: Loading, preprocessing, validation
        - **Features**: Generation, selection, transformation
        - **Models**: Factory, evaluation, optimization
        - **Analysis**: Performance analysis, insights
        """)

    with architecture_col2:
        st.subheader("Benefits")
        st.markdown("""
        - ✅ **Modular**: Single-responsibility components
        - ✅ **Maintainable**: Easy to understand and modify
        - ✅ **Extensible**: Simple to add new algorithms
        - ✅ **Testable**: Components can be tested independently
        """)


if __name__ == "__main__":
    main()