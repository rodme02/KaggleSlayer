# KaggleSlayer - Issue Backlog

## High Priority (P1) - Performance/Design Issues

### P1-1: KS Test Computed But Never Used
- **File**: `core/features/selectors.py:365-394`
- **Issue**: Kolmogorov-Smirnov test is calculated and stored but NOT used for filtering
- **Impact**: Wasted computation on every feature
- **Fix**: Either remove KS test computation or use it for filtering
- **Estimated Effort**: Small (30 min)

### P1-2: Redundant Problem Type Detection
- **Files**:
  - `core/features/utils.py:93-119` (canonical)
  - `core/models/evaluators.py:175-183`
  - `core/models/ensembles.py:350-356`
  - `core/features/selectors.py:141-143`
- **Issue**: Problem type detection logic duplicated across 4+ files
- **Impact**: Inconsistent detection, maintenance burden
- **Fix**: Import and use `auto_detect_problem_type()` from utils everywhere
- **Estimated Effort**: Medium (1-2 hours)

### P1-3: Unsafe Label Encoding for Test Data
- **File**: `core/features/transformers.py:93-97`
- **Issue**: Unseen categories in test data replaced with first class (`encoder.classes_[0]`)
- **Impact**: Arbitrary category assignment could harm model performance
- **Fix**: Use most frequent category from training or special "unknown" encoding
- **Estimated Effort**: Small (30 min)

### P1-4: Early Stopping Not Enforced
- **Files**:
  - `core/models/factory.py:327-398` (method exists)
  - `agents/model_selector.py` (not called)
- **Issue**: `fit_with_early_stopping()` exists but is never called
- **Impact**: Boosting models train for full iterations unnecessarily
- **Fix**: Integrate early stopping into model training pipeline
- **Estimated Effort**: Medium (1-2 hours)

### P1-5: Hardcoded Sample Sizes
- **File**: `agents/model_selector.py:56-91`
- **Issue**: Sample sizes (5000, 10000) are hardcoded
- **Impact**: Not configurable, may not be optimal for all datasets
- **Fix**: Add `sample_size_small` and `sample_size_large` to `config.yaml`
- **Estimated Effort**: Small (30 min)

### P1-6: Inconsistent Verbosity Handling
- **File**: `core/models/factory.py:312-325`
- **Issue**: Verbosity set in both default params (200-209) and `_apply_verbosity_defaults()`
- **Impact**: Potential conflicts/overrides
- **Fix**: Centralize verbosity handling in one place
- **Estimated Effort**: Small (30 min)

### P1-7: Missing Validation for Ensemble Input
- **File**: `core/models/ensembles.py:254-325`
- **Issue**: `select_best_models()` doesn't validate non-empty input
- **Impact**: Could crash with empty list
- **Fix**: Add validation check at start of method
- **Estimated Effort**: Trivial (5 min)

### P1-8: ID Column Detection Timing
- **File**: `core/features/selectors.py:320-321`
- **Issue**: ID columns auto-detected in `check_distribution_stability()` but should be earlier
- **Impact**: IDs might get processed in earlier feature generation steps
- **Fix**: Detect IDs in DataScout, pass to feature engineering
- **Estimated Effort**: Medium (1 hour)

### P1-9: Duplicate ID Detection Logic
- **File**: `agents/coordinator.py:305-321`
- **Issue**: `_detect_id_column()` only checks 'id' in name, doesn't use `detect_id_columns()` utility
- **Impact**: Might miss ID columns with different patterns
- **Fix**: Use comprehensive `detect_id_columns()` from utils
- **Estimated Effort**: Small (15 min)

---

## Low Priority (P3) - Nice to Have

### P3-1: Outlier Handling Only on Training
- **File**: `agents/data_scout.py:77-81`
- **Issue**: Outliers winsorized on train but not test
- **Impact**: Test data might have extreme values that break scaling
- **Consideration**: Should winsorization be applied to test using train statistics?
- **Estimated Effort**: Small (30 min)

### P3-2: Memory Usage Not Optimized
- **Files**: Multiple
- **Issue**: Large datasets loaded into memory multiple times
- **Impact**: High memory usage, potential OOM errors
- **Examples**:
  - `feature_engineer.py` loads cleaned, creates engineered
  - `model_selector.py` loads engineered again
- **Fix**: Consider chunking or memory-mapped arrays
- **Estimated Effort**: Large (4+ hours)

### P3-3: No Feature Importance Tracking
- **File**: `agents/model_selector.py`
- **Issue**: Final model's feature importances not saved in results
- **Impact**: Can't interpret which features were most valuable
- **Fix**: Extract and save feature importances from best model
- **Estimated Effort**: Small (30 min)

---

## Recently Fixed (Completed)

### ✅ P0-1: Ensemble Score Metric Conversion Consistency
- **Status**: FIXED
- **Files Modified**: `core/models/ensembles.py`
- **Change**: Added clarifying comments that metrics are already converted to "higher is better"
- **Impact**: Ensures consistent metric handling across ensemble methods

### ✅ P0-2: Test Feature Alignment After Selection
- **Status**: FIXED
- **Files Modified**: `agents/feature_engineer.py`
- **Change**: Calculate fill values from original data BEFORE selection, not after
- **Impact**: Prevents incorrect statistics when filling missing test features

### ✅ P0-3: Distribution Stability Test Data Alignment
- **Status**: FIXED
- **Files Modified**:
  - `core/features/selectors.py` (return type changed to Tuple)
  - `agents/feature_engineer.py` (updated caller)
- **Change**: `check_distribution_stability()` now returns both filtered train and test dataframes
- **Impact**: Ensures test data has same features as training data

---

## Priority Guide

- **P0 (Critical)**: Correctness bugs that cause failures or wrong results → **FIX IMMEDIATELY**
- **P1 (High)**: Performance issues, design flaws, maintainability problems → **Fix when optimizing**
- **P2 (Medium)**: Code quality, minor improvements → **Fix when refactoring**
- **P3 (Low)**: Nice-to-have enhancements → **Fix if time permits**

---

## Next Steps

1. Review P1 issues and prioritize based on current goals (accuracy vs robustness vs maintainability)
2. Tackle P1-7 (trivial validation fix) first for quick win
3. Consider P1-2 (unified problem type detection) for code cleanliness
4. Evaluate P1-4 (early stopping) if training time is a concern
5. Keep P3 issues in mind for future optimization sprints
