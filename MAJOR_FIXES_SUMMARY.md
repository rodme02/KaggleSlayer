# Major Fixes to KaggleSlayer Pipeline

## Date: 2025-10-01

## Critical Issues Identified and Fixed

### Issue #1: Ensemble Models Not Fitted Before Saving
**Problem:** Voting, stacking, and weighted ensemble models were evaluated via cross-validation but never fitted on the full training dataset before being saved. This caused models to be in an unfitted state, resulting in failed predictions (all predictions = 0).

**Root Cause:** In `core/models/ensembles.py`, the ensemble creation methods only called `_cross_validate()` to evaluate the ensemble but didn't call `.fit()` on the full training data.

**Fix:** Added `ensemble.fit(X_train, y_train)` calls after evaluation in:
- `create_voting_ensemble()` (line 82)
- `create_stacking_ensemble()` (line 198)
- `create_weighted_ensemble()` (line 145)

**Impact:** Models now make proper predictions instead of returning all zeros.

---

### Issue #2: Original Categorical Columns Causing Overfitting
**Problem:** High-cardinality categorical columns (Name, Ticket, Cabin) were:
1. Kept in the dataset after extracting derived features
2. Label-encoded (Name with 891 unique values → labels 0-890)
3. Scaled as if they were numerical features

This caused catastrophic overfitting:
- Train: 891 unique Name values for 891 samples = perfect memorization
- Test: Only 3 unique Name values after scaling transformations
- CV Score: 0.846 vs Kaggle Score: 0.605 (0.241 gap!)

**Root Cause:** In `core/features/generators.py`, the `generate_categorical_features()` method created derived features (Name_len, Sex_freq, Cabin_rarity, etc.) but kept the original columns. These were then label-encoded and scaled in the transformation pipeline, destroying their categorical meaning.

**Fix:** Added logic to drop original categorical columns after extracting derived features in `core/features/generators.py:200-201`:
```python
# DROP original categorical columns after extracting features
print(f"Dropping {len(categorical_cols)} original categorical columns after feature extraction...")
df_engineered = df_engineered.drop(columns=categorical_cols, errors='ignore')
```

**Impact:**
- Removed problematic high-cardinality features that caused overfitting
- Kept only meaningful derived features (frequency, length, rarity)
- Reduced feature count from 50 to 44 in Titanic (removed 5 categorical + 1 redundant)
- Improved prediction distribution from 65% survival to 51% (much closer to training 38%)
- Should significantly reduce CV/Kaggle score gap

---

### Issue #3: Frequency Encoding Train/Test Mismatch
**Problem:** Frequency encoding was computed separately on training and test data, causing:
- "male" frequency = 0.738 in train (577/891)
- "male" frequency = 0.658 in test (266/418)
- After scaling, these became completely different feature values
- Model learned patterns that didn't transfer to test set

**Root Cause:** In `core/features/generators.py:179`, frequency encoding was computed using `df[col].value_counts()` on whatever dataset was passed in, without storing/reusing encodings from training.

**Fix:**
1. Added `frequency_encodings` dictionary to `FeatureGenerator` class to store training frequencies
2. Modified `generate_categorical_features()` to accept `fit` parameter (line 151)
3. When `fit=True` (training), compute and store frequency encodings
4. When `fit=False` (test), use stored encodings from training
5. Updated `feature_engineer.py` to pass `fit=True` for train, `fit=False` for test

**Impact:**
- Frequency encodings now consistent between train and test
- Test data uses exact same frequency values as training ("male"=0.738 in both)
- Eliminates major source of train/test distribution mismatch
- Should dramatically improve Kaggle scores

---

## Expected Improvements

### Titanic Competition:
- **Before Fix:** CV=0.846, Kaggle=0.605, 65% predicted survival
- **After Fix:** CV=0.845, Kaggle=TBD (should be 0.75-0.80), 51% predicted survival
- **Key Improvement:** Eliminated overfitting from high-cardinality categoricals

### General Improvements:
1. **Better Generalization:** Models can't memorize individual names/tickets
2. **More Realistic Predictions:** Distribution closer to training data
3. **Reduced Feature Count:** Removed redundant/harmful features
4. **Cleaner Pipeline:** Only numerical features go through scaling

---

## Files Modified

1. **core/models/ensembles.py**
   - Added `ensemble.fit(X_train, y_train)` in 3 methods
   - Lines: 82, 145, 198

2. **core/features/generators.py**
   - Added categorical column dropping after feature extraction (lines 197-201)
   - Added `frequency_encodings` dict to store training frequencies (line 23)
   - Modified `generate_categorical_features()` to support fit/transform pattern (lines 151-197)
   - Frequency encodings now computed once on training, reused for test

3. **agents/feature_engineer.py**
   - Updated to pass `fit=True` for training data (line 51)
   - Updated to pass `fit=False` for test data (line 57)

---

## Testing Checklist

- [x] Titanic: Pipeline runs successfully
- [x] Titanic: Predictions are diverse (not all 0s)
- [x] Titanic: No categorical columns in final engineered data
- [x] Titanic: Prediction distribution more reasonable
- [ ] Digit-Recognizer: Test pipeline works
- [ ] Playground-Series: Test pipeline works
- [ ] Submit to Kaggle and verify score improves

---

## Next Steps

1. Test the fix on digit-recognizer and playground-series-s5e9
2. Submit Titanic to Kaggle to verify actual score improvement
3. Consider additional improvements:
   - Better hyperparameter tuning (currently using simple grid search)
   - More sophisticated feature engineering
   - Better handling of class imbalance
   - Ensemble diversity improvements
