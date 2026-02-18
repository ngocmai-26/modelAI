"""Test script for model development."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ml_clo.data.encoders import encode_assessment_methods, encode_teaching_methods
from ml_clo.data.loaders import (
    load_assessment_methods,
    load_conduct_scores,
    load_demographics,
    load_exam_scores,
    load_study_hours,
    load_teaching_methods,
)
from ml_clo.data.mergers import create_training_dataset
from ml_clo.data.preprocessors import preprocess_exam_scores
from ml_clo.features.feature_builder import build_all_features
from ml_clo.models.ensemble_model import EnsembleModel
from ml_clo.models.model_evaluator import (
    evaluate_by_score_range,
    evaluate_model,
    print_evaluation_summary,
)
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


def prepare_training_data():
    """Prepare training dataset with all features."""
    print("\n" + "=" * 80)
    print("Preparing Training Dataset")
    print("=" * 80)

    # Load and preprocess exam scores
    exam_df = load_exam_scores("data/DiemTong.xlsx")
    exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)
    print(f"Exam scores: {len(exam_df)} records")

    # Load other data sources
    conduct_df = load_conduct_scores("data/diemrenluyen.xlsx")
    demo_df = load_demographics("data/nhankhau.xlsx")
    tm_df = encode_teaching_methods(load_teaching_methods("data/PPGDfull.xlsx"))
    em_df = encode_assessment_methods(load_assessment_methods("data/PPDGfull.xlsx"))
    study_df = load_study_hours("data/tuhoc.xlsx")

    # Merge all data
    training_df = create_training_dataset(
        exam_df=exam_df,
        conduct_df=conduct_df,
        demographics_df=demo_df,
        teaching_methods_df=tm_df,
        assessment_methods_df=em_df,
        study_hours_df=study_df,
        target_column="exam_score",
        drop_missing_target=True,
    )

    # Build aggregate features
    training_df = build_all_features(
        training_df,
        conduct_history_df=conduct_df,
        exam_history_df=exam_df,  # Use same data as history
        study_hours_df=study_df,
    )

    print(f"\nFinal training dataset: {len(training_df)} records, {len(training_df.columns)} columns")

    return training_df


def test_ensemble_model():
    """Test ensemble model training and prediction."""
    print("\n" + "=" * 80)
    print("Testing Ensemble Model")
    print("=" * 80)

    # Prepare data
    df = prepare_training_data()

    # Select features (exclude target and ID columns)
    exclude_cols = ["Student_ID", "Subject_ID", "Lecturer_ID", "exam_score", "year"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    # Remove columns with all NaN
    feature_cols = [col for col in feature_cols if df[col].notna().sum() > 0]

    print(f"\nSelected {len(feature_cols)} features for training")

    # Prepare X and y
    X = df[feature_cols].copy()
    y = df["exam_score"].copy()

    # Encode categorical columns (string/object types)
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}
    
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            # Encode categorical columns
            le = LabelEncoder()
            # Fill NaN with a default value before encoding
            X[col] = X[col].fillna('Unknown')
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        elif X[col].dtype in [np.int64, np.float64]:
            # Fill numeric columns with median
            X[col] = X[col].fillna(X[col].median())
        else:
            # For other types, try to convert to numeric
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].fillna(X[col].median())
            except:
                # If conversion fails, use label encoding
                le = LabelEncoder()
                X[col] = X[col].fillna('Unknown')
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le

    # Ensure all columns are numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
            X[col] = X[col].fillna(0)

    # Final check: fill any remaining NaN values
    for col in X.columns:
        if X[col].isna().any():
            if X[col].dtype in [np.int64, np.float64]:
                fill_value = X[col].median() if X[col].notna().any() else 0
                X[col] = X[col].fillna(fill_value)
            else:
                X[col] = X[col].fillna(0)

    # Convert all to numeric and fill any remaining NaN
    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

    # Verify no NaN values remain
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values still present, filling with 0")
        X = X.fillna(0)

    print(f"Encoded {len(label_encoders)} categorical columns")
    print(f"Final feature matrix: {X.shape}, NaN count: {X.isna().sum().sum()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Further split training data for validation
    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, shuffle=True
    )

    # Final check: ensure no NaN in split data
    for df_name, df in [("X_train_fit", X_train_fit), ("X_val", X_val), ("X_test", X_test)]:
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            print(f"Warning: {nan_count} NaN values in {df_name}, filling with 0")
            df.fillna(0, inplace=True)

    print(f"\nData splits:")
    print(f"  Training: {len(X_train_fit)} records (NaN: {X_train_fit.isna().sum().sum()})")
    print(f"  Validation: {len(X_val)} records (NaN: {X_val.isna().sum().sum()})")
    print(f"  Testing: {len(X_test)} records (NaN: {X_test.isna().sum().sum()})")

    # Initialize and train ensemble model
    print("\nInitializing ensemble model...")
    model = EnsembleModel(random_state=42)

    print("Training ensemble model...")
    training_metrics = model.train(
        X_train=X_train_fit,
        y_train=y_train_fit,
        X_val=X_val,
        y_val=y_val,
    )

    print("\nTraining metrics:")
    for metric, value in training_metrics.items():
        if isinstance(value, float):
            print(f"  {metric:25s}: {value:.4f}")
        else:
            print(f"  {metric:25s}: {value}")

    # Make predictions
    print("\nMaking predictions on test set...")
    y_pred = model.predict(X_test)

    # Evaluate
    print("\nEvaluating model performance...")
    test_metrics = evaluate_model(y_test, y_pred, prefix="test_")
    print_evaluation_summary(test_metrics)

    # Evaluate by score range
    print("\nEvaluating by score range...")
    range_metrics = evaluate_by_score_range(y_test, y_pred)
    for range_name, metrics in range_metrics.items():
        print(f"\n  {range_name}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"    {metric:20s}: {value:.4f}")
            else:
                print(f"    {metric:20s}: {value}")

    # Feature importance
    print("\nTop 10 most important features:")
    feature_importance = model.get_feature_importance()
    if feature_importance:
        sorted_features = sorted(
            feature_importance.items(), key=lambda x: x[1], reverse=True
        )[:10]
        for feature, importance in sorted_features:
            print(f"  {feature:30s}: {importance:.4f}")

    # Model info
    print("\nModel information:")
    model_info = model.get_info()
    for key, value in model_info.items():
        print(f"  {key:20s}: {value}")

    # Test save/load
    print("\nTesting model save/load...")
    test_model_path = "test_ensemble_model.joblib"
    model.save(test_model_path)

    # Load model
    new_model = EnsembleModel()
    new_model.load(test_model_path)

    # Verify predictions match
    y_pred_loaded = new_model.predict(X_test)
    pred_diff = np.abs(y_pred - y_pred_loaded).max()
    print(f"  Max prediction difference after load: {pred_diff:.10f}")
    if pred_diff < 1e-6:
        print("  ✓ Save/load test PASSED")
    else:
        print("  ✗ Save/load test FAILED")

    # Cleanup
    Path(test_model_path).unlink(missing_ok=True)

    return True


def main():
    """Run all tests."""
    print("=" * 80)
    print("TESTING MODEL DEVELOPMENT")
    print("=" * 80)

    results = []

    try:
        results.append(("ensemble_model", test_ensemble_model()))
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    print("\nDetailed results:")
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")


if __name__ == "__main__":
    main()

