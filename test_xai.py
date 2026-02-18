"""Test script for XAI integration.

This script tests SHAP explainability, reason generation, and solution mapping.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ml_clo.data.mergers import create_training_dataset
from ml_clo.data.preprocessors import preprocess_exam_scores
from ml_clo.features.feature_builder import build_all_features
from ml_clo.models.ensemble_model import EnsembleModel
from ml_clo.reasoning.reason_generator import generate_complete_explanation
from ml_clo.xai.shap_explainer import EnsembleSHAPExplainer
from ml_clo.xai.shap_postprocess import process_shap_for_analysis


def prepare_training_data():
    """Prepare training dataset."""
    print("=" * 80)
    print("Preparing Training Dataset")
    print("=" * 80)

    # Load and preprocess exam scores
    from ml_clo.data.loaders import (
        load_assessment_methods,
        load_conduct_scores,
        load_demographics,
        load_exam_scores,
        load_study_hours,
        load_teaching_methods,
    )
    from ml_clo.data.encoders import encode_assessment_methods, encode_teaching_methods

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
        exam_history_df=exam_df,
        study_hours_df=study_df,
    )

    print(f"\nFinal training dataset: {len(training_df)} records, {len(training_df.columns)} columns")

    return training_df


def prepare_features_for_training(training_df):
    """Prepare features for training (encode categorical, handle NaN)."""
    # Select features (exclude target and ID columns)
    exclude_cols = ["exam_score", "Student_ID", "Subject_ID", "Lecturer_ID", "year"]
    feature_cols = [col for col in training_df.columns if col not in exclude_cols]
    X = training_df[feature_cols].copy()
    y = training_df["exam_score"].copy()

    # Encode categorical columns
    from sklearn.preprocessing import LabelEncoder
    label_encoders = {}

    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype.name == 'category':
            le = LabelEncoder()
            X[col] = X[col].fillna('Unknown')
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        elif X[col].dtype in [np.int64, np.float64]:
            X[col] = X[col].fillna(X[col].median())
        else:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                X[col] = X[col].fillna(X[col].median())
            except:
                le = LabelEncoder()
                X[col] = X[col].fillna('Unknown')
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le

    # Final check: fill any remaining NaN
    for col in X.columns:
        if X[col].isna().any():
            if X[col].dtype in [np.int64, np.float64]:
                fill_value = X[col].median() if X[col].notna().any() else 0
                X[col] = X[col].fillna(fill_value)
            else:
                X[col] = X[col].fillna(0)

    X = X.apply(pd.to_numeric, errors='coerce')
    X = X.fillna(0)

    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"Warning: {nan_count} NaN values still present, filling with 0")
        X = X.fillna(0)

    print(f"Selected {len(X.columns)} features for training")
    print(f"Encoded {len(label_encoders)} categorical columns")
    print(f"Final feature matrix: {X.shape}, NaN count: {X.isna().sum().sum()}")

    return X, y, label_encoders, feature_cols


def test_xai_integration():
    """Test XAI integration with trained model."""
    print("\n" + "=" * 80)
    print("Testing XAI Integration")
    print("=" * 80)

    # Prepare data
    training_df = prepare_training_data()
    X, y, label_encoders, feature_cols = prepare_features_for_training(training_df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    X_train_fit, X_val, y_train_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, shuffle=True
    )

    # Ensure no NaN
    for df_name, df in [("X_train_fit", X_train_fit), ("X_val", X_val), ("X_test", X_test)]:
        nan_count = df.isna().sum().sum()
        if nan_count > 0:
            df.fillna(0, inplace=True)

    print(f"\nData splits:")
    print(f"  Training: {len(X_train_fit)} records")
    print(f"  Validation: {len(X_val)} records")
    print(f"  Testing: {len(X_test)} records")

    # Train model
    print("\nTraining ensemble model...")
    model = EnsembleModel(random_state=42)
    model.train(
        X_train=X_train_fit,
        y_train=y_train_fit,
        X_val=X_val,
        y_val=y_val,
    )
    print("✓ Model trained successfully")

    # Initialize SHAP explainer
    print("\nInitializing SHAP explainer...")
    explainer = EnsembleSHAPExplainer(model, cache_explainer=True)
    print("✓ SHAP explainer initialized")

    # Test individual prediction explanation
    print("\n" + "-" * 80)
    print("Testing Individual Prediction Explanation")
    print("-" * 80)

    # Select a few test instances
    test_indices = [0, 10, 50]  # Test with first few instances
    for idx in test_indices:
        if idx >= len(X_test):
            continue

        X_instance = X_test.iloc[[idx]]
        y_actual = y_test.iloc[idx]
        y_pred = model.predict(X_instance)[0]

        print(f"\nInstance {idx}:")
        print(f"  Actual score: {y_actual:.2f}")
        print(f"  Predicted score: {y_pred:.2f}")

        # Compute SHAP values
        shap_values = explainer.explain_instance(X_instance)
        shap_values_1d = shap_values[0]  # Get first (and only) instance

        # Debug: Print some SHAP values
        print(f"  SHAP values shape: {shap_values_1d.shape}")
        print(f"  Sample SHAP values (first 5): {shap_values_1d[:5]}")
        print(f"  Sample feature names (first 5): {feature_cols[:5]}")

        # Process SHAP for analysis
        processed = process_shap_for_analysis(
            shap_values_1d,
            feature_names=feature_cols,
            df=None,  # We don't have original df here
        )
        
        # Debug: Print grouped SHAP
        print(f"  Grouped SHAP: {list(processed['grouped_shap'].keys())}")
        print(f"  Top negative impacts: {len(processed['top_negative_impacts'])} items")
        if processed['top_negative_impacts']:
            print(f"    First impact: {processed['top_negative_impacts'][0]}")

        # Generate explanation
        explanation = generate_complete_explanation(
            top_negative_impacts=processed["top_negative_impacts"],
            predicted_score=y_pred,
            context="individual",
            include_solutions=True,
        )

        print(f"\n  Explanation Summary:")
        print(f"    {explanation['summary']}")
        print(f"\n  Top Reasons:")
        for i, reason in enumerate(explanation['reasons'][:3], 1):
            print(f"    {i}. {reason['group_name']}: {reason['reason_text']}")
            if reason.get('solutions'):
                print(f"       Solutions:")
                for sol in reason['solutions'][:2]:
                    print(f"         - {sol}")

    # Test batch explanation
    print("\n" + "-" * 80)
    print("Testing Batch Explanation")
    print("-" * 80)

    X_batch = X_test.iloc[:10]
    shap_values_batch = explainer.explain_batch(X_batch)
    print(f"✓ Computed SHAP values for {len(X_batch)} instances")
    print(f"  SHAP shape: {shap_values_batch.shape}")

    # Test feature importance
    print("\n" + "-" * 80)
    print("Testing Feature Importance")
    print("-" * 80)

    importance = explainer.get_feature_importance(X_test.iloc[:100])
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("Top 10 features by SHAP importance:")
    for i, (feature, imp) in enumerate(top_features, 1):
        print(f"  {i}. {feature}: {imp:.4f}")

    # Test class-level analysis (aggregate)
    print("\n" + "-" * 80)
    print("Testing Class-Level Analysis")
    print("-" * 80)

    # Compute SHAP for all test instances
    shap_values_all = explainer.explain_batch(X_test.iloc[:100])
    
    # Aggregate SHAP values
    from ml_clo.xai.shap_postprocess import aggregate_class_shap
    aggregated_shap = aggregate_class_shap(
        [shap_values_all[i] for i in range(len(shap_values_all))],
        feature_names=feature_cols,
    )

    # Process aggregated SHAP
    processed_class = process_shap_for_analysis(
        aggregated_shap,
        feature_names=feature_cols,
        df=None,
    )

    # Generate class-level explanation
    class_explanation = generate_complete_explanation(
        top_negative_impacts=processed_class["top_negative_impacts"],
        predicted_score=np.mean(model.predict(X_test.iloc[:100])),
        context="class",
        include_solutions=True,
    )

    print(f"\nClass-level Explanation Summary:")
    print(f"  {class_explanation['summary']}")
    print(f"\n  Top Reasons for Class:")
    for i, reason in enumerate(class_explanation['reasons'][:3], 1):
        print(f"    {i}. {reason['group_name']}: {reason['reason_text']}")
        if reason.get('solutions'):
            print(f"       Solutions:")
            for sol in reason['solutions'][:2]:
                print(f"         - {sol}")

    print("\n" + "=" * 80)
    print("✓ XAI Integration Test PASSED")
    print("=" * 80)

    return True


def main():
    """Main test function."""
    print("=" * 80)
    print("TESTING XAI INTEGRATION")
    print("=" * 80)

    try:
        test_xai_integration()
        print("\n✓ All XAI tests passed!")
        return 0
    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

