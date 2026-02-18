"""Prediction pipeline for individual student CLO prediction.

This module provides a complete prediction pipeline that integrates:
- Model loading
- Data loading and preprocessing for a single student
- Feature engineering
- CLO score prediction
- SHAP explainability
- Reason generation and solution mapping
- Structured output
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
from ml_clo.outputs.schemas import IndividualAnalysisOutput
from ml_clo.reasoning.reason_generator import generate_complete_explanation
from ml_clo.utils.exceptions import ModelLoadError
from ml_clo.utils.logger import get_logger
from ml_clo.xai.shap_explainer import EnsembleSHAPExplainer
from ml_clo.xai.shap_postprocess import process_shap_for_analysis

logger = get_logger(__name__)


class PredictionPipeline:
    """Prediction pipeline for individual student CLO prediction."""

    def __init__(self, model_path: str):
        """Initialize prediction pipeline.

        Args:
            model_path: Path to trained model file

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ModelLoadError(f"Model file not found: {model_path}")

        self.model: Optional[EnsembleModel] = None
        self.explainer: Optional[EnsembleSHAPExplainer] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: Optional[list] = None

    def load_model(self) -> EnsembleModel:
        """Load trained model.

        Returns:
            Loaded EnsembleModel

        Raises:
            ModelLoadError: If model cannot be loaded
        """
        logger.info(f"Loading model from {self.model_path}")

        self.model = EnsembleModel(random_state=42)
        self.model.load(str(self.model_path))

        if not self.model.is_trained:
            raise ModelLoadError("Model is not trained")

        self.feature_names = self.model.feature_names

        logger.info(
            f"Model loaded: {self.model.model_name} (version {self.model.version})"
        )

        return self.model

    def load_student_data(
        self,
        student_id: str,
        subject_id: str,
        lecturer_id: str,
        exam_scores_path: str,
        conduct_scores_path: Optional[str] = None,
        demographics_path: Optional[str] = None,
        teaching_methods_path: Optional[str] = None,
        assessment_methods_path: Optional[str] = None,
        study_hours_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load and prepare data for a single student.

        Args:
            student_id: Student ID
            subject_id: Subject ID
            lecturer_id: Lecturer ID
            exam_scores_path: Path to exam scores file
            conduct_scores_path: Path to conduct scores file (optional)
            demographics_path: Path to demographics file (optional)
            teaching_methods_path: Path to teaching methods file (optional)
            assessment_methods_path: Path to assessment methods file (optional)
            study_hours_path: Path to study hours file (optional)

        Returns:
            DataFrame with student data ready for prediction
        """
        logger.info(f"Loading data for student {student_id}, subject {subject_id}")

        # Load all data sources (same as training)
        # Imports are already at top of file

        exam_df = load_exam_scores(exam_scores_path)
        exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)

        # Filter for specific student, subject, lecturer
        student_data = exam_df[
            (exam_df["Student_ID"] == student_id)
            & (exam_df["Subject_ID"] == subject_id)
            & (exam_df["Lecturer_ID"] == lecturer_id)
        ].copy()

        if len(student_data) == 0:
            raise ValueError(
                f"No data found for student_id={student_id}, "
                f"subject_id={subject_id}, lecturer_id={lecturer_id}"
            )

        # Load other data sources
        data = {"exam_scores": student_data}

        if conduct_scores_path and Path(conduct_scores_path).exists():
            data["conduct_scores"] = load_conduct_scores(conduct_scores_path)

        if demographics_path and Path(demographics_path).exists():
            data["demographics"] = load_demographics(demographics_path)

        if teaching_methods_path and Path(teaching_methods_path).exists():
            tm_df = load_teaching_methods(teaching_methods_path)
            data["teaching_methods"] = encode_teaching_methods(tm_df)

        if assessment_methods_path and Path(assessment_methods_path).exists():
            em_df = load_assessment_methods(assessment_methods_path)
            data["assessment_methods"] = encode_assessment_methods(em_df)

        if study_hours_path and Path(study_hours_path).exists():
            data["study_hours"] = load_study_hours(study_hours_path)

        # Merge and build features (same as training)
        training_df = create_training_dataset(
            exam_df=student_data,
            conduct_df=data.get("conduct_scores"),
            demographics_df=data.get("demographics"),
            teaching_methods_df=data.get("teaching_methods"),
            assessment_methods_df=data.get("assessment_methods"),
            study_hours_df=data.get("study_hours"),
            target_column="exam_score",
            drop_missing_target=False,  # Don't drop, we need to predict
        )

        # Build aggregate features (need full history for features)
        full_exam_df = load_exam_scores(exam_scores_path)
        full_exam_df = preprocess_exam_scores(full_exam_df, convert_to_clo=True, create_result=False)

        training_df = build_all_features(
            training_df,
            conduct_history_df=data.get("conduct_scores"),
            exam_history_df=full_exam_df,  # Use full history for features
            study_hours_df=data.get("study_hours"),
        )

        logger.info(f"Student data prepared: {len(training_df)} records")

        return training_df

    def prepare_features(
        self,
        student_df: pd.DataFrame,
        label_encoders: Optional[Dict[str, LabelEncoder]] = None,
    ) -> pd.DataFrame:
        """Prepare features for prediction.

        Args:
            student_df: Student data DataFrame
            label_encoders: Label encoders from training (optional, will create new if None)

        Returns:
            Feature matrix ready for prediction
        """
        logger.info("Preparing features for prediction")

        # Select features (exclude target and ID columns)
        exclude_cols = [
            "Student_ID",
            "Subject_ID",
            "Lecturer_ID",
            "exam_score",
            "year",
        ]
        feature_cols = [col for col in student_df.columns if col not in exclude_cols]

        # Remove columns with all NaN
        feature_cols = [col for col in feature_cols if student_df[col].notna().sum() > 0]

        # Ensure feature order matches training
        if self.feature_names:
            # Use only features that exist in both
            feature_cols = [col for col in self.feature_names if col in feature_cols]
            # Add missing features with zeros
            missing_features = [col for col in self.feature_names if col not in feature_cols]
            for feat in missing_features:
                student_df[feat] = 0
                feature_cols.append(feat)

        X = student_df[feature_cols].copy()

        # Encode categorical columns (use training encoders if available)
        if label_encoders is None:
            label_encoders = {}

        for col in X.columns:
            if X[col].dtype == "object" or X[col].dtype.name == "category":
                if col in label_encoders:
                    le = label_encoders[col]
                    X[col] = X[col].fillna("Unknown")
                    X[col] = le.transform(X[col].astype(str))
                else:
                    # Create new encoder if not found
                    le = LabelEncoder()
                    X[col] = X[col].fillna("Unknown")
                    X[col] = le.fit_transform(X[col].astype(str))
                    label_encoders[col] = le
            elif X[col].dtype in [np.int64, np.float64]:
                X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)
            else:
                try:
                    X[col] = pd.to_numeric(X[col], errors="coerce")
                    X[col] = X[col].fillna(X[col].median() if X[col].notna().any() else 0)
                except Exception:
                    if col in label_encoders:
                        le = label_encoders[col]
                    else:
                        le = LabelEncoder()
                        label_encoders[col] = le
                    X[col] = X[col].fillna("Unknown")
                    X[col] = le.transform(X[col].astype(str))

        # Final check: fill any remaining NaN
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(0)

        # Ensure feature order matches model
        if self.feature_names:
            X = X[self.feature_names]

        logger.info(f"Features prepared: {X.shape}")

        return X

    def predict(
        self,
        student_id: str,
        subject_id: str,
        lecturer_id: str,
        exam_scores_path: str,
        conduct_scores_path: Optional[str] = None,
        demographics_path: Optional[str] = None,
        teaching_methods_path: Optional[str] = None,
        assessment_methods_path: Optional[str] = None,
        study_hours_path: Optional[str] = None,
    ) -> IndividualAnalysisOutput:
        """Predict CLO score and generate explanation for a student.

        Args:
            student_id: Student ID
            subject_id: Subject ID
            lecturer_id: Lecturer ID
            exam_scores_path: Path to exam scores file
            conduct_scores_path: Path to conduct scores file (optional)
            demographics_path: Path to demographics file (optional)
            teaching_methods_path: Path to teaching methods file (optional)
            assessment_methods_path: Path to assessment methods file (optional)
            study_hours_path: Path to study hours file (optional)

        Returns:
            IndividualAnalysisOutput with prediction and explanation
        """
        logger.info(f"Predicting CLO score for student {student_id}")

        # Load model if not already loaded
        if self.model is None:
            self.load_model()

        # Initialize explainer if not already initialized
        if self.explainer is None:
            self.explainer = EnsembleSHAPExplainer(self.model, cache_explainer=True)

        # Load student data
        student_df = self.load_student_data(
            student_id=student_id,
            subject_id=subject_id,
            lecturer_id=lecturer_id,
            exam_scores_path=exam_scores_path,
            conduct_scores_path=conduct_scores_path,
            demographics_path=demographics_path,
            teaching_methods_path=teaching_methods_path,
            assessment_methods_path=assessment_methods_path,
            study_hours_path=study_hours_path,
        )

        # Prepare features
        X = self.prepare_features(student_df)

        # Predict
        predicted_score = self.model.predict(X)[0]

        # Compute SHAP values
        shap_values = self.explainer.explain_instance(X)
        shap_values_1d = shap_values[0]

        # Process SHAP for analysis
        processed = process_shap_for_analysis(
            shap_values_1d,
            feature_names=self.feature_names,
            df=None,
        )

        # Generate explanation
        explanation = generate_complete_explanation(
            top_negative_impacts=processed["top_negative_impacts"],
            predicted_score=predicted_score,
            context="individual",
            include_solutions=True,
        )

        # Convert to output schema
        output = IndividualAnalysisOutput.from_explanation_dict(
            explanation,
            student_id=student_id,
            subject_id=subject_id,
            lecturer_id=lecturer_id,
        )

        logger.info(
            f"Prediction complete: score={predicted_score:.2f}, "
            f"reasons={len(output.reasons)}"
        )

        return output

