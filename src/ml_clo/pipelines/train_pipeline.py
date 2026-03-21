"""Training pipeline for CLO prediction model.

This module provides a complete training pipeline that integrates:
- Data loading and preprocessing
- Feature engineering
- Model training (Random Forest + Gradient Boosting Ensemble)
- Model evaluation
- Model saving with versioning
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ml_clo.config.model_config import TRAINING_CONFIG
from ml_clo.data.encoders import encode_assessment_methods, encode_teaching_methods
from ml_clo.data.loaders import (
    load_assessment_methods,
    load_attendance,
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
from ml_clo.utils.exceptions import DataLoadError, DataValidationError
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """Training pipeline for CLO prediction model."""

    def __init__(
        self,
        random_state: int = 42,
        test_size: float = 0.2,
        validation_size: float = 0.2,
    ):
        """Initialize training pipeline.

        Args:
            random_state: Random seed for reproducibility (default: 42)
            test_size: Proportion of data for testing (default: 0.2)
            validation_size: Proportion of training data for validation (default: 0.2)
        """
        self.random_state = random_state
        self.test_size = test_size
        self.validation_size = validation_size
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: Optional[list] = None

    def load_data(
        self,
        exam_scores_path: str,
        conduct_scores_path: Optional[str] = None,
        demographics_path: Optional[str] = None,
        teaching_methods_path: Optional[str] = None,
        assessment_methods_path: Optional[str] = None,
        study_hours_path: Optional[str] = None,
        attendance_path: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load all data sources.

        Args:
            exam_scores_path: Path to exam scores file (required)
            conduct_scores_path: Path to conduct scores file (optional)
            demographics_path: Path to demographics file (optional)
            teaching_methods_path: Path to teaching methods file (optional)
            assessment_methods_path: Path to assessment methods file (optional)
            study_hours_path: Path to study hours file (optional)

        Returns:
            Dictionary mapping data source names to DataFrames

        Raises:
            DataLoadError: If required files cannot be loaded
        """
        logger.info("Loading data sources")

        data = {}

        # Load exam scores (required)
        if not Path(exam_scores_path).exists():
            raise DataLoadError(f"Exam scores file not found: {exam_scores_path}")

        exam_df = load_exam_scores(exam_scores_path)
        exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)
        data["exam_scores"] = exam_df
        logger.info(f"Loaded exam scores: {len(exam_df)} records")

        # Load optional data sources
        if conduct_scores_path and Path(conduct_scores_path).exists():
            data["conduct_scores"] = load_conduct_scores(conduct_scores_path)
            logger.info(f"Loaded conduct scores: {len(data['conduct_scores'])} records")

        if demographics_path and Path(demographics_path).exists():
            data["demographics"] = load_demographics(demographics_path)
            logger.info(f"Loaded demographics: {len(data['demographics'])} records")

        if teaching_methods_path and Path(teaching_methods_path).exists():
            tm_df = load_teaching_methods(teaching_methods_path)
            data["teaching_methods"] = encode_teaching_methods(tm_df)
            logger.info(f"Loaded teaching methods: {len(data['teaching_methods'])} records")

        if assessment_methods_path and Path(assessment_methods_path).exists():
            em_df = load_assessment_methods(assessment_methods_path)
            data["assessment_methods"] = encode_assessment_methods(em_df)
            logger.info(f"Loaded assessment methods: {len(data['assessment_methods'])} records")

        if study_hours_path and Path(study_hours_path).exists():
            data["study_hours"] = load_study_hours(study_hours_path)
            logger.info(f"Loaded study hours: {len(data['study_hours'])} records")

        if attendance_path and Path(attendance_path).exists():
            data["attendance"] = load_attendance(attendance_path)
            logger.info(f"Loaded attendance: {len(data['attendance'])} records")

        return data

    def prepare_training_dataset(
        self,
        data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """Prepare complete training dataset.

        Args:
            data: Dictionary of data sources from load_data()

        Returns:
            Complete training dataset with all features
        """
        logger.info("Preparing training dataset")

        exam_df = data["exam_scores"]
        conduct_df = data.get("conduct_scores")
        demo_df = data.get("demographics")
        tm_df = data.get("teaching_methods")
        em_df = data.get("assessment_methods")
        study_df = data.get("study_hours")
        attendance_df = data.get("attendance")

        # Merge all data
        training_df = create_training_dataset(
            exam_df=exam_df,
            conduct_df=conduct_df,
            demographics_df=demo_df,
            teaching_methods_df=tm_df,
            assessment_methods_df=em_df,
            study_hours_df=study_df,
            attendance_df=attendance_df,
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

        logger.info(
            f"Training dataset prepared: {len(training_df)} records, "
            f"{len(training_df.columns)} columns"
        )

        return training_df

    def prepare_features(
        self,
        training_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.Series, list]:
        """Prepare features for training.

        Args:
            training_df: Complete training dataset

        Returns:
            Tuple of (X, y, feature_names)
        """
        logger.info("Preparing features for training")

        # Select features (exclude target and ID columns)
        exclude_cols = [
            "Student_ID",
            "Subject_ID",
            "Lecturer_ID",
            "exam_score",
            "year",
        ]
        feature_cols = [col for col in training_df.columns if col not in exclude_cols]

        # Remove columns with all NaN
        feature_cols = [col for col in feature_cols if training_df[col].notna().sum() > 0]

        # Prepare X and y
        X = training_df[feature_cols].copy()
        y = training_df["exam_score"].copy()

        # Encode categorical columns
        self.label_encoders = {}
        for col in X.columns:
            if X[col].dtype == "object" or X[col].dtype.name == "category":
                le = LabelEncoder()
                X[col] = X[col].fillna("Unknown")
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
            elif X[col].dtype in [np.int64, np.float64]:
                X[col] = X[col].fillna(X[col].median())
            else:
                try:
                    X[col] = pd.to_numeric(X[col], errors="coerce")
                    X[col] = X[col].fillna(X[col].median())
                except Exception:
                    le = LabelEncoder()
                    X[col] = X[col].fillna("Unknown")
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.label_encoders[col] = le

        # Final check: fill any remaining NaN
        X = X.apply(pd.to_numeric, errors="coerce")
        X = X.fillna(0)

        nan_count = X.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"{nan_count} NaN values still present, filling with 0")
            X = X.fillna(0)

        self.feature_names = list(X.columns)

        logger.info(
            f"Features prepared: {len(self.feature_names)} features, "
            f"{len(X)} samples, NaN count: {X.isna().sum().sum()}"
        )

        return X, y, self.feature_names

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split data into train, validation, and test sets.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        logger.info("Splitting data into train/validation/test sets")

        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            shuffle=True,
        )

        # Further split training data for validation
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train,
            y_train,
            test_size=self.validation_size,
            random_state=self.random_state,
            shuffle=True,
        )

        # Ensure no NaN
        for df_name, df in [
            ("X_train_fit", X_train_fit),
            ("X_val", X_val),
            ("X_test", X_test),
        ]:
            nan_count = df.isna().sum().sum()
            if nan_count > 0:
                logger.warning(f"{nan_count} NaN values in {df_name}, filling with 0")
                df.fillna(0, inplace=True)

        logger.info(
            f"Data split: Train={len(X_train_fit)}, "
            f"Validation={len(X_val)}, Test={len(X_test)}"
        )

        return X_train_fit, X_val, X_test, y_train_fit, y_val, y_test

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        random_state: Optional[int] = None,
    ) -> EnsembleModel:
        """Train ensemble model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            random_state: Random seed (default: use pipeline random_state)

        Returns:
            Trained EnsembleModel
        """
        logger.info("Training ensemble model")

        if random_state is None:
            random_state = self.random_state

        model = EnsembleModel(random_state=random_state)
        training_metrics = model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
        )

        logger.info(
            f"Model training complete. Train MAE: {training_metrics['ensemble_train_mae']:.4f}, "
            f"Train R²: {training_metrics['ensemble_train_r2']:.4f}, "
            f"Val MAE: {training_metrics['ensemble_val_mae']:.4f}, "
            f"Val R²: {training_metrics['ensemble_val_r2']:.4f}"
        )

        return model

    def evaluate(
        self,
        model: EnsembleModel,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, float]:
        """Evaluate model on test set.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating model on test set")

        predictions = model.predict(X_test)
        metrics = evaluate_model(y_test, predictions, prefix="test_")

        # Evaluate by score range
        evaluate_by_score_range(y_test, predictions)

        logger.info(
            f"Test metrics: MAE={metrics['test_mae']:.4f}, "
            f"RMSE={metrics['test_rmse']:.4f}, R²={metrics['test_r2']:.4f}"
        )

        return metrics

    def save_model(
        self,
        model: EnsembleModel,
        output_path: str,
    ) -> None:
        """Save trained model to file.

        Args:
            model: Trained model
            output_path: Path to save model file
        """
        logger.info(f"Saving model to {output_path}")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        model.save(str(output_path))

        logger.info(f"Model saved: {model.model_name} (version {model.version})")

    def run(
        self,
        exam_scores_path: str,
        output_path: str,
        conduct_scores_path: Optional[str] = None,
        demographics_path: Optional[str] = None,
        teaching_methods_path: Optional[str] = None,
        assessment_methods_path: Optional[str] = None,
        study_hours_path: Optional[str] = None,
        attendance_path: Optional[str] = None,
    ) -> Tuple[EnsembleModel, Dict[str, float]]:
        """Run complete training pipeline.

        Args:
            exam_scores_path: Path to exam scores file (required)
            output_path: Path to save trained model
            conduct_scores_path: Path to conduct scores file (optional)
            demographics_path: Path to demographics file (optional)
            teaching_methods_path: Path to teaching methods file (optional)
            assessment_methods_path: Path to assessment methods file (optional)
            study_hours_path: Path to study hours file (optional)
            attendance_path: Path to attendance (điểm danh) file (optional)

        Returns:
            Tuple of (trained_model, evaluation_metrics)
        """
        logger.info("Starting training pipeline")

        # Step 1: Load data
        data = self.load_data(
            exam_scores_path=exam_scores_path,
            conduct_scores_path=conduct_scores_path,
            demographics_path=demographics_path,
            teaching_methods_path=teaching_methods_path,
            assessment_methods_path=assessment_methods_path,
            study_hours_path=study_hours_path,
            attendance_path=attendance_path,
        )

        # Step 2: Prepare training dataset
        training_df = self.prepare_training_dataset(data)

        # Step 3: Prepare features
        X, y, feature_names = self.prepare_features(training_df)

        # Step 4: Split data
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y)

        # Step 5: Train model
        model = self.train(X_train, y_train, X_val, y_val)

        # Step 6: Evaluate model
        metrics = self.evaluate(model, X_test, y_test)

        # Merge train/val metrics from model for CLI and API
        tm = getattr(model, "training_metrics", {}) or {}
        metrics["train_mae"] = tm.get("ensemble_train_mae")
        metrics["train_rmse"] = tm.get("ensemble_train_rmse")
        metrics["train_r2"] = tm.get("ensemble_train_r2")
        metrics["val_mae"] = tm.get("ensemble_val_mae")
        metrics["val_rmse"] = tm.get("ensemble_val_rmse")
        metrics["val_r2"] = tm.get("ensemble_val_r2")

        # Step 7: Save model
        self.save_model(model, output_path)

        logger.info("Training pipeline completed successfully")

        return model, metrics

