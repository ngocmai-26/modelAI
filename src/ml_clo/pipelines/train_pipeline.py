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
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split

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
from ml_clo.features.feature_encoder import prepare_features as shared_prepare_features
from ml_clo.models.ensemble_model import EnsembleModel
from ml_clo.models.model_evaluator import (
    evaluate_by_score_range,
    evaluate_model,
    print_evaluation_summary,
)
from ml_clo.utils.hash_utils import stable_hash_int
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
        group_split_by_student: bool = True,
    ):
        """Initialize training pipeline.

        Args:
            random_state: Random seed for reproducibility (default: 42)
            test_size: Proportion of data for testing (default: 0.2)
            validation_size: Proportion of training data for validation (default: 0.2)
            group_split_by_student: Mặc định True — GroupShuffleSplit theo Student_ID
                (một MSSV không vừa train vừa test). Đặt False hoặc dùng --no-group-split
                trên CLI để chia ngẫu nhiên theo dòng.
        """
        self.random_state = random_state
        self.test_size = test_size
        self.validation_size = validation_size
        self.group_split_by_student = group_split_by_student
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

    def report_data_quality(self, training_df: pd.DataFrame) -> Dict[str, object]:
        """MISSING-04: Emit a data-quality snapshot before feature encoding.

        Logs row count, target stats, missing-value rate per column, and
        duplicate keys. Returns the same info as a dict so callers can
        persist it alongside training metrics.
        """
        target = "exam_score"
        n_rows = len(training_df)
        n_missing_target = (
            int(training_df[target].isna().sum()) if target in training_df.columns else n_rows
        )
        missing_rate = training_df.isna().mean().sort_values(ascending=False)
        high_missing = {c: float(r) for c, r in missing_rate.items() if r > 0.30}

        dup_keys = 0
        if {"Student_ID", "Subject_ID", "year"}.issubset(training_df.columns):
            dup_keys = int(
                training_df.duplicated(subset=["Student_ID", "Subject_ID", "year"]).sum()
            )

        target_stats: Dict[str, float] = {}
        if target in training_df.columns:
            valid = training_df[target].dropna()
            if len(valid) > 0:
                target_stats = {
                    "min": float(valid.min()),
                    "max": float(valid.max()),
                    "mean": float(valid.mean()),
                    "std": float(valid.std()),
                }

        report = {
            "rows": n_rows,
            "missing_target": n_missing_target,
            "duplicate_student_subject_year": dup_keys,
            "high_missing_columns": high_missing,
            "target_stats": target_stats,
        }

        logger.info(f"Data quality: rows={n_rows}, missing_target={n_missing_target}")
        if dup_keys > 0:
            logger.warning(
                f"Data quality: {dup_keys} duplicate (Student_ID, Subject_ID, year) keys"
            )
        if high_missing:
            logger.warning(
                f"Data quality: {len(high_missing)} columns with >30% missing: "
                f"{list(high_missing.items())[:5]}"
            )
        if target_stats:
            logger.info(
                f"Data quality: target {target} "
                f"min={target_stats['min']:.2f}, max={target_stats['max']:.2f}, "
                f"mean={target_stats['mean']:.2f}, std={target_stats['std']:.2f}"
            )

        return report

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

        # DESIGN-02: Single source of truth for feature selection + encoding.
        X, y, feature_cols = shared_prepare_features(
            training_df,
            feature_names=None,
            target_column="exam_score",
        )
        self.feature_names = feature_cols

        logger.info(
            f"Features prepared: {len(self.feature_names)} features, "
            f"{len(X)} samples"
        )

        return X, y, self.feature_names

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        groups: Optional[pd.Series] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Split data into train, validation, and test sets.

        Args:
            X: Feature matrix
            y: Target vector
            groups: Optional group id per row (e.g. Student_ID) for GroupShuffleSplit

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if groups is not None and self.group_split_by_student:
            logger.info(
                "Splitting data into train/validation/test sets (grouped by student)"
            )
            gss = GroupShuffleSplit(
                n_splits=1,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            train_idx, test_idx = next(gss.split(X, y, groups=groups))
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            groups_train = groups.iloc[train_idx]

            gss_val = GroupShuffleSplit(
                n_splits=1,
                test_size=self.validation_size,
                random_state=self.random_state,
            )
            sub_tr, sub_val = next(gss_val.split(X_train, y_train, groups=groups_train))
            X_train_fit = X_train.iloc[sub_tr]
            X_val = X_train.iloc[sub_val]
            y_train_fit = y_train.iloc[sub_tr]
            y_val = y_train.iloc[sub_val]
        else:
            logger.info("Splitting data into train/validation/test sets (random rows)")
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

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        groups: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """MISSING-02: K-fold cross-validation over the ensemble.

        Trains a fresh `EnsembleModel` on each fold and aggregates test-fold
        MAE/RMSE/R². Uses `GroupKFold(Student_ID)` when groups are supplied
        and `group_split_by_student=True`, so a student never appears in both
        train and test of the same fold (mirrors the leakage protection in
        `split_data`).

        Args:
            X: Feature matrix (already encoded by prepare_features)
            y: Target vector
            n_splits: Number of folds (default: 5)
            groups: Per-row group identifier (e.g. Student_ID); required for
                grouped CV.

        Returns:
            Dict with mean and std of fold metrics:
            ``cv_mae_mean``, ``cv_mae_std``, ``cv_rmse_mean``, ``cv_rmse_std``,
            ``cv_r2_mean``, ``cv_r2_std``, ``cv_n_splits``.
        """
        if n_splits < 2:
            raise ValueError(f"n_splits must be >= 2, got {n_splits}")

        use_groups = groups is not None and self.group_split_by_student
        if use_groups:
            logger.info(f"Cross-validating with GroupKFold (k={n_splits}, by Student_ID)")
            splitter = GroupKFold(n_splits=n_splits)
            split_iter = splitter.split(X, y, groups=groups)
        else:
            logger.info(f"Cross-validating with KFold (k={n_splits})")
            splitter = KFold(
                n_splits=n_splits, shuffle=True, random_state=self.random_state
            )
            split_iter = splitter.split(X, y)

        fold_maes, fold_rmses, fold_r2s = [], [], []
        for fold_idx, (tr_idx, te_idx) in enumerate(split_iter, start=1):
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
            X_tr = X_tr.fillna(0)
            X_te = X_te.fillna(0)

            fold_model = EnsembleModel(random_state=self.random_state)
            fold_model.train(X_train=X_tr, y_train=y_tr, X_val=X_te, y_val=y_te)
            preds = fold_model.predict(X_te)
            fold_metrics = evaluate_model(y_te, preds, prefix="")
            fold_maes.append(fold_metrics["mae"])
            fold_rmses.append(fold_metrics["rmse"])
            fold_r2s.append(fold_metrics["r2"])
            logger.info(
                f"Fold {fold_idx}/{n_splits}: MAE={fold_metrics['mae']:.4f}, "
                f"RMSE={fold_metrics['rmse']:.4f}, R²={fold_metrics['r2']:.4f}"
            )

        result = {
            "cv_n_splits": n_splits,
            "cv_mae_mean": float(np.mean(fold_maes)),
            "cv_mae_std": float(np.std(fold_maes)),
            "cv_rmse_mean": float(np.mean(fold_rmses)),
            "cv_rmse_std": float(np.std(fold_rmses)),
            "cv_r2_mean": float(np.mean(fold_r2s)),
            "cv_r2_std": float(np.std(fold_r2s)),
        }
        logger.info(
            f"CV summary: MAE={result['cv_mae_mean']:.4f}±{result['cv_mae_std']:.4f}, "
            f"R²={result['cv_r2_mean']:.4f}±{result['cv_r2_std']:.4f}"
        )
        return result

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

        # Step 4: Split data (by student when enabled — no student in both train and test)
        groups = None
        if self.group_split_by_student and "Student_ID" in training_df.columns:
            groups = training_df.loc[X.index, "Student_ID"]

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y, groups=groups)

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

