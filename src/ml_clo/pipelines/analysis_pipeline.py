"""Analysis pipeline for class-level CLO prediction and analysis.

This module provides a complete analysis pipeline that integrates:
- Model loading
- Class data loading (all students in a subject)
- Batch prediction
- SHAP aggregation
- Class-level reason generation
- Data storage for retraining
- Structured output
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

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
from ml_clo.data.mergers import (
    create_student_record_from_ids,
    create_training_dataset,
    merge_attendance,
    merge_exam_and_conduct_scores,
    merge_study_hours,
)
from ml_clo.data.preprocessors import preprocess_exam_scores
from ml_clo.features.feature_builder import build_all_features
from ml_clo.features.feature_encoder import prepare_features as shared_prepare_features
from ml_clo.models.ensemble_model import EnsembleModel
from ml_clo.outputs.schemas import ClassAnalysisOutput
from ml_clo.reasoning.reason_generator import (
    generate_complete_explanation,
    generate_explanation_from_distribution,
)
from ml_clo.utils.exceptions import ModelLoadError
from ml_clo.utils.logger import get_logger
from ml_clo.xai.shap_explainer import EnsembleSHAPExplainer
from ml_clo.xai.shap_postprocess import (
    aggregate_class_shap,
    group_shap_by_pedagogy,
    process_shap_for_analysis,
)

logger = get_logger(__name__)


class AnalysisPipeline:
    """Analysis pipeline for class-level CLO prediction and analysis."""

    def __init__(self, model_path: str):
        """Initialize analysis pipeline.

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

    def load_class_data(
        self,
        subject_id: str,
        lecturer_id: str,
        exam_scores_path: str,
        conduct_scores_path: Optional[str] = None,
        demographics_path: Optional[str] = None,
        teaching_methods_path: Optional[str] = None,
        assessment_methods_path: Optional[str] = None,
        study_hours_path: Optional[str] = None,
        attendance_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load data for all students in a class (filter DiemTong).

        .. deprecated::
            Dùng :meth:`analyze_class_from_scores` với danh sách điểm CLO thay vì filter DiemTong.

        Args:
            subject_id: Subject ID
            lecturer_id: Lecturer ID
            exam_scores_path: Path to exam scores file
            conduct_scores_path: Path to conduct scores file (optional)
            demographics_path: Path to demographics file (optional)
            teaching_methods_path: Path to teaching methods file (optional)
            assessment_methods_path: Path to assessment methods file (optional)
            study_hours_path: Path to study hours file (optional)
            attendance_path: Path to attendance file (optional)

        Returns:
            DataFrame with class data ready for analysis
        """
        logger.warning(
            "load_class_data (filter DiemTong) deprecated. "
            "Dùng analyze_class_from_scores(clo_scores=...) thay thế."
        )
        logger.info(f"Loading class data for subject {subject_id}, lecturer {lecturer_id}")

        # Load all data sources
        exam_df = load_exam_scores(exam_scores_path)
        exam_df = preprocess_exam_scores(exam_df, convert_to_clo=True, create_result=False)

        # Filter for specific subject and lecturer
        class_data = exam_df[
            (exam_df["Subject_ID"] == subject_id) & (exam_df["Lecturer_ID"] == lecturer_id)
        ].copy()

        if len(class_data) == 0:
            raise ValueError(
                f"No data found for subject_id={subject_id}, lecturer_id={lecturer_id}"
            )

        logger.info(f"Found {len(class_data)} students in class")

        # Load other data sources
        data = {"exam_scores": class_data}

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

        if attendance_path and Path(attendance_path).exists():
            data["attendance"] = load_attendance(attendance_path)

        # Merge and build features (need full history for features)
        full_exam_df = load_exam_scores(exam_scores_path)
        full_exam_df = preprocess_exam_scores(full_exam_df, convert_to_clo=True, create_result=False)

        training_df = create_training_dataset(
            exam_df=class_data,
            conduct_df=data.get("conduct_scores"),
            demographics_df=data.get("demographics"),
            teaching_methods_df=data.get("teaching_methods"),
            assessment_methods_df=data.get("assessment_methods"),
            study_hours_df=data.get("study_hours"),
            attendance_df=data.get("attendance"),
            target_column="exam_score",
            drop_missing_target=False,  # Don't drop, we need to predict
        )

        # Build aggregate features
        training_df = build_all_features(
            training_df,
            conduct_history_df=data.get("conduct_scores"),
            exam_history_df=full_exam_df,  # Use full history for features
            study_hours_df=data.get("study_hours"),
        )

        logger.info(f"Class data prepared: {len(training_df)} records")

        return training_df

    def prepare_features(
        self,
        class_df: pd.DataFrame,
        label_encoders: Optional[Dict[str, LabelEncoder]] = None,
    ) -> pd.DataFrame:
        """Prepare features for prediction.

        Args:
            class_df: Class data DataFrame
            label_encoders: Label encoders from training (optional)

        Returns:
            Feature matrix ready for prediction
        """
        logger.info("Preparing features for class prediction")

        # DESIGN-02: shared encoder. The legacy `label_encoders` parameter is
        # accepted for backward compatibility but no longer used (hash encoding
        # is stateless and consistent across train/predict/analyze).
        del label_encoders  # explicitly mark as unused
        X, _, _ = shared_prepare_features(
            class_df,
            feature_names=self.feature_names,
            target_column="exam_score",
        )

        logger.info(f"Features prepared: {X.shape}")

        return X

    def store_actual_scores(
        self,
        class_df: pd.DataFrame,
        actual_scores: Optional[Dict[str, float]] = None,
        storage_path: Optional[str] = None,
    ) -> None:
        """Store actual CLO scores for future retraining.

        Args:
            class_df: Class data DataFrame
            actual_scores: Dictionary mapping student_id to actual CLO score (optional)
            storage_path: Path to store data (optional, if None, only logs)
        """
        if actual_scores is None or len(actual_scores) == 0:
            logger.info("No actual scores provided, skipping storage")
            return

        logger.info(f"Storing {len(actual_scores)} actual CLO scores for retraining")

        # Create storage DataFrame
        storage_data = []
        for student_id, clo_score in actual_scores.items():
            student_data = class_df[class_df["Student_ID"] == student_id].iloc[0]
            storage_data.append({
                "Student_ID": student_id,
                "Subject_ID": student_data.get("Subject_ID"),
                "Lecturer_ID": student_data.get("Lecturer_ID"),
                "clo_score": clo_score,
                "year": student_data.get("year"),
                # Store key features for retraining
                "exam_score": clo_score,  # Actual CLO score
            })

        storage_df = pd.DataFrame(storage_data)

        if storage_path:
            storage_path = Path(storage_path)
            storage_path.parent.mkdir(parents=True, exist_ok=True)
            storage_df.to_csv(storage_path, index=False)
            logger.info(f"Stored actual scores to {storage_path}")
        else:
            logger.info("Storage path not provided, data logged only")

    def analyze_class(
        self,
        subject_id: str,
        lecturer_id: str,
        exam_scores_path: str,
        conduct_scores_path: Optional[str] = None,
        demographics_path: Optional[str] = None,
        teaching_methods_path: Optional[str] = None,
        assessment_methods_path: Optional[str] = None,
        study_hours_path: Optional[str] = None,
        attendance_path: Optional[str] = None,
        actual_scores: Optional[Dict[str, float]] = None,
        storage_path: Optional[str] = None,
    ) -> ClassAnalysisOutput:
        """Analyze class by filtering DiemTong (chế độ cũ).

        .. deprecated::
            Dùng :meth:`analyze_class_from_scores` với danh sách điểm CLO.
            API/backend truyền điểm trực tiếp, không cần DiemTong.

        Args:
            subject_id: Subject ID
            lecturer_id: Lecturer ID
            exam_scores_path: Path to exam scores file
            conduct_scores_path: Path to conduct scores file (optional)
            demographics_path: Path to demographics file (optional)
            teaching_methods_path: Path to teaching methods file (optional)
            assessment_methods_path: Path to assessment methods file (optional)
            study_hours_path: Path to study hours file (optional)
            actual_scores: Dictionary mapping student_id to actual CLO score (optional)
            storage_path: Path to store actual scores for retraining (optional)

        Returns:
            ClassAnalysisOutput with class-level analysis
        """
        logger.warning(
            "analyze_class (filter DiemTong) deprecated. "
            "Dùng analyze_class_from_scores(clo_scores=...) thay thế."
        )
        logger.info(f"Analyzing class: subject {subject_id}, lecturer {lecturer_id}")

        # Load model if not already loaded
        if self.model is None:
            self.load_model()

        # Initialize explainer if not already initialized
        if self.explainer is None:
            self.explainer = EnsembleSHAPExplainer(self.model, cache_explainer=True)

        # Load class data
        class_df = self.load_class_data(
            subject_id=subject_id,
            lecturer_id=lecturer_id,
            exam_scores_path=exam_scores_path,
            conduct_scores_path=conduct_scores_path,
            demographics_path=demographics_path,
            teaching_methods_path=teaching_methods_path,
            assessment_methods_path=assessment_methods_path,
            study_hours_path=study_hours_path,
            attendance_path=attendance_path,
        )

        # Prepare features
        X = self.prepare_features(class_df)

        # Predict for all students
        predictions = self.model.predict(X)
        average_predicted_score = float(np.mean(predictions))

        # Compute SHAP values for all students
        shap_values_batch = self.explainer.explain_batch(X)

        # Aggregate SHAP values for class-level analysis
        aggregated_shap = aggregate_class_shap(
            [shap_values_batch[i] for i in range(len(shap_values_batch))],
            feature_names=self.feature_names,
        )

        # Process aggregated SHAP
        processed = process_shap_for_analysis(
            aggregated_shap,
            feature_names=self.feature_names,
            df=None,
        )

        # NEW-05: Pass a class-mean feature row so reason templates can
        # calibrate against the actual class profile (e.g. good attendance
        # → avoid phrasing SHAP-negative groups as real-world deficits).
        class_mean_row = X.mean(numeric_only=True)

        # Generate class-level explanation
        explanation = generate_complete_explanation(
            top_negative_impacts=processed["top_negative_impacts"],
            predicted_score=average_predicted_score,
            context="class",
            include_solutions=True,
            raw_feature_row=class_mean_row,
        )

        # DESIGN-09: real per-group affected counts (not blanket class size)
        affected_counts = self._count_affected_students_per_group(shap_values_batch)
        self._inject_affected_counts(explanation, affected_counts, len(class_df))

        # Store actual scores if provided
        if actual_scores:
            self.store_actual_scores(class_df, actual_scores, storage_path)

        # Convert to output schema
        output = ClassAnalysisOutput.from_explanation_dict(
            explanation,
            subject_id=subject_id,
            lecturer_id=lecturer_id,
            total_students=len(class_df),
            average_predicted_score=average_predicted_score,
        )

        logger.info(
            f"Class analysis complete: {len(output.common_reasons)} reasons, "
            f"average_score={average_predicted_score:.2f}"
        )

        return output

    def _count_affected_students_per_group(
        self,
        shap_values_batch: np.ndarray,
    ) -> Dict[str, int]:
        """DESIGN-09: Compute how many students each pedagogical group
        actually drags down (negative SHAP) at the per-student level.

        Previously we defaulted `affected_students_count` to the full class
        size, which is misleading for a class where only a minority are hurt
        by a given group. This uses `group_shap_by_pedagogy` on the full
        (n_samples, n_features) batch so we get (n_samples,) per group, then
        counts rows with sum < 0.
        """
        per_sample_grouped = group_shap_by_pedagogy(
            shap_values_batch,
            feature_names=self.feature_names,
            df=None,
        )
        counts: Dict[str, int] = {}
        for group_name, vals in per_sample_grouped.items():
            arr = np.asarray(vals)
            if arr.ndim == 0:
                counts[group_name] = int(arr < 0)
            else:
                counts[group_name] = int(np.sum(arr < 0))
        return counts

    @staticmethod
    def _inject_affected_counts(
        explanation: Dict,
        counts: Dict[str, int],
        fallback_total: int,
    ) -> None:
        """Attach affected_students_count to each reason in-place."""
        for reason in explanation.get("reasons", []):
            key = reason.get("reason_key") or reason.get("group_name")
            reason["affected_students_count"] = int(counts.get(key, fallback_total))

    def _normalize_clo_scores(
        self,
        clo_scores: Union[Dict[str, float], List[float], List[Tuple[str, float]]],
    ) -> List[Tuple[str, float]]:
        """Chuẩn hóa clo_scores thành List[(student_id, score)]."""
        if isinstance(clo_scores, dict):
            return [(str(k), float(v)) for k, v in clo_scores.items()]
        if isinstance(clo_scores, list) and len(clo_scores) > 0:
            first = clo_scores[0]
            if isinstance(first, (list, tuple)) and len(first) >= 2:
                return [(str(p[0]), float(p[1])) for p in clo_scores]
            return [("_anonymous_" + str(i), float(s)) for i, s in enumerate(clo_scores)]
        return []

    def analyze_class_from_scores(
        self,
        subject_id: str,
        lecturer_id: str,
        clo_scores: Union[Dict[str, float], List[float], List[Tuple[str, float]]],
        demographics_path: Optional[str] = None,
        conduct_scores_path: Optional[str] = None,
        teaching_methods_path: Optional[str] = None,
        assessment_methods_path: Optional[str] = None,
        study_hours_path: Optional[str] = None,
        attendance_path: Optional[str] = None,
    ) -> ClassAnalysisOutput:
        """Phân tích lớp từ danh sách điểm CLO (không cần DiemTong).

        Đầu vào: môn, mã GV, danh sách điểm. Có thể kèm MSSV hoặc chỉ điểm.

        Args:
            subject_id: Mã môn học
            lecturer_id: Mã giảng viên
            clo_scores: Một trong:
                - Dict[student_id, score]
                - List[score] (chỉ điểm)
                - List[(student_id, score)]
            demographics_path: Path nhân khẩu (bắt buộc khi có MSSV)
            conduct_scores_path: Path điểm rèn luyện (tùy chọn)
            teaching_methods_path: Path PPGD (bắt buộc khi có MSSV)
            assessment_methods_path: Path PPDG (bắt buộc khi có MSSV)
            study_hours_path: Path tự học (tùy chọn)

        Returns:
            ClassAnalysisOutput
        """
        logger.info(f"Analyzing class from scores: subject {subject_id}, lecturer {lecturer_id}")

        pairs = self._normalize_clo_scores(clo_scores)
        if not pairs:
            raise ValueError("clo_scores trống hoặc định dạng không hợp lệ")

        scores_only = [p[1] for p in pairs]
        has_student_ids = all(
            sid and not sid.startswith("_anonymous_") and sid != "_no_id"
            for sid, _ in pairs
        )

        if not has_student_ids:
            return self._analyze_from_distribution(
                subject_id=subject_id,
                lecturer_id=lecturer_id,
                scores=scores_only,
            )

        has_data = (
            demographics_path and Path(demographics_path).exists()
            and teaching_methods_path and Path(teaching_methods_path).exists()
            and assessment_methods_path and Path(assessment_methods_path).exists()
        )
        if not has_data:
            return self._analyze_from_distribution(
                subject_id=subject_id,
                lecturer_id=lecturer_id,
                scores=scores_only,
            )

        return self._analyze_with_shap(
            subject_id=subject_id,
            lecturer_id=lecturer_id,
            pairs=pairs,
            demographics_path=demographics_path,
            conduct_scores_path=conduct_scores_path,
            teaching_methods_path=teaching_methods_path,
            assessment_methods_path=assessment_methods_path,
            study_hours_path=study_hours_path,
            attendance_path=attendance_path,
        )

    def _analyze_from_distribution(
        self,
        subject_id: str,
        lecturer_id: str,
        scores: List[float],
    ) -> ClassAnalysisOutput:
        """Phân tích chỉ từ phân phối điểm (không SHAP)."""
        explanation = generate_explanation_from_distribution(scores, context="class")
        return ClassAnalysisOutput.from_explanation_dict(
            explanation,
            subject_id=subject_id,
            lecturer_id=lecturer_id,
            total_students=len(scores),
            average_predicted_score=explanation["predicted_score"],
        )

    def _analyze_with_shap(
        self,
        subject_id: str,
        lecturer_id: str,
        pairs: List[Tuple[str, float]],
        demographics_path: str,
        conduct_scores_path: Optional[str] = None,
        teaching_methods_path: Optional[str] = None,
        assessment_methods_path: Optional[str] = None,
        study_hours_path: Optional[str] = None,
        attendance_path: Optional[str] = None,
    ) -> ClassAnalysisOutput:
        """Phân tích có MSSV: build features, predict, SHAP, aggregate."""
        if self.model is None:
            self.load_model()
        if self.explainer is None:
            self.explainer = EnsembleSHAPExplainer(self.model, cache_explainer=True)

        data = {}
        data["demographics"] = load_demographics(demographics_path)
        tm_df = load_teaching_methods(teaching_methods_path)
        data["teaching_methods"] = encode_teaching_methods(tm_df)
        em_df = load_assessment_methods(assessment_methods_path)
        data["assessment_methods"] = encode_assessment_methods(em_df)
        data["conduct_scores"] = (
            load_conduct_scores(conduct_scores_path)
            if conduct_scores_path and Path(conduct_scores_path).exists()
            else None
        )
        data["study_hours"] = (
            load_study_hours(study_hours_path)
            if study_hours_path and Path(study_hours_path).exists()
            else None
        )
        data["attendance"] = (
            load_attendance(attendance_path)
            if attendance_path and Path(attendance_path).exists()
            else None
        )

        full_exam_df = pd.DataFrame(columns=["Student_ID", "Subject_ID", "Lecturer_ID", "year", "exam_score"])
        records = []

        for student_id, score in pairs:
            base_df = create_student_record_from_ids(
                student_id=student_id,
                subject_id=subject_id,
                lecturer_id=lecturer_id,
                demographics_df=data["demographics"],
                teaching_methods_df=data["teaching_methods"],
                assessment_methods_df=data["assessment_methods"],
                study_hours_df=data.get("study_hours"),
                year=2024,
            )
            base_df["exam_score"] = score
            if data["conduct_scores"] is not None:
                base_df = merge_exam_and_conduct_scores(base_df, data["conduct_scores"], year_column="year")
            if data["study_hours"] is not None:
                base_df = merge_study_hours(base_df, data["study_hours"], year_column="year")
            if data["attendance"] is not None:
                base_df = merge_attendance(base_df, data["attendance"], year_column="year")
            records.append(base_df)

        class_df = pd.concat(records, ignore_index=True)
        class_df = build_all_features(
            class_df,
            conduct_history_df=data["conduct_scores"],
            exam_history_df=full_exam_df,
            study_hours_df=data["study_hours"],
        )

        X = self.prepare_features(class_df)
        predictions = self.model.predict(X)
        average_predicted_score = float(np.mean(predictions))

        shap_values_batch = self.explainer.explain_batch(X)
        aggregated_shap = aggregate_class_shap(
            [shap_values_batch[i] for i in range(len(shap_values_batch))],
            feature_names=self.feature_names,
        )
        processed = process_shap_for_analysis(
            aggregated_shap,
            feature_names=self.feature_names,
            df=None,
        )

        # NEW-05: calibrate class reasons against the class-mean feature row.
        class_mean_row = X.mean(numeric_only=True)

        explanation = generate_complete_explanation(
            top_negative_impacts=processed["top_negative_impacts"],
            predicted_score=average_predicted_score,
            context="class",
            include_solutions=True,
            raw_feature_row=class_mean_row,
        )

        return ClassAnalysisOutput.from_explanation_dict(
            explanation,
            subject_id=subject_id,
            lecturer_id=lecturer_id,
            total_students=len(class_df),
            average_predicted_score=average_predicted_score,
        )

