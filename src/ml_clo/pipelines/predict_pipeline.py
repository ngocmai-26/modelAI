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
from typing import Any, Dict, Optional

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
from ml_clo.data.mergers import create_student_record_from_ids, create_training_dataset
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
    """Prediction pipeline for individual student CLO prediction.

    Model đã train chỉ nhận **feature vector** (khoảng 76 số) → điểm CLO. Để có feature
    vector cho một sinh viên cần dữ liệu gốc (điểm thi, rèn luyện, nhân khẩu, ...). Có hai cách:

    1. **Truyền data paths khi khởi tạo**: data được load và cache một lần, sau đó
       predict(student_id, subject_id, lecturer_id) không cần path.
    2. **Truyền paths vào predict() mỗi lần** (cách cũ, vẫn hỗ trợ).
    """

    def __init__(
        self,
        model_path: str,
        exam_scores_path: Optional[str] = None,
        conduct_scores_path: Optional[str] = None,
        demographics_path: Optional[str] = None,
        teaching_methods_path: Optional[str] = None,
        assessment_methods_path: Optional[str] = None,
        study_hours_path: Optional[str] = None,
    ):
        """Initialize prediction pipeline.

        Args:
            model_path: Path to trained model file (bắt buộc).
            exam_scores_path: Path file điểm thi. Nếu truyền, data sẽ được load và cache;
                sau đó predict() chỉ cần student_id, subject_id, lecturer_id.
            conduct_scores_path: Path điểm rèn luyện (tùy chọn).
            demographics_path: Path nhân khẩu (tùy chọn).
            teaching_methods_path: Path PPGD (tùy chọn).
            assessment_methods_path: Path PPDG (tùy chọn).
            study_hours_path: Path tự học (tùy chọn).

        Raises:
            ModelLoadError: Nếu không tìm thấy file model.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise ModelLoadError(f"Model file not found: {model_path}")

        self.model: Optional[EnsembleModel] = None
        self.explainer: Optional[EnsembleSHAPExplainer] = None
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: Optional[list] = None
        self._data_cache: Optional[Dict[str, Any]] = None

        if (
            exam_scores_path
            or (demographics_path and teaching_methods_path and assessment_methods_path)
        ):
            self.load_data_cache(
                exam_scores_path=exam_scores_path,
                conduct_scores_path=conduct_scores_path,
                demographics_path=demographics_path,
                teaching_methods_path=teaching_methods_path,
                assessment_methods_path=assessment_methods_path,
                study_hours_path=study_hours_path,
            )

    def load_data_cache(
        self,
        exam_scores_path: Optional[str] = None,
        conduct_scores_path: Optional[str] = None,
        demographics_path: Optional[str] = None,
        teaching_methods_path: Optional[str] = None,
        assessment_methods_path: Optional[str] = None,
        study_hours_path: Optional[str] = None,
    ) -> None:
        """Load và cache data. Cần ít nhất exam_scores_path HOẶC (demographics_path + teaching_methods_path + assessment_methods_path)."""
        logger.info("Loading and caching data for prediction (one-time)")
        cache: Dict[str, Any] = {}
        if exam_scores_path and Path(exam_scores_path).exists():
            exam_df = load_exam_scores(exam_scores_path)
            cache["exam_scores"] = preprocess_exam_scores(
                exam_df, convert_to_clo=True, create_result=False
            )
        else:
            cache["exam_scores"] = None
        if conduct_scores_path and Path(conduct_scores_path).exists():
            cache["conduct_scores"] = load_conduct_scores(conduct_scores_path)
        if demographics_path and Path(demographics_path).exists():
            cache["demographics"] = load_demographics(demographics_path)
        if teaching_methods_path and Path(teaching_methods_path).exists():
            tm_df = load_teaching_methods(teaching_methods_path)
            cache["teaching_methods"] = encode_teaching_methods(tm_df)
        if assessment_methods_path and Path(assessment_methods_path).exists():
            em_df = load_assessment_methods(assessment_methods_path)
            cache["assessment_methods"] = encode_assessment_methods(em_df)
        if study_hours_path and Path(study_hours_path).exists():
            cache["study_hours"] = load_study_hours(study_hours_path)
        self._data_cache = cache
        logger.info("Data cache ready")

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
        exam_scores_path: Optional[str] = None,
        conduct_scores_path: Optional[str] = None,
        demographics_path: Optional[str] = None,
        teaching_methods_path: Optional[str] = None,
        assessment_methods_path: Optional[str] = None,
        study_hours_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load and prepare data for a single student.

        Nếu đã gọi load_data_cache() (hoặc truyền paths khi khởi tạo), có thể bỏ qua các path.
        """
        logger.info(f"Loading data for student {student_id}, subject {subject_id}")

        if self._data_cache is not None:
            data = dict(self._data_cache)
        else:
            data = {}
            if exam_scores_path and Path(exam_scores_path).exists():
                exam_df = load_exam_scores(exam_scores_path)
                data["exam_scores"] = preprocess_exam_scores(
                    exam_df, convert_to_clo=True, create_result=False
                )
            else:
                data["exam_scores"] = None
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

        exam_df = data.get("exam_scores")
        if exam_df is not None:
            _sid = int(student_id) if isinstance(student_id, str) and student_id.isdigit() else student_id
            _subj = str(subject_id).strip()
            _lec = str(lecturer_id).strip()
            student_data = exam_df[
                (exam_df["Student_ID"] == _sid)
                & (exam_df["Subject_ID"] == _subj)
                & (exam_df["Lecturer_ID"] == _lec)
            ].copy()
        else:
            def _has_valid(v):
                return v is not None and (not isinstance(v, pd.DataFrame) or not v.empty)

            demo = data.get("demographics")
            ppgd = data.get("teaching_methods")
            ppdg = data.get("assessment_methods")
            if not (_has_valid(demo) and _has_valid(ppgd) and _has_valid(ppdg)):
                raise ValueError(
                    "No exam_scores and no demographics+teaching_methods+assessment_methods. "
                    "Pass exam_scores_path or (demographics_path, teaching_methods_path, assessment_methods_path)."
                )
            student_data = None

        if student_data is not None and len(student_data) > 0:
            full_exam_df = exam_df
            training_df = create_training_dataset(
                exam_df=student_data,
                conduct_df=data.get("conduct_scores"),
                demographics_df=data.get("demographics"),
                teaching_methods_df=data.get("teaching_methods"),
                assessment_methods_df=data.get("assessment_methods"),
                study_hours_df=data.get("study_hours"),
                target_column="exam_score",
                drop_missing_target=False,
            )
        else:
            # Fallback: tạo record từ nhân khẩu + PPGD/PPDG (SV/môn/GV mới không cần có trong DiemTong)
            base_df = create_student_record_from_ids(
                student_id=student_id,
                subject_id=subject_id,
                lecturer_id=lecturer_id,
                demographics_df=data.get("demographics"),
                teaching_methods_df=data.get("teaching_methods"),
                assessment_methods_df=data.get("assessment_methods"),
                year=2024,
            )
            from ml_clo.data.mergers import merge_exam_and_conduct_scores, merge_study_hours

            full_exam_df = pd.DataFrame(columns=["Student_ID", "Subject_ID", "Lecturer_ID", "year", "exam_score"])
            if data.get("conduct_scores") is not None:
                base_df = merge_exam_and_conduct_scores(base_df, data["conduct_scores"], year_column="year")
            if data.get("study_hours") is not None:
                base_df = merge_study_hours(base_df, data["study_hours"], year_column="year")
            training_df = build_all_features(
                base_df,
                conduct_history_df=data.get("conduct_scores"),
                exam_history_df=exam_df if exam_df is not None else full_exam_df,
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
                    X[col] = X[col].fillna("__UNKNOWN__")
                    vals = X[col].astype(str)
                    mask = vals.isin(le.classes_)
                    encoded = np.full(len(vals), -1, dtype=np.int64)
                    if mask.any():
                        encoded[mask] = le.transform(vals[mask])
                    X[col] = encoded
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
        exam_scores_path: Optional[str] = None,
        conduct_scores_path: Optional[str] = None,
        demographics_path: Optional[str] = None,
        teaching_methods_path: Optional[str] = None,
        assessment_methods_path: Optional[str] = None,
        study_hours_path: Optional[str] = None,
        actual_clo_score: Optional[float] = None,
    ) -> IndividualAnalysisOutput:
        """Predict CLO score and generate explanation for a student.

        Nếu đã truyền data paths khi khởi tạo (hoặc gọi load_data_cache()), chỉ cần:
        predict(student_id=..., subject_id=..., lecturer_id=...).

        Môn đã học & đã đỗ: truyền actual_clo_score → output ưu tiên điểm thực, vẫn dùng SHAP cho nguyên nhân.
        Môn chưa học: không truyền actual_clo_score → trả về điểm dự đoán + nguyên nhân.

        Args:
            student_id: Student ID
            subject_id: Subject ID
            lecturer_id: Lecturer ID
            exam_scores_path: Path file điểm thi (bỏ qua nếu đã cache data khi init)
            conduct_scores_path: Path điểm rèn luyện (optional)
            demographics_path: Path nhân khẩu (optional)
            teaching_methods_path: Path PPGD (optional)
            assessment_methods_path: Path PPDG (optional)
            study_hours_path: Path tự học (optional)
            actual_clo_score: Điểm CLO thực (môn đã đỗ) — nếu có thì output ưu tiên giá trị này

        Returns:
            IndividualAnalysisOutput with prediction and explanation
        """
        logger.info(f"Predicting CLO score for student {student_id}")

        if self.model is None:
            self.load_model()
        if self.explainer is None:
            self.explainer = EnsembleSHAPExplainer(self.model, cache_explainer=True)

        has_cache = self._data_cache is not None
        has_exam = exam_scores_path is not None and Path(exam_scores_path).exists()
        has_demo_tm_am = (
            demographics_path and teaching_methods_path and assessment_methods_path
            and Path(demographics_path).exists()
            and Path(teaching_methods_path).exists()
            and Path(assessment_methods_path).exists()
        )
        if not has_cache and not has_exam and not has_demo_tm_am:
            raise ValueError(
                "No data. Pass at init or to predict(): "
                "exam_scores_path, OR (demographics_path + teaching_methods_path + assessment_methods_path)."
            )

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

        # Generate explanation — khi có actual_clo_score thì summary/context dùng actual
        display_score = actual_clo_score if actual_clo_score is not None else predicted_score
        explanation = generate_complete_explanation(
            top_negative_impacts=processed["top_negative_impacts"],
            predicted_score=display_score,
            context="individual",
            include_solutions=True,
        )

        # Convert to output schema (predicted = model, actual = điểm thực nếu có)
        output = IndividualAnalysisOutput.from_explanation_dict(
            explanation,
            student_id=student_id,
            subject_id=subject_id,
            lecturer_id=lecturer_id,
            predicted_clo_score=predicted_score,
            actual_clo_score=actual_clo_score,
        )

        logger.info(
            f"Prediction complete: predicted={predicted_score:.2f}, "
            f"actual={'%.2f' % actual_clo_score if actual_clo_score is not None else 'N/A'}, "
            f"reasons={len(output.reasons)}"
        )

        return output

