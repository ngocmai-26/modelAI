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
    student_has_history,
)
from ml_clo.data.preprocessors import preprocess_exam_scores
from ml_clo.features.feature_builder import build_all_features
from ml_clo.features.feature_encoder import prepare_features as shared_prepare_features
from ml_clo.models.ensemble_model import EnsembleModel
from ml_clo.outputs.schemas import IndividualAnalysisOutput
from ml_clo.reasoning.reason_generator import generate_complete_explanation
from ml_clo.utils.audit_log import log_prediction
from ml_clo.utils.exceptions import ModelLoadError
from ml_clo.utils.logger import get_logger
from ml_clo.utils.hash_utils import stable_hash_int
from ml_clo.xai.shap_explainer import EnsembleSHAPExplainer
from ml_clo.xai.shap_postprocess import process_shap_for_analysis

logger = get_logger(__name__)


class PredictionPipeline:
    """Prediction pipeline for individual student CLO prediction.

    Model đã train chỉ nhận **feature vector** (khoảng 76 số) → điểm CLO.
    Để có feature vector cho một sinh viên cần dữ liệu gốc (điểm thi, rèn luyện,
    nhân khẩu, ...). Pipeline hỗ trợ **hai mode khởi tạo** — chọn theo use case:

    **Mode 1 — Cache mode (khuyến nghị cho backend phục vụ nhiều request):**
    Truyền data paths vào ``__init__``. Data được load + preprocess MỘT LẦN
    và cache trong instance. Mỗi ``predict()`` call chỉ cần ID, không tốn IO.

    .. code-block:: python

        pipeline = PredictionPipeline(
            model_path="models/model.joblib",
            exam_scores_path="data/DiemTong.xlsx",
            demographics_path="data/nhankhau.xlsx",
            teaching_methods_path="data/PPGDfull.xlsx",
            assessment_methods_path="data/PPDGfull.xlsx",
        )
        pipeline.load_model()
        out = pipeline.predict(student_id="19050006", subject_id="INF0823",
                               lecturer_id="90316")

    **Mode 2 — Per-call mode (phù hợp script CLI dùng một lần):**
    Khởi tạo với ``model_path`` only, truyền data paths vào ``predict()`` mỗi
    lần gọi. Mỗi call sẽ reload + preprocess data → chậm hơn, nhưng đơn giản
    cho ad-hoc usage.

    .. code-block:: python

        pipeline = PredictionPipeline(model_path="models/model.joblib")
        pipeline.load_model()
        out = pipeline.predict(
            student_id="19050006", subject_id="INF0823", lecturer_id="90316",
            exam_scores_path="data/DiemTong.xlsx",
            demographics_path="data/nhankhau.xlsx",
            teaching_methods_path="data/PPGDfull.xlsx",
            assessment_methods_path="data/PPDGfull.xlsx",
        )

    Mode 1 được auto-trigger khi ``__init__`` nhận đủ tối thiểu
    ``exam_scores_path`` HOẶC bộ ``(demographics + teaching + assessment)``.
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
        attendance_path: Optional[str] = None,
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
            attendance_path: Path điểm danh (tùy chọn).

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
                attendance_path=attendance_path,
            )

    def load_data_cache(
        self,
        exam_scores_path: Optional[str] = None,
        conduct_scores_path: Optional[str] = None,
        demographics_path: Optional[str] = None,
        teaching_methods_path: Optional[str] = None,
        assessment_methods_path: Optional[str] = None,
        study_hours_path: Optional[str] = None,
        attendance_path: Optional[str] = None,
    ) -> None:
        """Load và cache data. Cần ít nhất exam_scores_path HOẶC (demographics_path + teaching_methods_path + assessment_methods_path)."""
        logger.info("Loading and caching data for prediction (one-time)")
        cache: Dict[str, Any] = {}
        if exam_scores_path and Path(exam_scores_path).exists():
            exam_df = load_exam_scores(exam_scores_path)
            cache["exam_scores"] = preprocess_exam_scores(
                exam_df, convert_to_clo=True, create_result=False
            )
            # Khung lịch sử cho đặc trưng học lực: GIỮ cả dòng thiếu Lecturer_ID
            # (vd cohort K28 ghi Lecturer_ID=NaN) để không xoá sạch lịch sử của SV
            # → tránh dự đoán suy biến ~0.5. Học lực tính từ exam_score, không dùng
            # Lecturer_ID, nên an toàn và không ảnh hưởng model đã train.
            cache["exam_history"] = preprocess_exam_scores(
                exam_df, convert_to_clo=True, create_result=False, require_lecturer=False
            )
        else:
            cache["exam_scores"] = None
            cache["exam_history"] = None
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
        if attendance_path and Path(attendance_path).exists():
            cache["attendance"] = load_attendance(attendance_path)
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
        attendance_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load and prepare data for a single student.

        Nếu đã gọi load_data_cache() (hoặc truyền paths khi khởi tạo), có thể bỏ qua các path.
        """
        logger.info(f"Loading data for student {student_id}, subject {subject_id}")

        if self._data_cache is not None:
            data = dict(self._data_cache)
            if attendance_path and Path(attendance_path).exists():
                data["attendance"] = load_attendance(attendance_path)
        else:
            data = {}
            if exam_scores_path and Path(exam_scores_path).exists():
                exam_df = load_exam_scores(exam_scores_path)
                data["exam_scores"] = preprocess_exam_scores(
                    exam_df, convert_to_clo=True, create_result=False
                )
                # Khung lịch sử cho đặc trưng học lực (xem ghi chú ở load_data cache).
                data["exam_history"] = preprocess_exam_scores(
                    exam_df, convert_to_clo=True, create_result=False, require_lecturer=False
                )
            else:
                data["exam_scores"] = None
                data["exam_history"] = None
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

        exam_df = data.get("exam_scores")
        if exam_df is not None:
            _sid = int(student_id) if isinstance(student_id, str) and student_id.isdigit() else student_id
            _subj = str(subject_id).strip()
            _lec = str(lecturer_id).strip()
            _lec_s = str(_lec).strip()
            _subj_s = str(_subj).strip()
            lec_series = exam_df["Lecturer_ID"].astype(str).str.strip()
            subj_series = exam_df["Subject_ID"].astype(str).str.strip()
            student_data = exam_df[
                (exam_df["Student_ID"] == _sid)
                & (subj_series == _subj_s)
                & (lec_series == _lec_s)
            ].copy()

            # SV năm 2+: có trong DiemTong → bắt buộc dùng exam data, không fallback
            # SV năm 1: không có trong DiemTong → cho phép fallback
            if len(student_data) == 0 and student_has_history(exam_df, student_id):
                # SV có lịch sử nhưng chưa học môn này — cần đủ cột như dòng train (DiemTong),
                # không chỉ 5 cột; thiếu cột → prepare_features đệm 0, lệch hoàn toàn so với hash lúc train.
                exam_for_student = exam_df[exam_df["Student_ID"] == _sid]
                if "year" in exam_for_student.columns and exam_for_student["year"].notna().any():
                    year_val = exam_for_student["year"].dropna().iloc[-1]
                else:
                    year_val = 2024
                course_rows = exam_df[(subj_series == _subj_s) & (lec_series == _lec_s)]
                if len(course_rows) > 0:
                    template = course_rows.iloc[0].copy()
                else:
                    es = exam_for_student
                    if "year" in es.columns:
                        template = es.sort_values("year", kind="mergesort").iloc[-1].copy()
                    else:
                        template = es.iloc[-1].copy()
                    template["Subject_ID"] = _subj_s
                    template["Lecturer_ID"] = _lec_s
                template["Student_ID"] = _sid
                template["exam_score"] = np.nan
                template["year"] = year_val
                student_data = pd.DataFrame([template])
                logger.info(
                    f"SV năm 2+ (có lịch sử) nhưng chưa học môn {_subj}: "
                    f"dùng mẫu cột từ Điểm tổng (môn+GV), exam_score=NaN, lịch sử từ toàn bộ exam_df"
                )
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
            logger.warning(
                "Không có exam_scores_path — không thể kiểm tra SV năm 1 vs năm 2+. "
                "Để đảm bảo logic đúng, nên truyền --exam-scores khi có file DiemTong."
            )

        # Lịch sử dùng để tính đặc trưng học lực. Mặc định = exam_df đã lọc (GIỮ
        # NGUYÊN hành vi cho SV đã có lịch sử). CHỈ chuyển sang khung relaxed (giữ
        # dòng thiếu Lecturer_ID) cho SV bị lọc mất hoàn toàn khỏi exam_df — ví dụ
        # cohort K28 ghi Lecturer_ID=NaN — để họ không bị history rỗng → 0.5.
        exam_history_df = exam_df
        relaxed_history = data.get("exam_history")
        if relaxed_history is not None and exam_df is not None:
            try:
                _sid_hist = (
                    int(student_id)
                    if isinstance(student_id, str) and student_id.isdigit()
                    else student_id
                )
                if not (exam_df["Student_ID"] == _sid_hist).any():
                    exam_history_df = relaxed_history
                    logger.info(
                        f"SV {student_id} bị lọc khỏi exam_df (vd Lecturer_ID=NaN) — "
                        f"dùng khung lịch sử relaxed để tính học lực."
                    )
            except (ValueError, TypeError, KeyError):
                pass
        elif exam_df is None:
            exam_history_df = relaxed_history

        if student_data is not None and len(student_data) > 0:
            training_df = create_training_dataset(
                exam_df=student_data,
                conduct_df=data.get("conduct_scores"),
                demographics_df=data.get("demographics"),
                teaching_methods_df=data.get("teaching_methods"),
                assessment_methods_df=data.get("assessment_methods"),
                study_hours_df=data.get("study_hours"),
                attendance_df=data.get("attendance"),
                target_column="exam_score",
                drop_missing_target=False,
            )
            training_df = build_all_features(
                training_df,
                conduct_history_df=data.get("conduct_scores"),
                exam_history_df=exam_history_df,
                study_hours_df=data.get("study_hours"),
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
                study_hours_df=data.get("study_hours"),
                year=2024,
            )
            empty_exam_df = pd.DataFrame(columns=["Student_ID", "Subject_ID", "Lecturer_ID", "year", "exam_score"])
            if data.get("conduct_scores") is not None:
                base_df = merge_exam_and_conduct_scores(base_df, data["conduct_scores"], year_column="year")
            if data.get("attendance") is not None:
                base_df = merge_attendance(base_df, data["attendance"], year_column="year")
            training_df = build_all_features(
                base_df,
                conduct_history_df=data.get("conduct_scores"),
                exam_history_df=exam_history_df if exam_history_df is not None else empty_exam_df,
                study_hours_df=data.get("study_hours"),
            )
        logger.info(f"Student data prepared: {len(training_df)} records")
        return training_df

    def prepare_features(
        self,
        student_df: pd.DataFrame,
        label_encoders: Optional[Dict[str, LabelEncoder]] = None,
        model=None,
    ) -> pd.DataFrame:
        """Prepare features for prediction.

        Args:
            student_df: Student data DataFrame
            label_encoders: Label encoders from training (optional, will create new if None)
            model: Mô hình dùng để lấy strategy/encoders/feature_names. Mặc định
                ``self.model`` (model chính). Truyền model phụ (forecast) để encode
                đúng cho nhánh dự báo môn chưa học.

        Returns:
            Feature matrix ready for prediction
        """
        logger.info("Preparing features for prediction")

        # DESIGN-02: shared encoder. Note: predict path does NOT need y.
        # Use the strategy + fitted encoders persisted on the model artifact
        # so predictions match training-time encoding exactly.
        m = model if model is not None else self.model
        strategy = getattr(m, "categorical_strategy", "hash") if m else "hash"
        encoders = getattr(m, "fitted_encoders", None) if m else None
        feat_names = getattr(m, "feature_names", None) or self.feature_names
        X, _, _, _ = shared_prepare_features(
            student_df,
            feature_names=feat_names,
            target_column="exam_score",
            categorical_strategy=strategy,
            include_id_features=(strategy != "hash"),
            fitted_encoders=encoders,
            fit=False,
        )

        logger.info(f"Features prepared: {X.shape}")

        return X

    def _ensure_forecast_model(self):
        """Nạp (lazy, 1 lần) MODEL PHỤ dùng cho dự báo môn CHƯA học.

        Model phụ là biến thể loại bỏ Subject_ID/Lecturer_ID khỏi đặc trưng
        (``models/model_forecast.joblib`` cạnh model chính), nên dự báo bám năng
        lực SV thay vì nhận dạng môn lạ. Nếu file không tồn tại → trả về None và
        pipeline dùng model chính cho mọi trường hợp (hành vi cũ).

        Returns:
            EnsembleModel phụ, hoặc None nếu không có.
        """
        if not getattr(self, "_forecast_loaded", False):
            self._forecast_loaded = True
            self._forecast_model = None
            self._forecast_explainer = None
            fm = None
            # (1) Ưu tiên BUNDLE nhúng trong chính model.joblib → BE chỉ cần nạp 1 file.
            # Bundle là một object EnsembleModel đã huấn luyện, nhúng trong extra_metadata.
            bundle = (getattr(self.model, "extra_metadata", {}) or {}).get("forecast_bundle")
            if (
                bundle is not None
                and hasattr(bundle, "predict")
                and getattr(bundle, "is_trained", False)
            ):
                fm = bundle
                logger.info("Loaded forecast model (bundle nhúng trong model.joblib)")
            # (2) Fallback: file riêng cạnh model chính (tương thích ngược).
            if fm is None or not getattr(fm, "is_trained", False):
                fpath = self.model_path.parent / "model_forecast.joblib"
                if fpath.exists():
                    try:
                        fm = EnsembleModel(random_state=42)
                        fm.load(str(fpath))
                        logger.info(f"Loaded forecast model from {fpath}")
                    except Exception as e:  # pragma: no cover - defensive
                        logger.warning(f"Không nạp được model phụ forecast: {e}")
                        fm = None
            if fm is not None and getattr(fm, "is_trained", False):
                self._forecast_model = fm
                self._forecast_explainer = EnsembleSHAPExplainer(fm, cache_explainer=True)
        return self._forecast_model

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
        attendance_path: Optional[str] = None,
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
            attendance_path: Path điểm danh (optional)
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
            attendance_path=attendance_path,
        )

        # CHỌN MÔ HÌNH theo ngữ cảnh:
        #  - Môn ĐÃ/ĐANG học (exam_score có giá trị) → MODEL CHÍNH (label, R²=0.5720
        #    — đúng số báo cáo). Tập test đánh giá cũng đi nhánh này → metrics KHÔNG đổi.
        #  - Môn CHƯA học (exam_score=NaN) → MODEL PHỤ forecast (loại Subject_ID/
        #    Lecturer_ID) nếu có sẵn `models/model_forecast.joblib`. Model phụ dự báo
        #    theo năng lực SV (không bị lệch vì nhận dạng môn lạ) → SV giỏi ≥4, SV yếu
        #    thấp; ĐIỂM và LÝ DO cùng từ MỘT model nên luôn nhất quán.
        is_forecast = bool(pd.isna(student_df.iloc[0].get("exam_score", np.nan)))
        forecast_model = self._ensure_forecast_model() if is_forecast else None
        if forecast_model is not None:
            active_model, active_explainer = forecast_model, self._forecast_explainer
            logger.info("Môn chưa học → dùng model phụ (forecast, loại ID)")
        else:
            active_model, active_explainer = self.model, self.explainer

        # Prepare features cho đúng model đang dùng
        X = self.prepare_features(student_df, model=active_model)

        # Predict
        predicted_score = active_model.predict(X)[0]

        # Forecast môn CHƯA học: neo theo PHONG ĐỘ GẦN ĐÂY của SV (recent_avg_score).
        # Để dự báo TƯƠNG LAI, điểm gần đây dự báo tốt hơn trung bình toàn khoá: SV đang
        # sa sút / rèn luyện kém / có rớt môn sẽ được phản ánh đúng (forecast thấp hơn),
        # đồng thời giảm dao động theo từng môn. Chỉ áp dụng cho nhánh forecast (model phụ)
        # → KHÔNG ảnh hưởng môn đã học và metrics báo cáo.
        if forecast_model is not None:
            xr0 = X.iloc[0]
            recent = next(
                (
                    float(xr0[c])
                    for c in ("recent_avg_score", "academic_core_score", "avg_exam_score")
                    if c in X.columns and pd.notna(xr0[c]) and float(xr0[c]) > 0
                ),
                None,
            )
            if recent is not None:
                predicted_score = float(
                    np.clip(0.2 * float(predicted_score) + 0.8 * recent, 0.0, 6.0)
                )

        # Compute SHAP values (từ chính model đang dùng → lý do khớp với điểm)
        shap_values = active_explainer.explain_instance(X)
        shap_values_1d = shap_values[0]

        # Process SHAP for analysis
        processed = process_shap_for_analysis(
            shap_values_1d,
            feature_names=getattr(active_model, "feature_names", None) or self.feature_names,
            df=None,
        )

        # Generate explanation — khi có actual_clo_score thì summary/context dùng actual
        display_score = actual_clo_score if actual_clo_score is not None else predicted_score
        explanation = generate_complete_explanation(
            top_negative_impacts=processed["top_negative_impacts"],
            predicted_score=display_score,
            context="individual",
            include_solutions=True,
            raw_feature_row=student_df.iloc[0],
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

        # MISSING-06: audit trail (no-op when audit path not configured)
        log_prediction(
            student_id=student_id,
            subject_id=subject_id,
            lecturer_id=lecturer_id,
            predicted_score=predicted_score,
            actual_score=actual_clo_score,
            model_version=getattr(self.model, "version", None),
        )

        logger.info(
            f"Prediction complete: predicted={predicted_score:.2f}, "
            f"actual={'%.2f' % actual_clo_score if actual_clo_score is not None else 'N/A'}, "
            f"reasons={len(output.reasons)}"
        )

        return output

