"""Unit tests for feature builder."""

import pandas as pd
import pytest

from ml_clo.features.feature_builder import (
    build_academic_history_features,
    build_all_features,
    build_conduct_features,
    build_study_hours_features,
)


class TestBuildConductFeatures:
    """Test build_conduct_features function (df cần Student_ID, year)."""

    def test_build_conduct_features_success(self):
        """Test successful conduct features building (numeric Student_ID)."""
        df = pd.DataFrame({
            "Student_ID": [19050006, 19050007],
            "year": [2023, 2023],
        })
        conduct_df = pd.DataFrame({
            "Student_ID": [19050006, 19050007],
            "conduct_score": [85.0, 90.0],
            "year": [2023, 2023],
        })
        result = build_conduct_features(df, conduct_df)
        assert "avg_conduct_score" in result.columns

    def test_build_conduct_features_no_history(self):
        """Test building conduct features with empty history."""
        df = pd.DataFrame({
            "Student_ID": [19050006, 19050007],
            "year": [2024, 2024],
        })
        empty_conduct = pd.DataFrame(columns=["Student_ID", "conduct_score", "year"])
        result = build_conduct_features(df, empty_conduct)
        assert len(result) == 2


class TestBuildAcademicHistoryFeatures:
    """Test build_academic_history_features function (cần df + exam_history_df)."""

    def test_build_academic_history_features_success(self, sample_exam_scores):
        """Test successful academic history features building."""
        df = sample_exam_scores[["Student_ID", "year"]].drop_duplicates()
        result = build_academic_history_features(df, exam_history_df=sample_exam_scores)
        assert "total_subjects" in result.columns
        assert "avg_exam_score" in result.columns
        assert len(result) == len(df)

    def test_build_academic_history_features_no_history(self, sample_exam_scores):
        """Test with empty exam history."""
        df = pd.DataFrame({"Student_ID": [19050006], "year": [2024]})
        empty_exam = pd.DataFrame(columns=["Student_ID", "year", "exam_score"])
        result = build_academic_history_features(df, exam_history_df=empty_exam)
        assert len(result) == 1
        assert "total_subjects" in result.columns


class TestBuildStudyHoursFeatures:
    """Test build_study_hours_features function (df cần Student_ID, year; study_hours cần accumulated_study_hours)."""

    def test_build_study_hours_features_success(self):
        """Test successful study hours features building."""
        df = pd.DataFrame({
            "Student_ID": [19050006, 19050007],
            "year": [2024, 2024],
        })
        study_hours_df = pd.DataFrame({
            "Student_ID": [19050006],
            "year": [2024],
            "accumulated_study_hours": [100.0],
        })
        result = build_study_hours_features(df, study_hours_df)
        assert len(result) == len(df)

    def test_build_study_hours_features_no_history(self):
        """Test with empty study hours (cần df với cột year)."""
        df = pd.DataFrame({
            "Student_ID": [19050006, 19050007],
            "year": [2024, 2024],
        })
        empty_hours = pd.DataFrame(columns=["Student_ID", "year", "accumulated_study_hours"])
        result = build_study_hours_features(df, empty_hours)
        assert len(result) == len(df)


class TestBuildAllFeatures:
    """Test build_all_features function."""

    def test_build_all_features_success(self, sample_exam_scores, sample_conduct_scores):
        """Test successful building of all features."""
        result = build_all_features(
            sample_exam_scores,
            conduct_history_df=sample_conduct_scores,
            exam_history_df=sample_exam_scores,
        )

        assert len(result) == len(sample_exam_scores)
        assert len(result.columns) > len(sample_exam_scores.columns)

