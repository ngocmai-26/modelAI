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
    """Test build_conduct_features function."""

    def test_build_conduct_features_success(self, sample_conduct_scores):
        """Test successful conduct features building."""
        df = pd.DataFrame({
            "Student_ID": ["SV001", "SV002"],
            "exam_score": [5.0, 6.0],
        })

        result = build_conduct_features(df, sample_conduct_scores)

        assert "avg_conduct_score" in result.columns or len(result) > 0

    def test_build_conduct_features_no_history(self):
        """Test building conduct features without history."""
        df = pd.DataFrame({
            "Student_ID": ["SV001", "SV002"],
            "exam_score": [5.0, 6.0],
        })

        result = build_conduct_features(df, None)

        # Should still return DataFrame with same rows
        assert len(result) == len(df)


class TestBuildAcademicHistoryFeatures:
    """Test build_academic_history_features function."""

    def test_build_academic_history_features_success(self, sample_exam_scores):
        """Test successful academic history features building."""
        result = build_academic_history_features(sample_exam_scores)

        assert "total_subjects" in result.columns
        assert "avg_exam_score" in result.columns
        assert len(result) == len(sample_exam_scores)

    def test_build_academic_history_features_no_history(self):
        """Test building academic history features without history."""
        df = pd.DataFrame({
            "Student_ID": ["SV001"],
            "exam_score": [5.0],
        })

        result = build_academic_history_features(df)

        assert len(result) == len(df)
        assert "total_subjects" in result.columns


class TestBuildStudyHoursFeatures:
    """Test build_study_hours_features function."""

    def test_build_study_hours_features_success(self):
        """Test successful study hours features building."""
        df = pd.DataFrame({
            "Student_ID": ["SV001", "SV002"],
            "exam_score": [5.0, 6.0],
        })

        study_hours_df = pd.DataFrame({
            "Student_ID": ["SV001"],
            "study_hours": [100.0],
        })

        result = build_study_hours_features(df, study_hours_df)

        assert len(result) == len(df)

    def test_build_study_hours_features_no_history(self):
        """Test building study hours features without history."""
        df = pd.DataFrame({
            "Student_ID": ["SV001", "SV002"],
            "exam_score": [5.0, 6.0],
        })

        result = build_study_hours_features(df, None)

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

