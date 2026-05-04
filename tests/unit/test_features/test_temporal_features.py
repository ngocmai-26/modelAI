"""Tests for ml_clo.features.temporal_features (Phase 2)."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from ml_clo.features.temporal_features import (
    build_temporal_attendance_features,
    merge_temporal_attendance_features,
)


def _attendance_rows(student_id, statuses, year="2024-2025", start=datetime(2024, 9, 1)):
    """Build a list of attendance rows, one per week."""
    rows = []
    for i, status in enumerate(statuses):
        rows.append({
            "MSSV": student_id,
            "Ngày học": start + timedelta(days=7 * i),
            "Buổi thứ": 1,
            "Điểm danh": status,
            "Năm học": year,
            "Học kì": 1,
        })
    return rows


class TestBuildTemporalAttendanceFeatures:
    def test_empty_input_returns_empty_with_keys(self):
        out = build_temporal_attendance_features(pd.DataFrame())
        assert "Student_ID" in out.columns
        assert "year" in out.columns
        assert "attendance_slope_3w" in out.columns
        assert len(out) == 0

    def test_none_input_safe(self):
        out = build_temporal_attendance_features(None)
        assert len(out) == 0

    def test_perfect_attendance_zero_slope(self):
        df = pd.DataFrame(_attendance_rows("19050001", ["Có"] * 8))
        out = build_temporal_attendance_features(df)
        assert len(out) == 1
        assert out.iloc[0]["attendance_slope_full"] == pytest.approx(0.0)
        assert out.iloc[0]["attendance_volatility"] == pytest.approx(0.0)
        assert out.iloc[0]["late_streak_max"] == 0
        assert out.iloc[0]["early_dropoff_flag"] == 0

    def test_negative_slope_when_attendance_drops(self):
        # 3 weeks present then 3 absent
        df = pd.DataFrame(_attendance_rows("19050002", ["Có", "Có", "Có", "Vắng", "Vắng", "Vắng"]))
        out = build_temporal_attendance_features(df)
        assert out.iloc[0]["attendance_slope_full"] < 0
        assert out.iloc[0]["attendance_slope_3w"] <= 0  # last 3 weeks are absent

    def test_volatility_nonzero_when_alternating(self):
        df = pd.DataFrame(_attendance_rows("X", ["Có", "Vắng", "Có", "Vắng", "Có", "Vắng"]))
        out = build_temporal_attendance_features(df)
        assert out.iloc[0]["attendance_volatility"] > 0

    def test_late_streak_counts_consecutive(self):
        # 4 absences in a row
        df = pd.DataFrame(_attendance_rows("X", ["Có", "Vắng", "Vắng", "Vắng", "Vắng", "Có"]))
        out = build_temporal_attendance_features(df)
        assert out.iloc[0]["late_streak_max"] >= 4

    def test_early_dropoff_flag_set_when_second_half_drops(self):
        # First half full attendance, second half absent
        df = pd.DataFrame(_attendance_rows("X", ["Có"] * 4 + ["Vắng"] * 4))
        out = build_temporal_attendance_features(df)
        assert out.iloc[0]["early_dropoff_flag"] == 1

    def test_no_early_dropoff_when_consistent(self):
        df = pd.DataFrame(_attendance_rows("X", ["Có"] * 8))
        out = build_temporal_attendance_features(df)
        assert out.iloc[0]["early_dropoff_flag"] == 0

    def test_groups_by_student_year(self):
        rows = (
            _attendance_rows("A", ["Có"] * 4, year="2023-2024", start=datetime(2023, 9, 1))
            + _attendance_rows("A", ["Có"] * 4, year="2024-2025", start=datetime(2024, 9, 1))
            + _attendance_rows("B", ["Có"] * 4, year="2024-2025", start=datetime(2024, 9, 1))
        )
        out = build_temporal_attendance_features(pd.DataFrame(rows))
        assert len(out) == 3
        assert set(out["Student_ID"]) == {"A", "A", "B"} or len(set(out["Student_ID"].unique())) == 2

    def test_unknown_status_treated_as_missing(self):
        # Mix valid + invalid statuses; row count after filtering is what matters
        rows = _attendance_rows("X", ["Có", "INVALID", "Có", "Có", "Có"])
        out = build_temporal_attendance_features(pd.DataFrame(rows))
        # Should still produce a row (valid statuses kept)
        assert len(out) == 1
        assert out.iloc[0]["num_weeks_observed"] == 4  # 4 valid weeks

    def test_num_weeks_observed_matches(self):
        df = pd.DataFrame(_attendance_rows("X", ["Có"] * 6))
        out = build_temporal_attendance_features(df)
        assert out.iloc[0]["num_weeks_observed"] == 6


class TestMergeTemporalAttendanceFeatures:
    def test_none_temporal_returns_input_unchanged(self):
        df = pd.DataFrame({"Student_ID": [1, 2], "year": [2024, 2024]})
        out = merge_temporal_attendance_features(df, None)
        assert len(out) == 2
        assert "attendance_slope_3w" not in out.columns

    def test_empty_temporal_returns_input_unchanged(self):
        df = pd.DataFrame({"Student_ID": [1, 2], "year": [2024, 2024]})
        out = merge_temporal_attendance_features(df, pd.DataFrame())
        assert len(out) == 2

    def test_merge_left_join_preserves_unmatched(self):
        df = pd.DataFrame({"Student_ID": [1, 2], "year": [2024, 2024]})
        temporal = pd.DataFrame({
            "Student_ID": [1],
            "year": [2024],
            "attendance_slope_3w": [0.5],
        })
        out = merge_temporal_attendance_features(df, temporal)
        assert len(out) == 2
        assert "attendance_slope_3w" in out.columns
        assert out.loc[out["Student_ID"] == 1, "attendance_slope_3w"].iloc[0] == 0.5
        # Unmatched row gets NaN
        assert pd.isna(out.loc[out["Student_ID"] == 2, "attendance_slope_3w"].iloc[0])

    def test_dtype_alignment_str_vs_int(self):
        df = pd.DataFrame({"Student_ID": [1, 2], "year": [2024, 2024]})
        temporal = pd.DataFrame({
            "Student_ID": ["1", "2"],
            "year": [2024, 2024],
            "attendance_slope_3w": [0.1, 0.2],
        })
        out = merge_temporal_attendance_features(df, temporal)
        # Should still match despite str vs int Student_ID
        assert out["attendance_slope_3w"].notna().all()
