"""Unit tests for feature groups."""

import pandas as pd
import pytest

from ml_clo.features.feature_groups import (
    get_all_existing_features,
    get_features_by_group,
    get_feature_groups,
    get_pedagogical_groups,
    group_features_by_pedagogy,
)


class TestGetFeatureGroups:
    """Test get_feature_groups function."""

    def test_get_feature_groups_success(self):
        """Test successful feature groups retrieval."""
        groups = get_feature_groups()

        assert isinstance(groups, dict)
        assert len(groups) > 0


class TestGetPedagogicalGroups:
    """Test get_pedagogical_groups function."""

    def test_get_pedagogical_groups_success(self):
        """Test successful pedagogical groups retrieval."""
        groups = get_pedagogical_groups()

        assert isinstance(groups, dict)
        assert len(groups) > 0


class TestGroupFeaturesByPedagogy:
    """Test group_features_by_pedagogy function."""

    def test_group_features_by_pedagogy_success(self):
        """Test successful feature grouping."""
        features = ["avg_conduct_score", "total_study_hours", "avg_exam_score"]

        grouped = group_features_by_pedagogy(features)

        assert isinstance(grouped, dict)
        assert len(grouped) > 0


class TestGetFeaturesByGroup:
    """Test get_features_by_group function (cần df)."""

    def test_get_features_by_group_success(self):
        """Test successful feature retrieval by group."""
        df = pd.DataFrame({"avg_conduct_score": [85.0], "total_study_hours": [100.0]})
        features = get_features_by_group("conduct", df)
        assert isinstance(features, list)


class TestGetAllExistingFeatures:
    """Test get_all_existing_features function (cần df)."""

    def test_get_all_existing_features_success(self):
        """Test successful retrieval of all existing features."""
        df = pd.DataFrame({
            "avg_conduct_score": [85.0],
            "total_study_hours": [100.0],
            "avg_exam_score": [4.5],
        })
        features = get_all_existing_features(df)
        assert isinstance(features, list)
        assert len(features) >= 0

