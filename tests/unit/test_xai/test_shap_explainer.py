"""Unit tests for SHAP explainer."""

import numpy as np
import pandas as pd
import pytest

from ml_clo.xai.shap_explainer import EnsembleSHAPExplainer


class TestEnsembleSHAPExplainer:
    """Test EnsembleSHAPExplainer class."""

    def test_shap_explainer_init(self, trained_model):
        """Test SHAP explainer initialization."""
        explainer = EnsembleSHAPExplainer(trained_model, cache_explainer=True)

        assert explainer.model == trained_model
        assert explainer.cache_explainer

    def test_shap_explainer_explain_instance(self, trained_model, sample_features):
        """Test SHAP explanation for single instance."""
        X, _ = sample_features
        instance = X.iloc[[0]]

        explainer = EnsembleSHAPExplainer(trained_model, cache_explainer=True)
        shap_values = explainer.explain_instance(instance)

        assert shap_values is not None
        assert len(shap_values) == 1  # Single instance
        assert len(shap_values[0]) == len(instance.columns)  # One value per feature

    def test_shap_explainer_explain_batch(self, trained_model, sample_features):
        """Test SHAP explanation for batch."""
        X, _ = sample_features
        batch = X.iloc[:5]

        explainer = EnsembleSHAPExplainer(trained_model, cache_explainer=True)
        shap_values = explainer.explain_batch(batch)

        assert shap_values is not None
        assert len(shap_values) == len(batch)  # One per instance

    def test_shap_explainer_get_feature_importance(self, trained_model):
        """Test feature importance retrieval."""
        explainer = EnsembleSHAPExplainer(trained_model, cache_explainer=True)
        importance = explainer.get_feature_importance()

        assert importance is not None
        assert isinstance(importance, dict) or isinstance(importance, np.ndarray)

    def test_shap_explainer_get_top_features(self, trained_model):
        """Test top features retrieval."""
        explainer = EnsembleSHAPExplainer(trained_model, cache_explainer=True)
        top_features = explainer.get_top_features(top_k=5)

        assert top_features is not None
        assert len(top_features) <= 5

