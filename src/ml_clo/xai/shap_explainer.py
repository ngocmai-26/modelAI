"""SHAP explainer for ensemble model.

This module provides SHAP-based explainability for the ensemble model predictions.
Uses TreeExplainer for tree-based models (Random Forest and Gradient Boosting).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap

from ml_clo.config.xai_config import SHAP_CONFIG
from ml_clo.models.ensemble_model import EnsembleModel
from ml_clo.utils.exceptions import ModelLoadError
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleSHAPExplainer:
    """SHAP explainer for ensemble model.

    Computes SHAP values for ensemble predictions by combining SHAP values
    from Random Forest and Gradient Boosting models with ensemble weights.
    """

    def __init__(
        self,
        model: EnsembleModel,
        background_data: Optional[pd.DataFrame] = None,
        cache_explainer: bool = None,
    ):
        """Initialize SHAP explainer for ensemble model.

        Args:
            model: Trained EnsembleModel instance
            background_data: Background dataset for SHAP (optional, not used for TreeExplainer)
            cache_explainer: Whether to cache explainers (default: from config)

        Raises:
            ModelLoadError: If model is not trained
        """
        if not model.is_trained:
            raise ModelLoadError("Model must be trained before creating SHAP explainer")

        self.model = model
        self.cache_explainer = cache_explainer if cache_explainer is not None else SHAP_CONFIG.get("cache_explainer", True)

        # Initialize explainers for each sub-model
        self.rf_explainer: Optional[shap.TreeExplainer] = None
        self.gb_explainer: Optional[shap.TreeExplainer] = None

        # Cache for explainers (if enabled)
        self._rf_explainer_cached = None
        self._gb_explainer_cached = None

        logger.info("Initialized SHAP explainer for ensemble model")

    def _get_rf_explainer(self) -> shap.TreeExplainer:
        """Get or create Random Forest explainer.

        Returns:
            TreeExplainer for Random Forest model
        """
        if self.cache_explainer and self._rf_explainer_cached is not None:
            return self._rf_explainer_cached

        explainer = shap.TreeExplainer(self.model.rf_model)
        
        if self.cache_explainer:
            self._rf_explainer_cached = explainer

        return explainer

    def _get_gb_explainer(self) -> shap.TreeExplainer:
        """Get or create Gradient Boosting explainer.

        Returns:
            TreeExplainer for Gradient Boosting model
        """
        if self.cache_explainer and self._gb_explainer_cached is not None:
            return self._gb_explainer_cached

        explainer = shap.TreeExplainer(self.model.gb_model)
        
        if self.cache_explainer:
            self._gb_explainer_cached = explainer

        return explainer

    def explain_instance(
        self,
        X: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Compute SHAP values for a single instance or batch.

        Args:
            X: Feature values (single row or multiple rows)
            feature_names: Optional list of feature names (default: from model)

        Returns:
            SHAP values array with shape (n_samples, n_features)
        """
        if feature_names is None:
            feature_names = self.model.feature_names

        # Ensure X has correct feature order
        if isinstance(X, pd.DataFrame):
            X = X[feature_names]
        else:
            # Convert to DataFrame if needed
            X = pd.DataFrame(X, columns=feature_names)

        # Get explainers
        rf_explainer = self._get_rf_explainer()
        gb_explainer = self._get_gb_explainer()

        # Compute SHAP values for each model
        rf_shap_values = rf_explainer.shap_values(X)
        gb_shap_values = gb_explainer.shap_values(X)

        # Handle single instance vs batch
        if X.shape[0] == 1:
            rf_shap_values = rf_shap_values.reshape(1, -1)
            gb_shap_values = gb_shap_values.reshape(1, -1)

        # Combine SHAP values using ensemble weights
        ensemble_shap_values = (
            self.model.rf_weight * rf_shap_values
            + self.model.gb_weight * gb_shap_values
        )

        logger.debug(
            f"Computed SHAP values for {X.shape[0]} instance(s), "
            f"shape: {ensemble_shap_values.shape}"
        )

        return ensemble_shap_values

    def explain_batch(
        self,
        X: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Compute SHAP values for a batch of instances.

        Args:
            X: Feature values (multiple rows)
            feature_names: Optional list of feature names (default: from model)

        Returns:
            SHAP values array with shape (n_samples, n_features)
        """
        return self.explain_instance(X, feature_names)

    def get_feature_importance(
        self,
        X: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Get feature importance based on mean absolute SHAP values.

        Args:
            X: Feature values (can be single or multiple rows)
            feature_names: Optional list of feature names (default: from model)

        Returns:
            Dictionary mapping feature names to mean absolute SHAP values
        """
        shap_values = self.explain_instance(X, feature_names)

        if feature_names is None:
            feature_names = self.model.feature_names

        # Calculate mean absolute SHAP values
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

        importance_dict = {
            feature: float(importance)
            for feature, importance in zip(feature_names, mean_abs_shap)
        }

        return importance_dict

    def get_top_features(
        self,
        X: pd.DataFrame,
        top_n: int = 10,
        feature_names: Optional[List[str]] = None,
        use_absolute: bool = True,
    ) -> List[Tuple[str, float]]:
        """Get top N features by SHAP importance.

        Args:
            X: Feature values (can be single or multiple rows)
            top_n: Number of top features to return (default: 10)
            feature_names: Optional list of feature names (default: from model)
            use_absolute: Whether to use absolute SHAP values (default: True)

        Returns:
            List of tuples (feature_name, importance) sorted by importance
        """
        shap_values = self.explain_instance(X, feature_names)

        if feature_names is None:
            feature_names = self.model.feature_names

        # Calculate mean SHAP values (absolute if requested)
        if use_absolute:
            mean_shap = np.mean(np.abs(shap_values), axis=0)
        else:
            mean_shap = np.mean(shap_values, axis=0)

        # Sort by importance
        feature_importance = list(zip(feature_names, mean_shap))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        return feature_importance[:top_n]

