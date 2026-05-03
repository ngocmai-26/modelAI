"""Ensemble model combining Random Forest and Gradient Boosting."""

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from ml_clo.config.model_config import (
    ENSEMBLE_CONFIG,
    GRADIENT_BOOSTING_CONFIG,
    RANDOM_FOREST_CONFIG,
)
from ml_clo.models.base_model import BaseModel
from ml_clo.utils.exceptions import ModelLoadError
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleModel(BaseModel):
    """Ensemble model combining Random Forest and Gradient Boosting.

    Uses weighted average of predictions from both models, with weights
    determined by validation performance.
    """

    def __init__(
        self,
        version: Optional[str] = None,
        random_state: int = 42,
        rf_config: Optional[Dict] = None,
        gb_config: Optional[Dict] = None,
        ensemble_config: Optional[Dict] = None,
    ):
        """Initialize ensemble model.

        Args:
            version: Model version string (default: None, auto-generated)
            random_state: Random seed for reproducibility (default: 42)
            rf_config: Random Forest hyperparameters (default: None, uses default)
            gb_config: Gradient Boosting hyperparameters (default: None, uses default)
            ensemble_config: Ensemble configuration (default: None, uses default)
        """
        super().__init__(
            model_name="Ensemble",
            version=version,
            random_state=random_state,
        )

        self.rf_config = rf_config or RANDOM_FOREST_CONFIG.copy()
        self.gb_config = gb_config or GRADIENT_BOOSTING_CONFIG.copy()
        self.ensemble_config = ensemble_config or ENSEMBLE_CONFIG.copy()

        # Remove random_state from configs to avoid duplicate argument
        rf_config_clean = {k: v for k, v in self.rf_config.items() if k != "random_state"}
        gb_config_clean = {k: v for k, v in self.gb_config.items() if k != "random_state"}

        # Initialize sub-models
        self.rf_model = RandomForestRegressor(
            **rf_config_clean,
            random_state=random_state,
        )
        self.gb_model = GradientBoostingRegressor(
            **gb_config_clean,
            random_state=random_state,
        )

        # Ensemble weights (will be set during training)
        self.rf_weight = 0.5
        self.gb_weight = 0.5

        # Store predictions for ensemble
        self.model = None  # Will be set to a composite model

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Train ensemble model.

        Trains both Random Forest and Gradient Boosting models, then determines
        optimal weights based on validation performance.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional, used for weight determination)
            y_val: Validation target (optional, used for weight determination)

        Returns:
            Dictionary of training metrics
        """
        logger.info(f"Training ensemble model (version {self.version})")

        # Store feature names
        self.feature_names = list(X_train.columns)

        # Train Random Forest
        logger.info("Training Random Forest model...")
        self.rf_model.fit(X_train, y_train)
        rf_train_pred = self.rf_model.predict(X_train)
        rf_train_mae = np.mean(np.abs(rf_train_pred - y_train))

        # Train Gradient Boosting
        logger.info("Training Gradient Boosting model...")
        self.gb_model.fit(X_train, y_train)
        gb_train_pred = self.gb_model.predict(X_train)
        gb_train_mae = np.mean(np.abs(gb_train_pred - y_train))

        # Determine weights based on validation performance
        if X_val is not None and y_val is not None:
            rf_val_pred = self.rf_model.predict(X_val)
            gb_val_pred = self.gb_model.predict(X_val)

            rf_val_mae = np.mean(np.abs(rf_val_pred - y_val))
            gb_val_mae = np.mean(np.abs(gb_val_pred - y_val))

            # Calculate weights inversely proportional to MAE
            # Lower MAE gets higher weight
            total_inv_mae = 1.0 / rf_val_mae + 1.0 / gb_val_mae
            self.rf_weight = (1.0 / rf_val_mae) / total_inv_mae
            self.gb_weight = (1.0 / gb_val_mae) / total_inv_mae

            # Apply min/max weight constraints
            min_weight = self.ensemble_config.get("min_weight", 0.1)
            max_weight = self.ensemble_config.get("max_weight", 0.9)

            self.rf_weight = np.clip(self.rf_weight, min_weight, max_weight)
            self.gb_weight = np.clip(self.gb_weight, min_weight, max_weight)

            # Normalize weights
            total_weight = self.rf_weight + self.gb_weight
            self.rf_weight /= total_weight
            self.gb_weight /= total_weight

            logger.info(
                f"Ensemble weights determined from validation: "
                f"RF={self.rf_weight:.3f}, GB={self.gb_weight:.3f}"
            )
            logger.info(f"Validation MAE: RF={rf_val_mae:.4f}, GB={gb_val_mae:.4f}")
        else:
            # Equal weights if no validation data
            logger.warning("No validation data provided, using equal weights")
            self.rf_weight = 0.5
            self.gb_weight = 0.5

        # Create composite model for saving
        self.model = {
            "rf_model": self.rf_model,
            "gb_model": self.gb_model,
            "rf_weight": self.rf_weight,
            "gb_weight": self.gb_weight,
        }

        # Calculate ensemble training metrics
        ensemble_train_pred = self.rf_weight * rf_train_pred + self.gb_weight * gb_train_pred
        ensemble_train_mae = np.mean(np.abs(ensemble_train_pred - y_train))
        ensemble_train_rmse = np.sqrt(np.mean((ensemble_train_pred - y_train) ** 2))
        ensemble_train_r2 = 1 - np.sum((y_train - ensemble_train_pred) ** 2) / np.sum(
            (y_train - y_train.mean()) ** 2
        )

        self.training_metrics = {
            "rf_train_mae": rf_train_mae,
            "gb_train_mae": gb_train_mae,
            "ensemble_train_mae": ensemble_train_mae,
            "ensemble_train_rmse": ensemble_train_rmse,
            "ensemble_train_r2": ensemble_train_r2,
            "rf_weight": self.rf_weight,
            "gb_weight": self.gb_weight,
        }

        if X_val is not None and y_val is not None:
            ensemble_val_pred = self.rf_weight * rf_val_pred + self.gb_weight * gb_val_pred
            ensemble_val_mae = np.mean(np.abs(ensemble_val_pred - y_val))
            ensemble_val_rmse = np.sqrt(np.mean((ensemble_val_pred - y_val) ** 2))
            ensemble_val_r2 = 1 - np.sum((y_val - ensemble_val_pred) ** 2) / np.sum(
                (y_val - y_val.mean()) ** 2
            )

            self.training_metrics.update({
                "rf_val_mae": rf_val_mae,
                "gb_val_mae": gb_val_mae,
                "ensemble_val_mae": ensemble_val_mae,
                "ensemble_val_rmse": ensemble_val_rmse,
                "ensemble_val_r2": ensemble_val_r2,
            })

        self.is_trained = True
        self.training_timestamp = pd.Timestamp.now().isoformat()

        # Persist encoding method and ensemble config snapshot with the model
        # so predict-time behavior matches train-time behavior regardless of
        # later changes to ENSEMBLE_CONFIG in source (NEW-02, DESIGN-10).
        self.extra_metadata = {
            # hash_v2: stable_hash_int uses mod = 2**31 - 1 (was 1e9 in v1)
            # — bumped to reduce birthday-paradox collisions on
            # high-cardinality columns.
            "encoding_method": "hash_v2",
            "ensemble_config": dict(self.ensemble_config),
            "rf_config": dict(self.rf_config),
            "gb_config": dict(self.gb_config),
        }

        logger.info(
            f"Ensemble model training complete. "
            f"Train MAE: {ensemble_train_mae:.4f}, Train R²: {ensemble_train_r2:.4f}"
        )

        return self.training_metrics

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions using ensemble.

        Args:
            X: Features for prediction

        Returns:
            Array of predictions

        Raises:
            ModelLoadError: If model is not trained
        """
        super().predict(X)  # Check if trained

        # BUG-05: Enforce training-time column order. Sklearn predicts by
        # positional index, so a DataFrame with the right columns in the
        # wrong order would silently predict on the wrong features.
        if self.feature_names is not None and isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_names if c not in X.columns]
            if missing:
                raise ValueError(
                    f"Missing required feature columns: {missing[:10]}"
                    + ("..." if len(missing) > 10 else "")
                )
            X = X[self.feature_names]

        # Get predictions from both models
        rf_pred = self.rf_model.predict(X)
        gb_pred = self.gb_model.predict(X)

        max_gb = float(self.ensemble_config.get("gb_low_anomaly_max_gb", 0.75))
        min_gap = float(self.ensemble_config.get("gb_low_anomaly_min_gap", 0.35))
        br = float(self.ensemble_config.get("gb_low_anomaly_rf_blend", 0.88))
        low_gb = gb_pred < max_gb
        gap_ok = (rf_pred - gb_pred) > min_gap
        anomaly = low_gb & gap_ok
        gb_use = np.where(
            anomaly,
            np.clip(br * rf_pred + (1.0 - br) * gb_pred, 0.0, 6.0),
            gb_pred,
        )

        # Weighted average; clip to CLO scale [0, 6]
        ensemble_pred = self.rf_weight * rf_pred + self.gb_weight * gb_use
        return np.clip(ensemble_pred, 0.0, 6.0)

    def set_weights(self, rf_weight: float, gb_weight: float) -> None:
        """Override ensemble weights after loading a trained model.

        DESIGN-08: Allow operators to re-balance the ensemble post-hoc (e.g.
        after observing drift in one sub-model on recent data) without
        retraining. Weights are normalized to sum to 1 and clipped to the
        min/max bounds from ensemble_config.

        Args:
            rf_weight: Desired Random Forest weight (>0)
            gb_weight: Desired Gradient Boosting weight (>0)

        Raises:
            ValueError: If either weight is non-positive.
        """
        if rf_weight <= 0 or gb_weight <= 0:
            raise ValueError(
                f"Weights must be positive, got rf_weight={rf_weight}, gb_weight={gb_weight}"
            )

        min_w = float(self.ensemble_config.get("min_weight", 0.1))
        max_w = float(self.ensemble_config.get("max_weight", 0.9))
        rf_w = float(np.clip(rf_weight, min_w, max_w))
        gb_w = float(np.clip(gb_weight, min_w, max_w))
        total = rf_w + gb_w
        self.rf_weight = rf_w / total
        self.gb_weight = gb_w / total

        if isinstance(self.model, dict):
            self.model["rf_weight"] = self.rf_weight
            self.model["gb_weight"] = self.gb_weight

        logger.info(
            f"Ensemble weights updated: RF={self.rf_weight:.3f}, GB={self.gb_weight:.3f}"
        )

    def predict_with_uncertainty(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """MISSING-03: Predict with per-sample uncertainty estimate.

        Uses the standard deviation across the Random Forest sub-estimators
        as the uncertainty signal (a well-known cheap proxy for predictive
        variance on tree ensembles). Gradient Boosting cannot expose
        per-tree predictions in the same way, so its contribution is
        treated as point-estimate-only.

        Returns:
            Dict with keys:
              - ``prediction``: Final clipped ensemble prediction (same as
                :meth:`predict`).
              - ``rf_std``: Per-sample stdev across RF trees.
              - ``confidence_interval_low`` / ``confidence_interval_high``:
                Approximate ±2σ band around the prediction, clipped to [0,6].
        """
        super().predict(X)  # ensures trained
        if self.feature_names is not None and isinstance(X, pd.DataFrame):
            X = X[self.feature_names]

        # Per-tree predictions: shape (n_trees, n_samples)
        per_tree = np.stack(
            [tree.predict(X) for tree in self.rf_model.estimators_], axis=0
        )
        rf_std = per_tree.std(axis=0)

        prediction = self.predict(X)
        ci_low = np.clip(prediction - 2.0 * rf_std, 0.0, 6.0)
        ci_high = np.clip(prediction + 2.0 * rf_std, 0.0, 6.0)

        return {
            "prediction": prediction,
            "rf_std": rf_std,
            "confidence_interval_low": ci_low,
            "confidence_interval_high": ci_high,
        }

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get ensemble feature importance.

        Returns weighted average of feature importances from both models.

        Returns:
            Dictionary mapping feature names to importance scores, or None if not available
        """
        if not self.is_trained:
            return None

        rf_importance = self.rf_model.feature_importances_
        gb_importance = self.gb_model.feature_importances_

        # Weighted average
        ensemble_importance = (
            self.rf_weight * rf_importance + self.gb_weight * gb_importance
        )

        if self.feature_names is not None:
            return dict(zip(self.feature_names, ensemble_importance))
        else:
            return {f"feature_{i}": imp for i, imp in enumerate(ensemble_importance)}

    def load(self, file_path: str) -> None:
        """Load ensemble model from file.

        Args:
            file_path: Path to model file

        Raises:
            ModelLoadError: If file cannot be loaded
        """
        super().load(file_path)

        # Extract sub-models from composite model
        if isinstance(self.model, dict):
            self.rf_model = self.model["rf_model"]
            self.gb_model = self.model["gb_model"]
            self.rf_weight = self.model.get("rf_weight", 0.5)
            self.gb_weight = self.model.get("gb_weight", 0.5)
        else:
            raise ModelLoadError("Invalid model structure in saved file")

        # NEW-02: Enforce that the model was trained with the current
        # categorical encoding scheme. Models saved before hash encoding
        # was introduced do not have this field and are incompatible with
        # the current prediction pipeline.
        encoding_method = self.extra_metadata.get("encoding_method")
        if encoding_method is None:
            raise ModelLoadError(
                f"Model file {file_path} has no 'encoding_method' metadata. "
                "It was likely trained with the legacy LabelEncoder-based "
                "pipeline and is incompatible with the current hash-based "
                "encoding. Please retrain the model."
            )
        if encoding_method not in ("hash_v2",):
            raise ModelLoadError(
                f"Unsupported encoding_method '{encoding_method}' in model "
                f"{file_path}. Expected 'hash_v2'. Please retrain the model."
            )

        # DESIGN-10: Restore ensemble_config snapshot so predict() uses the
        # same anomaly thresholds that were in effect at training time.
        saved_ensemble_config = self.extra_metadata.get("ensemble_config")
        if saved_ensemble_config:
            self.ensemble_config = dict(saved_ensemble_config)
            logger.info(
                "Restored ensemble_config snapshot from model metadata "
                "(gb_low_anomaly_max_gb=%s, gb_low_anomaly_min_gap=%s, "
                "gb_low_anomaly_rf_blend=%s)",
                self.ensemble_config.get("gb_low_anomaly_max_gb"),
                self.ensemble_config.get("gb_low_anomaly_min_gap"),
                self.ensemble_config.get("gb_low_anomaly_rf_blend"),
            )
        else:
            logger.warning(
                "Model file has no saved ensemble_config; predict() will use "
                "ENSEMBLE_CONFIG from the current source, which may differ "
                "from training-time configuration."
            )

