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

