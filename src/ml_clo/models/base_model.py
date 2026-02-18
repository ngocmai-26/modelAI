"""Base model interface.

This module defines the abstract base class for all models in the CLO prediction system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime
import joblib
from pathlib import Path

import numpy as np
import pandas as pd

from ml_clo.utils.exceptions import ModelLoadError
from ml_clo.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """Abstract base class for all CLO prediction models.

    All models must inherit from this class and implement the abstract methods.
    """

    def __init__(
        self,
        model_name: str,
        version: Optional[str] = None,
        random_state: int = 42,
    ):
        """Initialize base model.

        Args:
            model_name: Name of the model (e.g., "RandomForest", "GradientBoosting")
            version: Model version string (default: None, auto-generated)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.model_name = model_name
        self.version = version or self._generate_version()
        self.random_state = random_state
        self.model: Optional[Any] = None
        self.is_trained = False
        self.feature_names: Optional[list] = None
        self.training_metrics: Dict[str, float] = {}
        self.training_timestamp: Optional[str] = None

    def _generate_version(self) -> str:
        """Generate model version string.

        Returns:
            Version string in format "v1.0_TIMESTAMP"
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"v1.0_{timestamp}"

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> Dict[str, float]:
        """Train the model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            Dictionary of training metrics
        """
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions.

        Args:
            X: Features for prediction

        Returns:
            Array of predictions

        Raises:
            ModelLoadError: If model is not trained
        """
        if not self.is_trained or self.model is None:
            raise ModelLoadError(f"Model {self.model_name} is not trained. Call train() first.")

    def save(self, file_path: str) -> None:
        """Save model to file.

        Args:
            file_path: Path to save model file

        Raises:
            ModelLoadError: If model is not trained
        """
        if not self.is_trained or self.model is None:
            raise ModelLoadError(f"Model {self.model_name} is not trained. Cannot save.")

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Save model and metadata
        model_data = {
            "model": self.model,
            "model_name": self.model_name,
            "version": self.version,
            "random_state": self.random_state,
            "is_trained": self.is_trained,
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "training_timestamp": self.training_timestamp,
        }

        joblib.dump(model_data, file_path)
        logger.info(f"Saved model {self.model_name} (version {self.version}) to {file_path}")

    def load(self, file_path: str) -> None:
        """Load model from file.

        Args:
            file_path: Path to model file

        Raises:
            ModelLoadError: If file cannot be loaded
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ModelLoadError(f"Model file not found: {file_path}")

        try:
            model_data = joblib.load(file_path)

            self.model = model_data["model"]
            self.model_name = model_data.get("model_name", self.model_name)
            self.version = model_data.get("version", self.version)
            self.random_state = model_data.get("random_state", self.random_state)
            self.is_trained = model_data.get("is_trained", False)
            self.feature_names = model_data.get("feature_names")
            self.training_metrics = model_data.get("training_metrics", {})
            self.training_timestamp = model_data.get("training_timestamp")

            logger.info(f"Loaded model {self.model_name} (version {self.version}) from {file_path}")
        except Exception as e:
            raise ModelLoadError(f"Failed to load model from {file_path}: {e}")

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance (if available).

        Returns:
            Dictionary mapping feature names to importance scores, or None if not available
        """
        if not self.is_trained or self.model is None:
            return None

        # Try to get feature_importances_ attribute
        if hasattr(self.model, "feature_importances_"):
            if self.feature_names is not None:
                return dict(zip(self.feature_names, self.model.feature_importances_))
            else:
                # Use indices if feature names not available
                return {f"feature_{i}": imp for i, imp in enumerate(self.model.feature_importances_)}

        return None

    def get_info(self) -> Dict[str, Any]:
        """Get model information.

        Returns:
            Dictionary with model metadata
        """
        return {
            "model_name": self.model_name,
            "version": self.version,
            "is_trained": self.is_trained,
            "random_state": self.random_state,
            "feature_count": len(self.feature_names) if self.feature_names else None,
            "training_metrics": self.training_metrics,
            "training_timestamp": self.training_timestamp,
        }


