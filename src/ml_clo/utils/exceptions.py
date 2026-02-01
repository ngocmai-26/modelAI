"""Custom exceptions for ML CLO library."""


class MLCLOError(Exception):
    """Base exception for ML CLO library."""

    pass


class DataLoadError(MLCLOError):
    """Raised when data loading fails."""

    pass


class DataValidationError(MLCLOError):
    """Raised when data validation fails."""

    pass


class ModelLoadError(MLCLOError):
    """Raised when model loading fails."""

    pass


class ModelSaveError(MLCLOError):
    """Raised when model saving fails."""

    pass


class PredictionError(MLCLOError):
    """Raised when prediction fails."""

    pass


class ConfigurationError(MLCLOError):
    """Raised when configuration is invalid."""

    pass

