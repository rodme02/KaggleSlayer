"""
Custom exceptions for KaggleSlayer pipeline.
"""


class KaggleSlayerError(Exception):
    """Base exception for all KaggleSlayer errors."""
    pass


class DataLoadError(KaggleSlayerError):
    """Raised when data loading fails."""
    pass


class DataValidationError(KaggleSlayerError):
    """Raised when data validation fails."""
    pass


class FeatureEngineeringError(KaggleSlayerError):
    """Raised when feature engineering fails."""
    pass


class ModelTrainingError(KaggleSlayerError):
    """Raised when model training fails."""
    pass


class ConfigurationError(KaggleSlayerError):
    """Raised when configuration is invalid."""
    pass


class KaggleAPIError(KaggleSlayerError):
    """Raised when Kaggle API operations fail."""
    pass


class LLMError(KaggleSlayerError):
    """Raised when LLM operations fail."""
    pass
