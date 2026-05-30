"""
Custom exceptions for the multi-agent traffic prediction system.
"""


class TrafficPredictionError(Exception):
    """Base exception for all traffic prediction errors."""
    pass


class DataValidationError(TrafficPredictionError):
    """Raised when data validation fails."""
    pass


class FileNotFoundError(TrafficPredictionError):
    """Raised when required data files are not found."""
    pass


class LLMError(TrafficPredictionError):
    """Raised when LLM API calls fail or return invalid results."""
    pass


class TriangleValidationError(DataValidationError):
    """Raised when triangle data validation fails."""
    pass


class PredictionValidationError(DataValidationError):
    """Raised when prediction results fail validation."""
    pass


class AgentError(TrafficPredictionError):
    """Raised when agent processing fails."""
    pass


class ConfigurationError(TrafficPredictionError):
    """Raised when configuration is invalid."""
    pass