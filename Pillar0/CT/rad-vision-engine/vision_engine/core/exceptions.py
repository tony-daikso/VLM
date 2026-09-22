"""
Custom exceptions for vision-engine
"""


class VisionEngineError(Exception):
    """Base exception for all vision-engine errors"""

    pass


class ConfigurationError(VisionEngineError):
    """Raised when there's an error in configuration"""

    pass


class InputError(VisionEngineError):
    """Raised when there's an error with input data or files"""

    pass


class ProcessingError(VisionEngineError):
    """Raised when there's an error during image processing"""

    pass


class ExportError(VisionEngineError):
    """Raised when there's an error during export/output generation"""

    pass
