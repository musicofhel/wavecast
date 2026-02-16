"""WaveCast exceptions."""


class WaveCastError(Exception):
    """Base exception for WaveCast."""


class DataError(WaveCastError):
    """Error fetching or processing data."""


class DataNotFoundError(DataError):
    """Requested data not found in cache or source."""


class DecompositionError(WaveCastError):
    """Error during wavelet decomposition."""


class ShapeletError(WaveCastError):
    """Error during shapelet discovery or matching."""


class ShapeletLibraryError(ShapeletError):
    """Error with shapelet library operations."""


class DTWError(WaveCastError):
    """Error during DTW computation."""


class FractalError(WaveCastError):
    """Error during fractal analysis."""


class ModelError(WaveCastError):
    """Error during model training or prediction."""


class ModelNotTrainedError(ModelError):
    """Model has not been trained yet."""


class PipelineError(WaveCastError):
    """Error during pipeline execution."""


class SAXError(WaveCastError):
    """Error during SAX transformation."""


class TokenizerError(WaveCastError):
    """Error during tokenization."""


class SequenceModelError(ModelError):
    """Error during sequence model training or prediction."""


class ConfigError(WaveCastError):
    """Configuration error."""
