"""ML CLO Prediction & XAI Library.

A Python library for predicting CLO scores and providing explainable AI insights.

Backend usage:
    from ml_clo import TrainingPipeline, PredictionPipeline, AnalysisPipeline
    from ml_clo.outputs.schemas import IndividualAnalysisOutput, ClassAnalysisOutput
"""

__version__ = "0.1.0"

from ml_clo.pipelines import (
    AnalysisPipeline,
    PredictionPipeline,
    TrainingPipeline,
)

__all__ = [
    "__version__",
    "AnalysisPipeline",
    "PredictionPipeline",
    "TrainingPipeline",
]

