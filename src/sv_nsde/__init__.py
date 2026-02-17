"""
SV-NSDE: Semantic Volatility-Modulated Neural SDE for Crisis Dynamics

A neural point process model that combines Heston stochastic volatility
with neural SDEs to model information diffusion during crises.

Based on: "Modeling Crisis Dynamics with Volatility-Modulated Neural SDEs"
"""

from .encoder import SemanticEncoder
from .sde import HestonSDEFunc, NeuralHestonSDE
from .intensity import DualChannelIntensity
from .decoder import SemanticDecoder
from .model import SVNSDE, SVNSDELite, SVNSDEConfig
from .data import WeiboCOVLoader, CascadeData, CascadeDataset, generate_synthetic_weibo_data
from .baselines import (
    RMTPP, NeuralHawkes, LatentODE, NeuralJumpSDE,
    SVNSDENoVolatilityChannel, SVNSDEDeterministicVolatility,
    BASELINE_MODELS, get_baseline,
)
from .evaluate import Evaluator, EvaluationMetrics, VolatilityAnalyzer, run_full_evaluation
from .experiment import (
    TrainConfig,
    precompute_bert_embeddings,
    train_model,
    train_all_baselines,
    run_comparison,
)

__all__ = [
    # Core model
    "SemanticEncoder",
    "HestonSDEFunc",
    "NeuralHestonSDE",
    "DualChannelIntensity",
    "SemanticDecoder",
    "SVNSDE",
    "SVNSDELite",
    "SVNSDEConfig",
    # Data
    "WeiboCOVLoader",
    "CascadeData",
    "CascadeDataset",
    "generate_synthetic_weibo_data",
    # Baselines
    "RMTPP",
    "NeuralHawkes",
    "LatentODE",
    "NeuralJumpSDE",
    "SVNSDENoVolatilityChannel",
    "SVNSDEDeterministicVolatility",
    "BASELINE_MODELS",
    "get_baseline",
    # Evaluation
    "Evaluator",
    "EvaluationMetrics",
    "VolatilityAnalyzer",
    "run_full_evaluation",
    # Experiment workflows
    "TrainConfig",
    "precompute_bert_embeddings",
    "train_model",
    "train_all_baselines",
    "run_comparison",
]

__version__ = "0.1.0"
