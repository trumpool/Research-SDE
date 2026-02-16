"""
Evaluation Script for SV-NSDE and Baselines

Implements the evaluation metrics from Section 4.3:
1. Time Prediction: RMSE (Root Mean Square Error)
2. Semantic Prediction: Cosine Similarity / MSE
3. Model Fit: Log-Likelihood on held-out test sets

Also includes:
- Volatility analysis for panic detection
- Per-phase evaluation (outbreak, plateau, decline)
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import logging
from tqdm import tqdm
from collections import defaultdict

from .baselines import BASELINE_MODELS, get_baseline
from .model import SVNSDE, SVNSDELite, SVNSDEConfig
from .data import CascadeData, CascadeDataset, WeiboCOVLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Evaluation Metrics
# =============================================================================

@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    # Time prediction
    time_rmse: float
    time_mae: float

    # Semantic prediction
    semantic_mse: float
    semantic_cosine: float

    # Model fit
    log_likelihood: float
    perplexity: float

    # Additional
    num_sequences: int
    num_events: int

    def to_dict(self) -> Dict:
        return {
            "time_rmse": self.time_rmse,
            "time_mae": self.time_mae,
            "semantic_mse": self.semantic_mse,
            "semantic_cosine": self.semantic_cosine,
            "log_likelihood": self.log_likelihood,
            "perplexity": self.perplexity,
            "num_sequences": self.num_sequences,
            "num_events": self.num_events,
        }


class MetricComputer:
    """Computes evaluation metrics for point process models."""

    def __init__(self, device: str = "cpu"):
        self.device = device

    def compute_time_metrics(
        self,
        predicted_times: torch.Tensor,
        actual_times: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Compute time prediction metrics.

        Args:
            predicted_times: Predicted next event times
            actual_times: Actual next event times

        Returns:
            (RMSE, MAE)
        """
        diff = predicted_times - actual_times
        rmse = torch.sqrt((diff ** 2).mean()).item()
        mae = torch.abs(diff).mean().item()
        return rmse, mae

    def compute_semantic_metrics(
        self,
        predicted_marks: torch.Tensor,
        actual_marks: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Compute semantic prediction metrics.

        Args:
            predicted_marks: Predicted semantic vectors
            actual_marks: Actual semantic vectors

        Returns:
            (MSE, Cosine Similarity)
        """
        mse = F.mse_loss(predicted_marks, actual_marks).item()

        # Cosine similarity
        pred_norm = F.normalize(predicted_marks, dim=-1)
        actual_norm = F.normalize(actual_marks, dim=-1)
        cosine = (pred_norm * actual_norm).sum(dim=-1).mean().item()

        return mse, cosine

    def compute_log_likelihood(
        self,
        model,
        event_times: torch.Tensor,
        event_marks: torch.Tensor,
        T: float,
    ) -> float:
        """
        Compute log-likelihood for a sequence.

        LL = Σ log λ(t_i) - ∫λ(t)dt
        """
        with torch.no_grad():
            loss_dict = model.compute_loss(event_times, event_marks, T)
            log_intensity = loss_dict.get("log_intensity", torch.tensor(0.0))
            integral = loss_dict.get("integral", torch.tensor(0.0))

            if isinstance(log_intensity, torch.Tensor):
                log_intensity = log_intensity.item()
            if isinstance(integral, torch.Tensor):
                integral = integral.item()

            return log_intensity - integral


# =============================================================================
# Evaluator Class
# =============================================================================

class Evaluator:
    """
    Evaluator for comparing SV-NSDE with baselines.

    Usage:
        evaluator = Evaluator(device="cuda")
        results = evaluator.evaluate_all(
            models={"sv_nsde": model, "rmtpp": rmtpp},
            test_data=test_cascades,
        )
    """

    def __init__(
        self,
        device: str = "cpu",
        n_time_samples: int = 100,
    ):
        self.device = device
        self.n_time_samples = n_time_samples
        self.metric_computer = MetricComputer(device)

    def evaluate_model(
        self,
        model: torch.nn.Module,
        cascades: List[CascadeData],
        precomputed_embeddings: Optional[Dict] = None,
        verbose: bool = True,
    ) -> EvaluationMetrics:
        """
        Evaluate a single model on test cascades.

        Args:
            model: The model to evaluate
            cascades: List of test cascades
            precomputed_embeddings: Optional precomputed embeddings
            verbose: Whether to show progress bar

        Returns:
            EvaluationMetrics object
        """
        model.eval()
        model.to(self.device)

        all_time_errors = []
        all_semantic_mse = []
        all_semantic_cosine = []
        all_log_likelihoods = []
        total_events = 0

        iterator = tqdm(cascades, desc="Evaluating") if verbose else cascades

        for cascade in iterator:
            if cascade.size < 3:
                continue

            # Get event marks
            if precomputed_embeddings and cascade.cascade_id in precomputed_embeddings:
                event_marks = precomputed_embeddings[cascade.cascade_id].to(self.device)
            else:
                # Use random embeddings for testing
                event_marks = torch.randn(cascade.size, 32).to(self.device)

            event_times = cascade.event_times.to(self.device)
            T = event_times.max().item() * 1.1  # Add 10% buffer

            # Normalize times
            if T > 0:
                event_times_norm = event_times / T
            else:
                continue

            n_events = len(event_times)
            total_events += n_events

            # 1. Log-likelihood
            try:
                ll = self.metric_computer.compute_log_likelihood(
                    model, event_times_norm, event_marks, 1.0
                )
                all_log_likelihoods.append(ll)
            except Exception as e:
                logger.warning(f"Error computing LL: {e}")

            # 2. Time prediction (predict each event given history)
            with torch.no_grad():
                for i in range(1, min(n_events, 50)):  # Limit for efficiency
                    # Use history up to event i-1
                    history_times = event_times_norm[:i]
                    history_marks = event_marks[:i]

                    # Simple prediction: use intensity to estimate next time
                    try:
                        # Get model state
                        if hasattr(model, 'forward'):
                            outputs = model.forward(
                                history_times, history_marks,
                                T=event_times_norm[i].item()
                            )

                        # Predict using exponential with estimated rate
                        if hasattr(model, 'intensity'):
                            if 'z_events' in outputs and outputs['z_events'] is not None:
                                z_last = outputs['z_events'][-1:]
                                if hasattr(outputs, 'get') and 'v_events' in outputs:
                                    v_last = outputs['v_events'][-1:]
                                    lam = model.intensity(z_last, v_last)
                                else:
                                    lam = model.intensity(z_last)

                                if isinstance(lam, torch.Tensor):
                                    lam = lam.mean().item()

                                # Expected next time: t_last + 1/λ
                                predicted_dt = 1.0 / (lam + 1e-8)
                                predicted_t = history_times[-1].item() + predicted_dt
                                actual_t = event_times_norm[i].item()

                                all_time_errors.append(abs(predicted_t - actual_t))

                    except Exception:
                        pass

            # 3. Semantic prediction (predict next mark)
            with torch.no_grad():
                try:
                    if hasattr(model, 'decoder') or hasattr(model, 'mark_decoder'):
                        outputs = model.forward(event_times_norm[:-1], event_marks[:-1], 1.0)

                        if 'z_events' in outputs and outputs['z_events'] is not None:
                            z_events = outputs['z_events']

                            if hasattr(model, 'decoder'):
                                if hasattr(model.decoder, 'forward'):
                                    predicted = model.decoder(z_events.squeeze(1))
                                else:
                                    predicted = model.decoder(z_events.squeeze(1))
                            elif hasattr(model, 'mark_decoder'):
                                predicted = model.mark_decoder(z_events.squeeze(1))
                            else:
                                predicted = z_events.squeeze(1)

                            # Compare with actual next marks
                            actual = event_marks[1:len(predicted) + 1]
                            if len(predicted) > 0 and len(actual) > 0:
                                min_len = min(len(predicted), len(actual))
                                mse, cosine = self.metric_computer.compute_semantic_metrics(
                                    predicted[:min_len], actual[:min_len]
                                )
                                all_semantic_mse.append(mse)
                                all_semantic_cosine.append(cosine)

                except Exception as e:
                    pass

        # Aggregate metrics
        time_rmse = np.sqrt(np.mean([e**2 for e in all_time_errors])) if all_time_errors else float('nan')
        time_mae = np.mean(all_time_errors) if all_time_errors else float('nan')
        semantic_mse = np.mean(all_semantic_mse) if all_semantic_mse else float('nan')
        semantic_cosine = np.mean(all_semantic_cosine) if all_semantic_cosine else float('nan')
        log_likelihood = np.mean(all_log_likelihoods) if all_log_likelihoods else float('nan')

        # Perplexity = exp(-LL / n_events)
        if not np.isnan(log_likelihood) and total_events > 0:
            perplexity = np.exp(-log_likelihood / (total_events / len(cascades)))
        else:
            perplexity = float('nan')

        return EvaluationMetrics(
            time_rmse=time_rmse,
            time_mae=time_mae,
            semantic_mse=semantic_mse,
            semantic_cosine=semantic_cosine,
            log_likelihood=log_likelihood,
            perplexity=perplexity,
            num_sequences=len(cascades),
            num_events=total_events,
        )

    def evaluate_all(
        self,
        models: Dict[str, torch.nn.Module],
        test_cascades: List[CascadeData],
        precomputed_embeddings: Optional[Dict] = None,
    ) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate all models and return comparison.

        Args:
            models: Dictionary of model_name -> model
            test_cascades: Test data
            precomputed_embeddings: Optional embeddings

        Returns:
            Dictionary of model_name -> EvaluationMetrics
        """
        results = {}

        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            metrics = self.evaluate_model(
                model, test_cascades, precomputed_embeddings
            )
            results[name] = metrics
            logger.info(f"  {name}: RMSE={metrics.time_rmse:.4f}, "
                       f"Cosine={metrics.semantic_cosine:.4f}, "
                       f"LL={metrics.log_likelihood:.4f}")

        return results

    def evaluate_by_phase(
        self,
        model: torch.nn.Module,
        cascades: List[CascadeData],
        phase_labels: List[str],  # "outbreak", "plateau", "decline"
        precomputed_embeddings: Optional[Dict] = None,
    ) -> Dict[str, EvaluationMetrics]:
        """
        Evaluate model performance by crisis phase.

        As per paper Section 4.1:
        - Outbreak: Jan-Feb 2020 (high volatility)
        - Plateau: Mar 2020
        - Decline: Apr+ 2020 (low volatility)
        """
        phase_cascades = defaultdict(list)
        for cascade, phase in zip(cascades, phase_labels):
            phase_cascades[phase].append(cascade)

        results = {}
        for phase, phase_data in phase_cascades.items():
            logger.info(f"Evaluating phase: {phase} ({len(phase_data)} cascades)")
            metrics = self.evaluate_model(
                model, phase_data, precomputed_embeddings, verbose=False
            )
            results[phase] = metrics

        return results


# =============================================================================
# Volatility Analysis
# =============================================================================

class VolatilityAnalyzer:
    """
    Analyzes the volatility decomposition of SV-NSDE.

    Key analysis: Can the model distinguish panic-driven bursts from trends?
    """

    def __init__(self, model: SVNSDE, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    def analyze_cascade(
        self,
        cascade: CascadeData,
        event_marks: torch.Tensor,
    ) -> Dict:
        """
        Analyze volatility decomposition for a single cascade.

        Returns:
            Dictionary with trend/volatility contributions per event
        """
        self.model.eval()

        event_times = cascade.event_times.to(self.device)
        event_marks = event_marks.to(self.device)

        with torch.no_grad():
            decomp = self.model.get_volatility_decomposition(
                event_times, event_marks, T=event_times.max().item()
            )

        return {
            "cascade_id": cascade.cascade_id,
            "event_times": decomp["event_times"].cpu().numpy(),
            "trend_contribution": decomp["trend_contribution"].cpu().numpy(),
            "volatility_contribution": decomp["volatility_contribution"].cpu().numpy(),
            "volatility_ratio": decomp["volatility_ratio"].cpu().numpy(),
            "is_panic_driven": decomp["is_panic_driven"].cpu().numpy(),
            "num_panic_events": decomp["is_panic_driven"].sum().item(),
            "panic_ratio": decomp["is_panic_driven"].float().mean().item(),
        }

    def find_burst_events(
        self,
        cascades: List[CascadeData],
        embeddings_dict: Dict[str, torch.Tensor],
        volatility_threshold: float = 0.6,
    ) -> List[Dict]:
        """
        Find events that are classified as panic-driven bursts.

        Returns events where volatility_ratio > threshold.
        """
        burst_events = []

        for cascade in tqdm(cascades, desc="Analyzing bursts"):
            if cascade.cascade_id not in embeddings_dict:
                continue

            event_marks = embeddings_dict[cascade.cascade_id]
            analysis = self.analyze_cascade(cascade, event_marks)

            vol_ratio = analysis["volatility_ratio"]
            is_burst = vol_ratio > volatility_threshold

            for i, (is_b, vr) in enumerate(zip(is_burst, vol_ratio)):
                if is_b:
                    burst_events.append({
                        "cascade_id": cascade.cascade_id,
                        "event_idx": i,
                        "time": cascade.event_times[i].item(),
                        "volatility_ratio": vr,
                        "text": cascade.event_texts[i] if i < len(cascade.event_texts) else "",
                    })

        return burst_events


# =============================================================================
# Comparison Report Generator
# =============================================================================

def generate_comparison_report(
    results: Dict[str, EvaluationMetrics],
    output_path: Optional[str] = None,
) -> str:
    """
    Generate a formatted comparison report.

    Args:
        results: Dictionary of model_name -> metrics
        output_path: Optional path to save report

    Returns:
        Formatted report string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("MODEL COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Header
    lines.append(f"{'Model':<25} {'Time RMSE':>10} {'Time MAE':>10} "
                f"{'Sem. MSE':>10} {'Cosine':>10} {'LL':>12} {'PPL':>10}")
    lines.append("-" * 80)

    # Sort by log-likelihood (higher is better)
    sorted_models = sorted(
        results.items(),
        key=lambda x: x[1].log_likelihood if not np.isnan(x[1].log_likelihood) else float('-inf'),
        reverse=True
    )

    for name, metrics in sorted_models:
        lines.append(
            f"{name:<25} "
            f"{metrics.time_rmse:>10.4f} "
            f"{metrics.time_mae:>10.4f} "
            f"{metrics.semantic_mse:>10.4f} "
            f"{metrics.semantic_cosine:>10.4f} "
            f"{metrics.log_likelihood:>12.2f} "
            f"{metrics.perplexity:>10.2f}"
        )

    lines.append("-" * 80)
    lines.append("")

    # Best model summary
    best_time = min(sorted_models, key=lambda x: x[1].time_rmse if not np.isnan(x[1].time_rmse) else float('inf'))
    best_semantic = max(sorted_models, key=lambda x: x[1].semantic_cosine if not np.isnan(x[1].semantic_cosine) else float('-inf'))
    best_ll = sorted_models[0]

    lines.append("BEST MODELS:")
    lines.append(f"  Time Prediction (RMSE):    {best_time[0]} ({best_time[1].time_rmse:.4f})")
    lines.append(f"  Semantic Prediction:       {best_semantic[0]} ({best_semantic[1].semantic_cosine:.4f})")
    lines.append(f"  Log-Likelihood:            {best_ll[0]} ({best_ll[1].log_likelihood:.2f})")
    lines.append("")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)
        logger.info(f"Report saved to {output_path}")

    return report


# =============================================================================
# Main Evaluation Script
# =============================================================================

def run_full_evaluation(
    data_path: str,
    output_dir: str = "./results",
    device: str = "cpu",
    d_latent: int = 32,
    d_hidden: int = 64,
):
    """
    Run full evaluation comparing SV-NSDE with all baselines.

    Args:
        data_path: Path to Weibo-COV data or synthetic data
        output_dir: Directory to save results
        device: Device to use
        d_latent: Latent dimension
        d_hidden: Hidden dimension
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info("Loading data...")
    loader = WeiboCOVLoader(data_path)
    loader.load()
    cascades = loader.build_cascades(min_size=5, max_size=200)

    # Split data
    train_cascades, val_cascades, test_cascades = loader.split_by_time()

    logger.info(f"Train: {len(train_cascades)}, Val: {len(val_cascades)}, Test: {len(test_cascades)}")

    # Initialize models
    logger.info("Initializing models...")
    models = {}

    # SV-NSDE (main model)
    models["sv_nsde"] = SVNSDELite(d_input=d_latent, d_latent=d_latent, d_hidden=d_hidden)

    # Baselines
    for name in ["rmtpp", "neural_hawkes", "latent_ode", "neural_jump_sde"]:
        models[name] = get_baseline(name, d_input=d_latent, d_latent=d_latent, d_hidden=d_hidden)

    # Ablations
    models["sv_nsde_no_vol"] = get_baseline("sv_nsde_no_vol", d_input=d_latent, d_latent=d_latent, d_hidden=d_hidden)
    models["sv_nsde_det_vol"] = get_baseline("sv_nsde_det_vol", d_input=d_latent, d_latent=d_latent, d_hidden=d_hidden)

    # Move to device
    for name, model in models.items():
        models[name] = model.to(device)

    # Generate random embeddings for testing (in practice, use real embeddings)
    logger.info("Generating test embeddings...")
    embeddings_dict = {}
    for cascade in test_cascades:
        embeddings_dict[cascade.cascade_id] = torch.randn(cascade.size, d_latent)

    # Evaluate
    logger.info("Running evaluation...")
    evaluator = Evaluator(device=device)
    results = evaluator.evaluate_all(models, test_cascades, embeddings_dict)

    # Generate report
    report = generate_comparison_report(results, output_dir / "comparison_report.txt")
    print(report)

    # Save detailed results
    results_dict = {name: metrics.to_dict() for name, metrics in results.items()}
    with open(output_dir / "results.json", "w") as f:
        json.dump(results_dict, f, indent=2)

    logger.info(f"Results saved to {output_dir}")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate SV-NSDE and baselines")
    parser.add_argument("--data", type=str, default="data/synthetic_weibo_cov.csv",
                       help="Path to data file")
    parser.add_argument("--output", type=str, default="./results",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (cpu/cuda)")
    parser.add_argument("--d_latent", type=int, default=32,
                       help="Latent dimension")

    args = parser.parse_args()

    run_full_evaluation(
        data_path=args.data,
        output_dir=args.output,
        device=args.device,
        d_latent=args.d_latent,
    )
