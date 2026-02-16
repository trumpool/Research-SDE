#!/usr/bin/env python
"""
Run Full Evaluation of SV-NSDE vs Baselines

Usage:
    python scripts/run_evaluation.py --data data/synthetic_weibo_cov.csv --output results/
    python scripts/run_evaluation.py --data data/weibo_cov_v2.csv --device cuda
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import numpy as np
import json
import argparse
from datetime import datetime
from collections import defaultdict

from sv_nsde import (
    SVNSDELite,
    get_baseline,
    WeiboCOVLoader,
    Evaluator,
    generate_synthetic_weibo_data,
)
from sv_nsde.evaluate import generate_comparison_report


def main(args):
    print("=" * 70)
    print("SV-NSDE EVALUATION SCRIPT")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Device: {args.device}")
    print(f"Output: {args.output}")
    print()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data if needed
    data_path = Path(args.data)
    if not data_path.exists():
        print("Data file not found. Generating synthetic data...")
        generate_synthetic_weibo_data(
            n_cascades=500,
            output_path=str(data_path),
        )

    # Load data
    print("\n[1/5] Loading data...")
    loader = WeiboCOVLoader(str(data_path))
    loader.load(nrows=args.max_rows)
    cascades = loader.build_cascades(min_size=5, max_size=200)
    stats = loader.get_statistics()

    print(f"  Total cascades: {stats['num_cascades']}")
    print(f"  Total events: {stats['total_events']}")
    print(f"  Avg cascade size: {stats['size_mean']:.1f} ± {stats['size_std']:.1f}")
    print(f"  Max cascade size: {stats['size_max']}")

    # Split data
    train, val, test = loader.split_by_time()
    print(f"\n  Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    # Use subset for quick testing
    if args.quick:
        test = test[:20]
        print(f"  (Quick mode: using {len(test)} test cascades)")

    # Initialize models
    print("\n[2/5] Initializing models...")
    d_latent = args.d_latent
    d_hidden = args.d_hidden

    models = {}

    # Main model: SV-NSDE
    models["SV-NSDE"] = SVNSDELite(
        d_input=d_latent,
        d_latent=d_latent,
        d_hidden=d_hidden,
    )

    # Baselines
    baseline_names = [
        ("RMTPP", "rmtpp"),
        ("Neural Hawkes", "neural_hawkes"),
        ("Latent ODE", "latent_ode"),
        ("Neural Jump SDE", "neural_jump_sde"),
    ]

    for display_name, model_name in baseline_names:
        try:
            models[display_name] = get_baseline(
                model_name,
                d_input=d_latent,
                d_latent=d_latent,
                d_hidden=d_hidden,
            )
        except Exception as e:
            print(f"  Warning: Could not load {display_name}: {e}")

    # Ablation variants
    ablation_names = [
        ("SV-NSDE (no vol)", "sv_nsde_no_vol"),
        ("SV-NSDE (det vol)", "sv_nsde_det_vol"),
    ]

    for display_name, model_name in ablation_names:
        try:
            models[display_name] = get_baseline(
                model_name,
                d_input=d_latent,
                d_latent=d_latent,
                d_hidden=d_hidden,
            )
        except Exception as e:
            print(f"  Warning: Could not load {display_name}: {e}")

    # Print model info
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {n_params:,} parameters")

    # Move to device
    device = args.device
    for name in models:
        models[name] = models[name].to(device)

    # Generate embeddings for test data
    print("\n[3/5] Preparing test embeddings...")
    embeddings = {}
    for cascade in test:
        embeddings[cascade.cascade_id] = torch.randn(cascade.size, d_latent)

    # Evaluate
    print("\n[4/5] Running evaluation...")
    evaluator = Evaluator(device=device)

    results = {}
    for name, model in models.items():
        print(f"\n  Evaluating {name}...")
        try:
            metrics = evaluator.evaluate_model(
                model, test, embeddings, verbose=True
            )
            results[name] = metrics

            print(f"    Time RMSE: {metrics.time_rmse:.4f}")
            print(f"    Semantic Cosine: {metrics.semantic_cosine:.4f}")
            print(f"    Log-Likelihood: {metrics.log_likelihood:.2f}")
        except Exception as e:
            print(f"    Error: {e}")

    # Generate report
    print("\n[5/5] Generating report...")
    report = generate_comparison_report(results, output_dir / "report.txt")
    print("\n" + report)

    # Save detailed results
    results_json = {
        name: metrics.to_dict()
        for name, metrics in results.items()
    }
    results_json["_metadata"] = {
        "timestamp": datetime.now().isoformat(),
        "data_file": str(args.data),
        "num_test_cascades": len(test),
        "d_latent": d_latent,
        "d_hidden": d_hidden,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    print(f"\nResults saved to {output_dir}")

    # Summary table for paper
    print("\n" + "=" * 70)
    print("TABLE FOR PAPER (LaTeX format):")
    print("=" * 70)
    print("\\begin{tabular}{lccccc}")
    print("\\toprule")
    print("Model & Time RMSE $\\downarrow$ & Cosine $\\uparrow$ & LL $\\uparrow$ \\\\")
    print("\\midrule")

    for name, metrics in sorted(results.items(), key=lambda x: -x[1].log_likelihood if not np.isnan(x[1].log_likelihood) else float('-inf')):
        rmse = f"{metrics.time_rmse:.3f}" if not np.isnan(metrics.time_rmse) else "-"
        cosine = f"{metrics.semantic_cosine:.3f}" if not np.isnan(metrics.semantic_cosine) else "-"
        ll = f"{metrics.log_likelihood:.1f}" if not np.isnan(metrics.log_likelihood) else "-"
        print(f"{name} & {rmse} & {cosine} & {ll} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate SV-NSDE and baselines")
    parser.add_argument("--data", type=str, default="data/synthetic_weibo_cov.csv",
                       help="Path to data CSV")
    parser.add_argument("--output", type=str, default="results",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (cpu/cuda)")
    parser.add_argument("--d_latent", type=int, default=32,
                       help="Latent dimension")
    parser.add_argument("--d_hidden", type=int, default=64,
                       help="Hidden dimension")
    parser.add_argument("--max_rows", type=int, default=None,
                       help="Max rows to load")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode (fewer test samples)")

    args = parser.parse_args()
    main(args)
