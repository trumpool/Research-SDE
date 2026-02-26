"""
Experiment Workflows for SV-NSDE

High-level functions for running experiments:
- BERT embedding precomputation
- Model training (SV-NSDE + baselines)
- Unified evaluation and comparison
"""

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from tqdm.auto import tqdm

from .data import CascadeData
from .model import SVNSDELite
from .baselines import get_baseline, BASELINE_MODELS
from .evaluate import Evaluator


@dataclass
class TrainConfig:
    """Training configuration."""
    lr: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 20
    grad_clip: float = 1.0
    min_cascade_size: int = 3
    print_every: int = 5


# ---------------------------------------------------------------------------
# 1. BERT Embedding Precomputation
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_bert_embeddings(
    cascades: List[CascadeData],
    tokenizer,
    bert_model: nn.Module,
    device: str = "cuda",
    batch_size: int = 64,
    max_len: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Precompute [CLS] embeddings for all cascade texts using a BERT model.

    Args:
        cascades: List of CascadeData objects
        tokenizer: HuggingFace tokenizer
        bert_model: HuggingFace BERT model (already on device)
        device: Device string
        batch_size: Encoding batch size
        max_len: Max token length

    Returns:
        Dict mapping cascade_id → Tensor [n_events, 768]
    """
    bert_model.eval()
    embeddings = {}
    all_texts = []
    cascade_map = []  # (cascade_id, start_idx, end_idx)

    for c in cascades:
        start = len(all_texts)
        texts = [
            t if isinstance(t, str) and len(t.strip()) > 0 else "[UNK]"
            for t in c.event_texts
        ]
        if len(texts) == 0:
            texts = ["[UNK]"] * c.size
        all_texts.extend(texts)
        cascade_map.append((c.cascade_id, start, start + len(texts)))

    print(f"  Total texts to encode: {len(all_texts):,}")

    if len(all_texts) == 0:
        print("  ⚠️ No texts found, using random embeddings as fallback")
        for c in cascades:
            embeddings[c.cascade_id] = torch.randn(c.size, 768)
        return embeddings

    all_embs = []
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Encoding", unit="batch"):
        batch_texts = all_texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(device)
        outputs = bert_model(**encoded)
        cls_emb = outputs.last_hidden_state[:, 0, :]
        all_embs.append(cls_emb.cpu())

    all_embs = torch.cat(all_embs, dim=0)

    for cascade_id, start, end in cascade_map:
        embeddings[cascade_id] = all_embs[start:end]

    return embeddings


# ---------------------------------------------------------------------------
# 2. Model Training
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_cascades: List[CascadeData],
    val_cascades: List[CascadeData],
    train_emb: Dict[str, torch.Tensor],
    val_emb: Dict[str, torch.Tensor],
    config: Optional[TrainConfig] = None,
    device: str = "cuda",
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Train a single model on cascade data.

    Args:
        model: Model with compute_loss(event_times, event_marks, T) API
        train_cascades: Training cascades
        val_cascades: Validation cascades
        train_emb: Precomputed train embeddings {cascade_id: Tensor}
        val_emb: Precomputed val embeddings {cascade_id: Tensor}
        config: Training configuration
        device: Device string

    Returns:
        (trained_model, train_losses, val_losses)
    """
    if config is None:
        config = TrainConfig()

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=config.lr * 0.01)

    train_losses = []
    val_losses = []

    for epoch in range(1, config.epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for cascade in train_cascades:
            if cascade.size < config.min_cascade_size:
                continue
            if cascade.cascade_id not in train_emb:
                continue

            event_times = cascade.event_times.to(device)
            event_marks = train_emb[cascade.cascade_id].to(device)

            T = event_times.max().item()
            if T > 0:
                event_times = event_times / T

            try:
                loss_dict = model.compute_loss(event_times, event_marks, T=1.0)
                loss = loss_dict["loss"]
                if torch.isnan(loss) or torch.isinf(loss):
                    continue
            except Exception as e:
                if epoch == 1 and n_batches == 0:
                    print(f"    ⚠️ compute_loss error: {e}")
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        n_val = 0

        with torch.no_grad():
            for cascade in val_cascades:
                if cascade.size < config.min_cascade_size:
                    continue
                if cascade.cascade_id not in val_emb:
                    continue

                event_times = cascade.event_times.to(device)
                event_marks = val_emb[cascade.cascade_id].to(device)

                T = event_times.max().item()
                if T > 0:
                    event_times = event_times / T

                try:
                    loss_dict = model.compute_loss(event_times, event_marks, T=1.0)
                    loss_val = loss_dict["loss"].item()
                    if not (np.isnan(loss_val) or np.isinf(loss_val)):
                        val_loss += loss_val
                        n_val += 1
                except Exception:
                    continue

        avg_val = val_loss / max(n_val, 1)
        val_losses.append(avg_val)

        scheduler.step()

        if epoch % config.print_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{config.epochs}: train={avg_train:.2f}, val={avg_val:.2f}")

    return model, train_losses, val_losses


# ---------------------------------------------------------------------------
# 3. Train All Baselines
# ---------------------------------------------------------------------------

def train_all_baselines(
    baseline_names: List[str],
    train_cascades: List[CascadeData],
    val_cascades: List[CascadeData],
    train_emb: Dict[str, torch.Tensor],
    val_emb: Dict[str, torch.Tensor],
    d_input: int = 768,
    d_latent: int = 32,
    d_hidden: int = 64,
    config: Optional[TrainConfig] = None,
    device: str = "cuda",
) -> Dict[str, Tuple[nn.Module, List[float], List[float]]]:
    """
    Create and train multiple baseline models.

    Args:
        baseline_names: List of baseline names (keys in BASELINE_MODELS)
        train_cascades, val_cascades: Data splits
        train_emb, val_emb: Precomputed embeddings
        d_input, d_latent, d_hidden: Model dimensions
        config: Training configuration
        device: Device string

    Returns:
        Dict[name -> (model, train_losses, val_losses)]
    """
    results = {}

    for name in baseline_names:
        print(f"\n{'='*50}")
        print(f"Training: {name}")
        print(f"{'='*50}")

        model = get_baseline(name, d_input=d_input, d_latent=d_latent, d_hidden=d_hidden)
        model, train_losses, val_losses = train_model(
            model, train_cascades, val_cascades, train_emb, val_emb, config, device,
        )
        results[name] = (model, train_losses, val_losses)

    return results


# ---------------------------------------------------------------------------
# 4. Unified Evaluation & Comparison
# ---------------------------------------------------------------------------

def run_comparison(
    models: Dict[str, nn.Module],
    test_cascades: List[CascadeData],
    test_emb: Dict[str, torch.Tensor],
    device: str = "cuda",
    max_cascades: int = 50,
) -> pd.DataFrame:
    """
    Evaluate all models on test data and return a comparison DataFrame.

    Args:
        models: Dict[model_name -> model]
        test_cascades: Test cascades
        test_emb: Precomputed test embeddings
        device: Device string
        max_cascades: Max cascades to evaluate (for speed)

    Returns:
        pd.DataFrame with columns [Model, Log-Likelihood, Semantic Cosine, Time RMSE]
    """
    evaluator = Evaluator(device=device)
    rows = []

    for name, model in models.items():
        print(f"Evaluating {name}...")
        try:
            metrics = evaluator.evaluate_model(
                model,
                test_cascades[:max_cascades],
                test_emb,
                verbose=False,
            )
            rows.append({
                "Model": name,
                "Log-Likelihood": metrics.log_likelihood,
                "Semantic Cosine": metrics.semantic_cosine,
                "Time RMSE": metrics.time_rmse,
            })
            print(f"  LL={metrics.log_likelihood:.2f}, Cosine={metrics.semantic_cosine:.4f}, RMSE={metrics.time_rmse:.4f}")
        except Exception as e:
            print(f"  Error: {e}")
            rows.append({
                "Model": name,
                "Log-Likelihood": float("nan"),
                "Semantic Cosine": float("nan"),
                "Time RMSE": float("nan"),
            })

    return pd.DataFrame(rows)
