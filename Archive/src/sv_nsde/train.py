"""
Training Script for SV-NSDE

Implements the training loop with ELBO optimization.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import json
from tqdm import tqdm
import logging

from .model import SVNSDE, SVNSDEConfig, SVNSDELite


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventSequenceDataset(Dataset):
    """
    Dataset for event sequences (cascades).

    Each sample is a sequence of (time, text) pairs representing
    a social media cascade.
    """

    def __init__(
        self,
        event_times: List[torch.Tensor],
        event_texts: Optional[List[List[str]]] = None,
        event_embeddings: Optional[List[torch.Tensor]] = None,
        tokenizer=None,
        max_seq_length: int = 128,
        max_events: int = 100,
    ):
        """
        Args:
            event_times: List of time tensors, one per cascade
            event_texts: List of text lists (if using raw text)
            event_embeddings: List of embedding tensors (if pre-computed)
            tokenizer: HuggingFace tokenizer (required if using raw text)
            max_seq_length: Maximum text sequence length
            max_events: Maximum events per cascade
        """
        self.event_times = event_times
        self.event_texts = event_texts
        self.event_embeddings = event_embeddings
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.max_events = max_events

        assert event_texts is not None or event_embeddings is not None, \
            "Must provide either event_texts or event_embeddings"

        if event_texts is not None and tokenizer is None:
            raise ValueError("tokenizer required when using event_texts")

    def __len__(self) -> int:
        return len(self.event_times)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        times = self.event_times[idx]

        # Truncate if too many events
        if len(times) > self.max_events:
            times = times[:self.max_events]

        # Normalize times to [0, T]
        T = times.max().item()
        if T > 0:
            times = times / T

        if self.event_embeddings is not None:
            embeddings = self.event_embeddings[idx]
            if len(embeddings) > self.max_events:
                embeddings = embeddings[:self.max_events]

            return {
                "times": times,
                "embeddings": embeddings,
                "T": torch.tensor(1.0),  # Normalized
                "n_events": len(times),
            }
        else:
            texts = self.event_texts[idx]
            if len(texts) > self.max_events:
                texts = texts[:self.max_events]

            # Tokenize texts
            encoded = self.tokenizer(
                texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_seq_length,
                return_tensors="pt",
            )

            return {
                "times": times,
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "T": torch.tensor(1.0),
                "n_events": len(times),
            }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for variable-length sequences.

    Pads sequences to the maximum length in the batch.
    """
    max_events = max(item["n_events"] for item in batch)

    # Pad times
    padded_times = []
    masks = []
    for item in batch:
        n = item["n_events"]
        times = item["times"]
        pad_len = max_events - n

        padded_times.append(
            torch.cat([times, torch.zeros(pad_len)])
        )
        masks.append(
            torch.cat([torch.ones(n), torch.zeros(pad_len)])
        )

    result = {
        "times": torch.stack(padded_times),
        "mask": torch.stack(masks),
        "T": torch.stack([item["T"] for item in batch]),
        "n_events": torch.tensor([item["n_events"] for item in batch]),
    }

    # Handle embeddings or input_ids
    if "embeddings" in batch[0]:
        padded_embeddings = []
        d_embed = batch[0]["embeddings"].shape[-1]
        for item in batch:
            n = item["n_events"]
            emb = item["embeddings"]
            pad_len = max_events - n
            padded_embeddings.append(
                torch.cat([emb, torch.zeros(pad_len, d_embed)])
            )
        result["embeddings"] = torch.stack(padded_embeddings)
    else:
        # Pad input_ids and attention_mask
        padded_input_ids = []
        padded_attention_mask = []
        seq_len = batch[0]["input_ids"].shape[-1]

        for item in batch:
            n = item["n_events"]
            pad_len = max_events - n
            padded_input_ids.append(
                torch.cat([
                    item["input_ids"],
                    torch.zeros(pad_len, seq_len, dtype=torch.long)
                ])
            )
            padded_attention_mask.append(
                torch.cat([
                    item["attention_mask"],
                    torch.zeros(pad_len, seq_len, dtype=torch.long)
                ])
            )

        result["input_ids"] = torch.stack(padded_input_ids)
        result["attention_mask"] = torch.stack(padded_attention_mask)

    return result


class Trainer:
    """
    Trainer for SV-NSDE model.
    """

    def __init__(
        self,
        model: SVNSDE,
        train_dataset: EventSequenceDataset,
        val_dataset: Optional[EventSequenceDataset] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        batch_size: int = 16,
        num_epochs: int = 100,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir: str = "./checkpoints",
        log_interval: int = 10,
        n_monte_carlo: int = 1,
    ):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        self.log_interval = log_interval
        self.n_monte_carlo = n_monte_carlo

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # DataLoaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )

        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )
        else:
            self.val_loader = None

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=learning_rate * 0.01,
        )

        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0
        total_log_intensity = 0
        total_integral = 0
        total_recon = 0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            # Move to device
            times = batch["times"].to(self.device)
            mask = batch["mask"].to(self.device)
            T = batch["T"].to(self.device)

            # Get event marks (embeddings or encode from text)
            if "embeddings" in batch:
                event_marks = batch["embeddings"].to(self.device)
            else:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                # Flatten for encoding
                bs, max_events, seq_len = input_ids.shape
                input_ids_flat = input_ids.view(-1, seq_len)
                attention_mask_flat = attention_mask.view(-1, seq_len)
                event_marks = self.model.encode_events(
                    input_ids_flat, attention_mask_flat
                ).view(bs, max_events, -1)

            # Process each sequence in batch
            batch_loss = 0
            for i in range(times.shape[0]):
                n_events = int(mask[i].sum().item())
                if n_events < 2:
                    continue

                seq_times = times[i, :n_events]
                seq_marks = event_marks[i, :n_events]
                seq_T = T[i].item()

                # Compute loss
                loss_dict = self.model.compute_loss(
                    event_times=seq_times,
                    event_marks=seq_marks,
                    T=seq_T,
                    n_samples=self.n_monte_carlo,
                )

                batch_loss += loss_dict["loss"]
                total_log_intensity += loss_dict["log_intensity"].item()
                total_integral += loss_dict["integral"].item()
                total_recon += loss_dict["reconstruction"].item()

            if batch_loss > 0:
                # Backward pass
                self.optimizer.zero_grad()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_loss += batch_loss.item()
                n_batches += 1

                pbar.set_postfix({
                    "loss": f"{batch_loss.item():.4f}",
                })

        avg_loss = total_loss / max(n_batches, 1)
        return {
            "loss": avg_loss,
            "log_intensity": total_log_intensity / max(n_batches, 1),
            "integral": total_integral / max(n_batches, 1),
            "reconstruction": total_recon / max(n_batches, 1),
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Run validation."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0
        n_batches = 0

        for batch in self.val_loader:
            times = batch["times"].to(self.device)
            mask = batch["mask"].to(self.device)
            T = batch["T"].to(self.device)

            if "embeddings" in batch:
                event_marks = batch["embeddings"].to(self.device)
            else:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                bs, max_events, seq_len = input_ids.shape
                input_ids_flat = input_ids.view(-1, seq_len)
                attention_mask_flat = attention_mask.view(-1, seq_len)
                event_marks = self.model.encode_events(
                    input_ids_flat, attention_mask_flat
                ).view(bs, max_events, -1)

            for i in range(times.shape[0]):
                n_events = int(mask[i].sum().item())
                if n_events < 2:
                    continue

                seq_times = times[i, :n_events]
                seq_marks = event_marks[i, :n_events]
                seq_T = T[i].item()

                loss_dict = self.model.compute_loss(
                    event_times=seq_times,
                    event_marks=seq_marks,
                    T=seq_T,
                    n_samples=1,
                )

                total_loss += loss_dict["loss"].item()
                n_batches += 1

        return {"val_loss": total_loss / max(n_batches, 1)}

    def train(self) -> Dict[str, List[float]]:
        """Run full training loop."""
        logger.info(f"Starting training for {self.num_epochs} epochs")
        logger.info(f"Device: {self.device}")

        for epoch in range(1, self.num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            self.train_losses.append(train_metrics["loss"])

            # Validate
            val_metrics = self.validate()
            if "val_loss" in val_metrics:
                self.val_losses.append(val_metrics["val_loss"])

                # Save best model
                if val_metrics["val_loss"] < self.best_val_loss:
                    self.best_val_loss = val_metrics["val_loss"]
                    self.save_checkpoint("best_model.pt")

            # Logging
            if epoch % self.log_interval == 0:
                logger.info(
                    f"Epoch {epoch}: "
                    f"train_loss={train_metrics['loss']:.4f}, "
                    f"log_λ={train_metrics['log_intensity']:.4f}, "
                    f"∫λ={train_metrics['integral']:.4f}, "
                    f"recon={train_metrics['reconstruction']:.4f}"
                )
                if val_metrics:
                    logger.info(f"  val_loss={val_metrics['val_loss']:.4f}")

            # Step scheduler
            self.scheduler.step()

            # Periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

        # Final save
        self.save_checkpoint("final_model.pt")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        path = self.checkpoint_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        path = self.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.best_val_loss = checkpoint["best_val_loss"]
        logger.info(f"Loaded checkpoint from {path}")


def create_synthetic_data(
    n_sequences: int = 100,
    d_latent: int = 32,
    max_events: int = 50,
    seed: int = 42,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Create synthetic event data for testing.

    Returns:
        event_times: List of time tensors
        event_embeddings: List of embedding tensors
    """
    torch.manual_seed(seed)

    event_times = []
    event_embeddings = []

    for _ in range(n_sequences):
        # Random number of events
        n_events = torch.randint(10, max_events, (1,)).item()

        # Generate times (sorted)
        times = torch.rand(n_events).sort()[0]

        # Generate embeddings (could be more sophisticated)
        embeddings = torch.randn(n_events, d_latent)

        event_times.append(times)
        event_embeddings.append(embeddings)

    return event_times, event_embeddings


if __name__ == "__main__":
    # Example usage with synthetic data
    print("Creating synthetic data...")
    event_times, event_embeddings = create_synthetic_data(
        n_sequences=100,
        d_latent=32,
    )

    # Create dataset
    dataset = EventSequenceDataset(
        event_times=event_times,
        event_embeddings=event_embeddings,
    )

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_times = event_times[:train_size]
    train_embeddings = event_embeddings[:train_size]
    val_times = event_times[train_size:]
    val_embeddings = event_embeddings[train_size:]

    train_dataset = EventSequenceDataset(
        event_times=train_times,
        event_embeddings=train_embeddings,
    )
    val_dataset = EventSequenceDataset(
        event_times=val_times,
        event_embeddings=val_embeddings,
    )

    # Create model (lightweight version for testing)
    print("Creating model...")
    model = SVNSDELite(d_input=32, d_latent=32, d_hidden=64)

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=1e-3,
        batch_size=8,
        num_epochs=10,
        device="cpu",
    )

    # Train
    print("Starting training...")
    history = trainer.train()
    print("Training complete!")
