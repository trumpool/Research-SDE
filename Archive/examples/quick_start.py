"""
Quick Start Example for SV-NSDE

Demonstrates basic usage of the model with synthetic data.
"""

import torch
import sys
sys.path.insert(0, "../src")

from sv_nsde import SVNSDE, SVNSDELite
from sv_nsde.model import SVNSDEConfig
from sv_nsde.train import EventSequenceDataset, Trainer, create_synthetic_data


def example_synthetic_training():
    """Train on synthetic data (no GPU or pretrained models needed)."""
    print("=" * 60)
    print("Example 1: Training with Synthetic Data")
    print("=" * 60)

    # Create synthetic event sequences
    event_times, event_embeddings = create_synthetic_data(
        n_sequences=50,
        d_latent=32,
        max_events=30,
    )

    # Create datasets
    train_dataset = EventSequenceDataset(
        event_times=event_times[:40],
        event_embeddings=event_embeddings[:40],
    )
    val_dataset = EventSequenceDataset(
        event_times=event_times[40:],
        event_embeddings=event_embeddings[40:],
    )

    # Create lightweight model (no BERT)
    model = SVNSDELite(
        d_input=32,
        d_latent=32,
        d_hidden=64,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        learning_rate=1e-3,
        batch_size=8,
        num_epochs=5,
        device="cpu",
        checkpoint_dir="./checkpoints_synthetic",
    )

    history = trainer.train()
    print(f"Final train loss: {history['train_losses'][-1]:.4f}")


def example_forward_pass():
    """Demonstrate a single forward pass."""
    print("\n" + "=" * 60)
    print("Example 2: Single Forward Pass")
    print("=" * 60)

    # Create model
    model = SVNSDELite(d_input=32, d_latent=32)

    # Create sample data
    n_events = 20
    event_times = torch.rand(n_events).sort()[0]
    event_embeddings = torch.randn(n_events, 32)

    # Forward pass
    with torch.no_grad():
        outputs = model(event_times, event_embeddings, T=1.0)

    print(f"Number of events: {n_events}")
    print(f"Event marks shape: {outputs['event_marks'].shape}")
    print(f"z_events shape: {outputs['z_events'].shape if outputs['z_events'] is not None else 'None'}")
    print(f"z_trajectory shape: {outputs['z_trajectory'].shape}")


def example_volatility_decomposition():
    """Demonstrate volatility decomposition analysis."""
    print("\n" + "=" * 60)
    print("Example 3: Volatility Decomposition")
    print("=" * 60)

    from sv_nsde.model import SVNSDE, SVNSDEConfig

    # Create full model with lightweight encoder
    config = SVNSDEConfig(
        d_latent=32,
        use_lightweight_encoder=True,
    )
    model = SVNSDE(config)

    # Create sample data
    n_events = 15
    event_times = torch.rand(n_events).sort()[0]

    # Simulate tokenized input (random for demo)
    input_ids = torch.randint(0, 1000, (n_events, 64))
    attention_mask = torch.ones_like(input_ids)

    # Encode events
    with torch.no_grad():
        event_marks = model.encode_events(input_ids, attention_mask)

        # Get decomposition
        decomp = model.get_volatility_decomposition(
            event_times, event_marks, T=1.0
        )

    print(f"Event times: {decomp['event_times'][:5]}...")
    print(f"Trend contribution: {decomp['trend_contribution'][:5]}...")
    print(f"Volatility contribution: {decomp['volatility_contribution'][:5]}...")
    print(f"Volatility ratio: {decomp['volatility_ratio'][:5]}...")
    print(f"Panic-driven events: {decomp['is_panic_driven'].sum().item()}/{n_events}")


def example_sde_path_sampling():
    """Demonstrate SDE path sampling for visualization."""
    print("\n" + "=" * 60)
    print("Example 4: SDE Path Sampling")
    print("=" * 60)

    from sv_nsde.sde import NeuralHestonSDE

    # Create SDE module
    sde = NeuralHestonSDE(d_latent=2)  # 2D for easy visualization

    # Sample paths (no events, pure diffusion)
    z_paths, v_paths, times = sde.sample_paths(
        n_paths=5,
        T=1.0,
        dt=0.01,
    )

    print(f"Sampled {z_paths.shape[1]} paths over {len(times)} time steps")
    print(f"z trajectory shape: {z_paths.shape}")  # [n_steps, n_paths, d_latent]
    print(f"v trajectory shape: {v_paths.shape}")

    # Could visualize with matplotlib:
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 4))
    # for i in range(z_paths.shape[1]):
    #     plt.plot(times, z_paths[:, i, 0].numpy(), alpha=0.7)
    # plt.xlabel("Time")
    # plt.ylabel("z(t) dimension 0")
    # plt.title("Sample SDE Trajectories")
    # plt.show()


def example_intensity_components():
    """Demonstrate dual-channel intensity computation."""
    print("\n" + "=" * 60)
    print("Example 5: Dual-Channel Intensity")
    print("=" * 60)

    from sv_nsde.intensity import DualChannelIntensity

    # Create intensity module
    intensity = DualChannelIntensity(d_latent=32, use_gating=True)

    # Sample states
    z = torch.randn(10, 32)  # 10 time points
    v = torch.abs(torch.randn(10, 32)) * 0.1  # variance (positive)

    # Compute intensity with components
    lam, components = intensity(z, v, return_components=True)

    print(f"Intensity λ(t): {lam[:5]}...")
    print(f"Trend contribution: {components['trend_contrib'][:5]}...")
    print(f"Volatility contribution: {components['vol_contrib'][:5]}...")
    print(f"Gate weights (trend): {components['w_trend'][:5]}...")
    print(f"Gate weights (vol): {components['w_vol'][:5]}...")


if __name__ == "__main__":
    example_forward_pass()
    example_sde_path_sampling()
    example_intensity_components()
    example_volatility_decomposition()

    # This one takes a bit longer
    print("\nRunning training example (this may take a moment)...")
    example_synthetic_training()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
