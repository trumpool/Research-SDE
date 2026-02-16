"""
Full SV-NSDE Model

Combines all components into the complete Semantic Volatility-Modulated
Neural SDE model for crisis dynamics modeling.

Reference: Full paper methodology.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass

from .encoder import SemanticEncoder, LightweightEncoder
from .sde import NeuralHestonSDE, SDEConfig
from .intensity import DualChannelIntensity
from .decoder import SemanticDecoder


@dataclass
class SVNSDEConfig:
    """Configuration for the full SV-NSDE model."""
    # Dimensions
    d_latent: int = 32
    d_hidden: int = 64

    # Encoder settings
    pretrained_model: str = "hfl/chinese-roberta-wwm-ext"
    freeze_encoder: bool = False
    use_lightweight_encoder: bool = False

    # SDE settings
    kappa_init: float = 1.0
    theta_init: float = 0.1
    xi_init: float = 0.3

    # Intensity settings
    use_gating: bool = True

    # Decoder settings
    sigma_obs: float = 0.1
    learn_variance: bool = False

    # Training settings
    dt: float = 0.01
    kl_weight: float = 0.01


class SVNSDE(nn.Module):
    """
    Semantic Volatility-Modulated Neural SDE.

    A neural point process model that combines:
    1. RoBERTa semantic encoding
    2. Coupled Heston-style SDE for trend and volatility
    3. Dual-channel intensity function
    4. VAE-style semantic reconstruction

    The model learns to distinguish between:
    - Trend-driven events (normal hotspots)
    - Volatility-driven events (panic bursts)
    """

    def __init__(self, config: Optional[SVNSDEConfig] = None):
        super().__init__()
        if config is None:
            config = SVNSDEConfig()
        self.config = config

        # 1. Semantic Encoder
        if config.use_lightweight_encoder:
            self.encoder = LightweightEncoder(
                d_latent=config.d_latent,
            )
        else:
            self.encoder = SemanticEncoder(
                d_latent=config.d_latent,
                pretrained_model=config.pretrained_model,
                freeze_bert=config.freeze_encoder,
            )

        # 2. Neural Heston SDE
        sde_config = SDEConfig(
            d_latent=config.d_latent,
            d_hidden=config.d_hidden,
            kappa_init=config.kappa_init,
            theta_init=config.theta_init,
            xi_init=config.xi_init,
        )
        self.sde = NeuralHestonSDE(config=sde_config)

        # 3. Dual-Channel Intensity
        self.intensity = DualChannelIntensity(
            d_latent=config.d_latent,
            d_hidden=config.d_hidden,
            use_gating=config.use_gating,
        )

        # 4. Semantic Decoder
        self.decoder = SemanticDecoder(
            d_latent=config.d_latent,
            d_hidden=config.d_hidden * 2,
            sigma_obs=config.sigma_obs,
            learn_variance=config.learn_variance,
        )

    def encode_events(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode event texts to semantic vectors.

        Args:
            input_ids: Token IDs [n_events, seq_len]
            attention_mask: Attention mask [n_events, seq_len]

        Returns:
            x_marks: Semantic vectors [n_events, d_latent]
        """
        return self.encoder(input_ids, attention_mask)

    def forward(
        self,
        event_times: torch.Tensor,
        event_marks: torch.Tensor,
        T: float,
        n_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass: solve SDE and compute outputs.

        Args:
            event_times: Event timestamps [n_events]
            event_marks: Event semantic vectors [n_events, d_latent]
            T: Terminal time
            n_samples: Number of Monte Carlo samples for SDE

        Returns:
            Dictionary containing model outputs
        """
        device = event_marks.device

        # Solve the coupled SDE
        sde_output = self.sde.solve(
            event_times=event_times,
            event_marks=event_marks,
            T=T,
            dt=self.config.dt,
            batch_size=n_samples,
            return_full_trajectory=True,
        )

        # Extract states at event times
        z_events = sde_output["z_events"]  # [n_events, n_samples, d_latent]
        v_events = sde_output["v_events"]  # [n_events, n_samples, d_latent]

        # Compute intensities at event times
        if z_events is not None:
            n_events = z_events.shape[0]
            intensities = []
            for i in range(n_events):
                lam = self.intensity(z_events[i], v_events[i])
                intensities.append(lam)
            intensities = torch.stack(intensities)  # [n_events, n_samples]
        else:
            intensities = None

        return {
            "z_events": z_events,
            "v_events": v_events,
            "z_trajectory": sde_output["z_trajectory"],
            "v_trajectory": sde_output["v_trajectory"],
            "times": sde_output["times"],
            "intensities": intensities,
        }

    def compute_loss(
        self,
        event_times: torch.Tensor,
        event_marks: torch.Tensor,
        T: float,
        n_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ELBO loss.

        Equation (6):
            L = Σ log λ(t_i) - ∫λ(t)dt + Σ log p(x_i|z(t_i)) - KL(Q||P)

        Args:
            event_times: Event timestamps [n_events]
            event_marks: Event semantic vectors [n_events, d_latent]
            T: Terminal time
            n_samples: Number of Monte Carlo samples

        Returns:
            Dictionary containing loss components
        """
        device = event_marks.device
        n_events = len(event_times)

        # Forward pass
        outputs = self.forward(event_times, event_marks, T, n_samples)

        z_events = outputs["z_events"]
        v_events = outputs["v_events"]
        z_traj = outputs["z_trajectory"]
        v_traj = outputs["v_trajectory"]
        times = outputs["times"]

        # 1. Event log-likelihood: Σ log λ(t_i)
        if z_events is not None and n_events > 0:
            log_intensity_sum = torch.zeros(n_samples, device=device)
            for i in range(n_events):
                lam = self.intensity(z_events[i], v_events[i])
                log_intensity_sum += torch.log(lam + 1e-8)
            log_intensity_sum = log_intensity_sum.mean()
        else:
            log_intensity_sum = torch.tensor(0.0, device=device)

        # 2. Integral term: -∫λ(t)dt
        integral = self.intensity.compute_integral(z_traj, v_traj, times)
        integral = integral.mean()

        # 3. Semantic reconstruction: Σ log p(x_i|z(t_i))
        if z_events is not None and n_events > 0:
            recon_loss = torch.zeros(n_samples, device=device)
            for i in range(n_events):
                x_i = event_marks[i].unsqueeze(0).expand(n_samples, -1)
                log_p = self.decoder.log_prob(x_i, z_events[i])
                recon_loss += log_p
            recon_loss = recon_loss.mean()
        else:
            recon_loss = torch.tensor(0.0, device=device)

        # 4. KL divergence (simplified version)
        # Full version would use Girsanov theorem for SDE paths
        kl_loss = self.config.kl_weight * (z_traj ** 2).mean()

        # ELBO = log_likelihood - integral + reconstruction - KL
        elbo = log_intensity_sum - integral + recon_loss - kl_loss
        loss = -elbo  # Minimize negative ELBO

        return {
            "loss": loss,
            "log_intensity": log_intensity_sum,
            "integral": integral,
            "reconstruction": recon_loss,
            "kl": kl_loss,
            "elbo": elbo,
        }

    def predict_next_event(
        self,
        event_times: torch.Tensor,
        event_marks: torch.Tensor,
        current_time: float,
        horizon: float = 1.0,
        n_samples: int = 100,
        dt: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict the next event time and intensity.

        Uses thinning algorithm for sampling from the point process.

        Args:
            event_times: Historical event times
            event_marks: Historical event marks
            current_time: Current time
            horizon: Prediction horizon
            n_samples: Number of Monte Carlo samples
            dt: Time discretization

        Returns:
            Dictionary with predictions
        """
        device = event_marks.device

        # Get current state by solving SDE up to current_time
        sde_output = self.sde.solve(
            event_times=event_times[event_times < current_time],
            event_marks=event_marks[event_times < current_time],
            T=current_time,
            dt=dt,
            batch_size=n_samples,
        )

        z = sde_output["z_final"]
        v = sde_output["v_final"]

        # Compute current intensity
        current_intensity = self.intensity(z, v)

        # Simple forward simulation for expected intensity
        predicted_times = []
        predicted_intensities = []

        t = current_time
        while t < current_time + horizon:
            # Evolve state
            z, v = self.sde.euler_maruyama_step(z, v, torch.tensor(t), dt)
            lam = self.intensity(z, v)

            predicted_times.append(t)
            predicted_intensities.append(lam.mean().item())
            t += dt

        return {
            "current_intensity": current_intensity.mean(),
            "predicted_times": predicted_times,
            "predicted_intensities": predicted_intensities,
            "z_current": z,
            "v_current": v,
        }

    def get_volatility_decomposition(
        self,
        event_times: torch.Tensor,
        event_marks: torch.Tensor,
        T: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Decompose intensity into trend and volatility components.

        This is the key analysis for distinguishing panic from trends.

        Args:
            event_times: Event timestamps
            event_marks: Event semantic vectors
            T: Terminal time

        Returns:
            Dictionary with decomposition results
        """
        outputs = self.forward(event_times, event_marks, T, n_samples=1)

        z_events = outputs["z_events"].squeeze(1)  # [n_events, d_latent]
        v_events = outputs["v_events"].squeeze(1)

        # Get intensity components
        trend_contribs = []
        vol_contribs = []
        total_intensities = []

        for i in range(len(event_times)):
            intensity, components = self.intensity(
                z_events[i:i+1],
                v_events[i:i+1],
                return_components=True,
            )
            trend_contribs.append(components["trend_contrib"].item())
            vol_contribs.append(components["vol_contrib"].item())
            total_intensities.append(intensity.item())

        # Classify events
        trend_contribs = torch.tensor(trend_contribs)
        vol_contribs = torch.tensor(vol_contribs)

        # Event is "panic-driven" if volatility contribution dominates
        vol_ratio = vol_contribs / (trend_contribs + vol_contribs + 1e-8)
        is_panic_driven = vol_ratio > 0.5

        return {
            "event_times": event_times,
            "trend_contribution": trend_contribs,
            "volatility_contribution": vol_contribs,
            "total_intensity": torch.tensor(total_intensities),
            "volatility_ratio": vol_ratio,
            "is_panic_driven": is_panic_driven,
            "z_states": z_events,
            "v_states": v_events,
        }


class SVNSDELite(nn.Module):
    """
    Lightweight version without pretrained encoder.

    Useful for quick experiments or when working with
    pre-computed embeddings.
    """

    def __init__(
        self,
        d_input: int = 768,
        d_latent: int = 32,
        d_hidden: int = 64,
    ):
        super().__init__()

        # Simple projection instead of full encoder
        self.projection = nn.Sequential(
            nn.Linear(d_input, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_latent),
        )

        # SDE components
        self.sde = NeuralHestonSDE(d_latent=d_latent, d_hidden=d_hidden)
        self.intensity = DualChannelIntensity(d_latent=d_latent, d_hidden=d_hidden)
        self.decoder = SemanticDecoder(d_latent=d_latent, d_hidden=d_hidden)

        self.d_latent = d_latent
        self.dt = 0.01
        self.kl_weight = 0.01

    def forward(
        self,
        event_times: torch.Tensor,
        event_embeddings: torch.Tensor,
        T: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with pre-computed embeddings.

        Args:
            event_times: Event timestamps [n_events]
            event_embeddings: Pre-computed embeddings [n_events, d_input]
            T: Terminal time

        Returns:
            Model outputs
        """
        # Project embeddings
        event_marks = self.projection(event_embeddings)

        # Solve SDE
        sde_output = self.sde.solve(
            event_times=event_times,
            event_marks=event_marks,
            T=T,
            dt=self.dt,
            batch_size=1,
            return_full_trajectory=True,
        )

        return {
            "event_marks": event_marks,
            **sde_output,
        }

    def compute_loss(
        self,
        event_times: torch.Tensor,
        event_marks: torch.Tensor,
        T: float,
        n_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute ELBO loss for pre-projected event marks.

        Args:
            event_times: Event timestamps [n_events]
            event_marks: Event semantic vectors [n_events, d_latent]
            T: Terminal time
            n_samples: Number of Monte Carlo samples

        Returns:
            Dictionary containing loss components
        """
        device = event_marks.device
        n_events = len(event_times)

        # Project embeddings from d_input to d_latent
        projected_marks = self.projection(event_marks)

        # Solve SDE
        sde_output = self.sde.solve(
            event_times=event_times,
            event_marks=projected_marks,
            T=T,
            dt=self.dt,
            batch_size=n_samples,
            return_full_trajectory=True,
        )

        z_events = sde_output["z_events"]
        v_events = sde_output["v_events"]
        z_traj = sde_output["z_trajectory"]
        v_traj = sde_output["v_trajectory"]
        times = sde_output["times"]

        # 1. Event log-likelihood: Σ log λ(t_i)
        if z_events is not None and n_events > 0:
            log_intensity_sum = torch.zeros(n_samples, device=device)
            for i in range(n_events):
                lam = self.intensity(z_events[i], v_events[i])
                log_intensity_sum += torch.log(lam + 1e-8)
            log_intensity_sum = log_intensity_sum.mean()
        else:
            log_intensity_sum = torch.tensor(0.0, device=device)

        # 2. Integral term: -∫λ(t)dt
        integral = self.intensity.compute_integral(z_traj, v_traj, times)
        integral = integral.mean()

        # 3. Semantic reconstruction: Σ log p(x_i|z(t_i))
        if z_events is not None and n_events > 0:
            recon_loss = torch.zeros(n_samples, device=device)
            for i in range(n_events):
                x_i = projected_marks[i].unsqueeze(0).expand(n_samples, -1)
                log_p = self.decoder.log_prob(x_i, z_events[i])
                recon_loss += log_p
            recon_loss = recon_loss.mean()
        else:
            recon_loss = torch.tensor(0.0, device=device)

        # 4. KL divergence (simplified)
        kl_loss = self.kl_weight * (z_traj ** 2).mean()

        # ELBO
        elbo = log_intensity_sum - integral + recon_loss - kl_loss
        loss = -elbo

        return {
            "loss": loss,
            "log_intensity": log_intensity_sum,
            "integral": integral,
            "reconstruction": recon_loss,
            "kl": kl_loss,
            "elbo": elbo,
        }
