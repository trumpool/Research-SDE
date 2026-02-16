"""
Dual-Channel Intensity Function Module

Implements the volatility-coupled intensity for the point process.
Reference: Section 3.4 of the paper.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class DualChannelIntensity(nn.Module):
    """
    Dual-channel gated intensity function.

    Equation (4):
        λ(t) = Softplus(w_tr^T m(z(t)) + w_vol^T g(v(t)) + μ_base)

    Channel 1 (Trend): Captures regular hotspots driven by semantic state z(t)
    Channel 2 (Volatility): Captures panic-driven bursts driven by variance v(t)

    The key insight is that even when z(t) is stable, high uncertainty v(t)
    can still drive increased event intensity (panic-driven posts).
    """

    def __init__(
        self,
        d_latent: int = 32,
        d_hidden: int = 64,
        use_gating: bool = True,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.use_gating = use_gating

        # Channel 1: Trend driver m(z(t))
        # Processes semantic state to capture content-driven intensity
        self.trend_net = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.SiLU(),
            nn.Linear(d_hidden // 2, 1),
        )

        # Channel 2: Volatility driver g(v(t))
        # Processes variance to capture uncertainty-driven intensity
        self.vol_net = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden // 2),
            nn.SiLU(),
            nn.Linear(d_hidden // 2, 1),
        )

        # Optional gating mechanism
        # Learns to dynamically weight the two channels
        if use_gating:
            self.gate_net = nn.Sequential(
                nn.Linear(d_latent * 2, d_hidden),
                nn.SiLU(),
                nn.Linear(d_hidden, 2),
                nn.Softmax(dim=-1),
            )

        # Base intensity (background rate)
        self.mu_base = nn.Parameter(torch.tensor(0.1))

        # Channel weights (if not using gating)
        self.w_trend = nn.Parameter(torch.tensor(1.0))
        self.w_vol = nn.Parameter(torch.tensor(1.0))

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        z: torch.Tensor,
        v: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """
        Compute event intensity λ(t).

        Args:
            z: Semantic state [batch, d_latent]
            v: Variance state [batch, d_latent]
            return_components: If True, also return individual channel outputs

        Returns:
            intensity: Event intensity λ(t) [batch]
            components: Dict with channel contributions (if return_components=True)
        """
        # Channel 1: Trend contribution
        trend_contrib = self.trend_net(z)  # [batch, 1]

        # Channel 2: Volatility contribution
        vol_contrib = self.vol_net(v)  # [batch, 1]

        if self.use_gating:
            # Dynamic gating based on both z and v
            combined = torch.cat([z, v], dim=-1)
            gates = self.gate_net(combined)  # [batch, 2]
            w_trend = gates[:, 0:1]
            w_vol = gates[:, 1:2]
        else:
            w_trend = self.w_trend
            w_vol = self.w_vol

        # Combined intensity with softplus for positivity
        pre_activation = (
            w_trend * trend_contrib +
            w_vol * vol_contrib +
            self.mu_base
        )
        intensity = nn.functional.softplus(pre_activation).squeeze(-1)

        if return_components:
            components = {
                "trend_contrib": trend_contrib.squeeze(-1),
                "vol_contrib": vol_contrib.squeeze(-1),
                "w_trend": w_trend.squeeze(-1) if self.use_gating else w_trend,
                "w_vol": w_vol.squeeze(-1) if self.use_gating else w_vol,
                "base": self.mu_base,
            }
            return intensity, components

        return intensity

    def log_intensity(self, z: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute log intensity log λ(t) for numerical stability.

        Args:
            z: Semantic state [batch, d_latent]
            v: Variance state [batch, d_latent]

        Returns:
            log_intensity: Log event intensity [batch]
        """
        intensity = self.forward(z, v)
        return torch.log(intensity + 1e-8)

    def compute_integral(
        self,
        z_trajectory: torch.Tensor,
        v_trajectory: torch.Tensor,
        times: list[float],
    ) -> torch.Tensor:
        """
        Compute integral ∫λ(t)dt using trapezoidal rule.

        Args:
            z_trajectory: Semantic states [n_steps, batch, d_latent]
            v_trajectory: Variance states [n_steps, batch, d_latent]
            times: Time points

        Returns:
            integral: ∫λ(t)dt [batch]
        """
        n_steps = len(times)
        batch_size = z_trajectory.shape[1]
        device = z_trajectory.device

        integral = torch.zeros(batch_size, device=device)

        for i in range(n_steps - 1):
            dt = times[i + 1] - times[i]

            # Intensity at endpoints
            lambda_t = self.forward(z_trajectory[i], v_trajectory[i])
            lambda_t_next = self.forward(z_trajectory[i + 1], v_trajectory[i + 1])

            # Trapezoidal rule
            integral += 0.5 * (lambda_t + lambda_t_next) * dt

        return integral


class HawkesIntensity(nn.Module):
    """
    Classical Hawkes-style intensity (baseline comparison).

    λ(t) = μ + Σ φ(t - t_i)

    This serves as a simpler baseline without volatility coupling.
    """

    def __init__(
        self,
        d_latent: int = 32,
        d_hidden: int = 64,
    ):
        super().__init__()

        self.intensity_net = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, 1),
        )

        self.mu_base = nn.Parameter(torch.tensor(0.1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute intensity based only on semantic state.

        Args:
            z: Semantic state [batch, d_latent]

        Returns:
            intensity: Event intensity [batch]
        """
        contrib = self.intensity_net(z)
        intensity = nn.functional.softplus(contrib + self.mu_base)
        return intensity.squeeze(-1)
