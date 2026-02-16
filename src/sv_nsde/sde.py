"""
Neural Heston SDE Module

Implements the coupled SDE dynamics for semantic trajectory and volatility.
Reference: Section 3.2 and 3.3 of the paper.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class SDEConfig:
    """Configuration for Heston SDE."""
    d_latent: int = 32
    d_hidden: int = 64
    kappa_init: float = 1.0      # Mean reversion speed
    theta_init: float = 0.1      # Long-term variance
    xi_init: float = 0.3         # Vol of vol
    rho: float = -0.5            # Correlation between Brownian motions
    min_variance: float = 1e-6   # Minimum variance for numerical stability


class HestonSDEFunc(nn.Module):
    """
    Heston-style coupled SDE dynamics.

    The system of SDEs (Equation 3):
        dz(t) = μ_θ(z,t)dt + √v(t) ⊙ dW_z(t) + J_φ(z(t-), x_i)dN(t)
        dv(t) = κ(θ - v(t))dt + ξ√v(t) ⊙ dW_v(t)

    Where:
        - z(t): Semantic trajectory (main opinion state)
        - v(t): Instantaneous variance (volatility process)
        - μ_θ: Neural drift network
        - J_φ: Neural jump network
        - κ, θ, ξ: CIR process parameters
    """

    def __init__(self, config: Optional[SDEConfig] = None):
        super().__init__()
        if config is None:
            config = SDEConfig()
        self.config = config
        d_latent = config.d_latent
        d_hidden = config.d_hidden

        # Drift network μ_θ for z(t)
        # Maps (z, t) -> drift direction
        self.drift_net = nn.Sequential(
            nn.Linear(d_latent + 1, d_hidden),  # +1 for time encoding
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_latent),
        )

        # CIR parameters for v(t) variance process
        # These are learnable to adapt to different crisis dynamics
        self.log_kappa = nn.Parameter(torch.tensor(config.kappa_init).log())
        self.log_theta = nn.Parameter(torch.tensor(config.theta_init).log())
        self.log_xi = nn.Parameter(torch.tensor(config.xi_init).log())

        # Jump network J_φ: processes event information
        # Maps (z, x) -> state update
        self.jump_net = nn.Sequential(
            nn.Linear(d_latent * 2, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_latent),
            nn.Tanh(),  # Bounded jumps for stability
        )

        # Jump scale (learnable magnitude)
        self.jump_scale = nn.Parameter(torch.tensor(0.5))

        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def kappa(self) -> torch.Tensor:
        """Mean reversion speed (positive)."""
        return self.log_kappa.exp()

    @property
    def theta(self) -> torch.Tensor:
        """Long-term variance (positive)."""
        return self.log_theta.exp()

    @property
    def xi(self) -> torch.Tensor:
        """Volatility of volatility (positive)."""
        return self.log_xi.exp()

    def drift_z(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute drift for semantic trajectory z(t).

        Args:
            z: Current state [batch, d_latent]
            t: Current time [batch, 1] or scalar

        Returns:
            drift: Drift vector [batch, d_latent]
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        t = t.expand(z.shape[0], 1)

        zt = torch.cat([z, t], dim=-1)
        return self.drift_net(zt)

    def drift_v(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute CIR drift for variance process v(t).

        Implements mean reversion: κ(θ - v)

        Args:
            v: Current variance [batch, d_latent]

        Returns:
            drift: Variance drift [batch, d_latent]
        """
        return self.kappa * (self.theta - v)

    def diffusion_z(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion coefficient for z(t).

        Returns √v(t) for stochastic volatility coupling.

        Args:
            v: Current variance [batch, d_latent]

        Returns:
            diffusion: Diffusion coefficient [batch, d_latent]
        """
        return torch.sqrt(torch.clamp(v, min=self.config.min_variance))

    def diffusion_v(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute diffusion coefficient for v(t).

        Returns ξ√v(t) (vol of vol term).

        Args:
            v: Current variance [batch, d_latent]

        Returns:
            diffusion: Variance diffusion [batch, d_latent]
        """
        return self.xi * torch.sqrt(torch.clamp(v, min=self.config.min_variance))

    def jump(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Compute jump update J_φ(z(t-), x_i).

        When an event arrives, this computes the instantaneous
        state update based on current state and event content.

        Args:
            z: Current state before jump [batch, d_latent]
            x: Event semantic vector [batch, d_latent]

        Returns:
            delta_z: State update [batch, d_latent]
        """
        combined = torch.cat([z, x], dim=-1)
        return self.jump_scale * self.jump_net(combined)


class NeuralHestonSDE(nn.Module):
    """
    Full Neural Heston SDE solver with jump-diffusion.

    Solves the coupled SDE system using Euler-Maruyama scheme
    with event-driven jumps.
    """

    def __init__(
        self,
        d_latent: int = 32,
        d_hidden: int = 64,
        config: Optional[SDEConfig] = None,
    ):
        super().__init__()
        if config is None:
            config = SDEConfig(d_latent=d_latent, d_hidden=d_hidden)
        self.config = config
        self.d_latent = config.d_latent

        self.sde_func = HestonSDEFunc(config)

        # Learnable initial conditions
        self.z0_mean = nn.Parameter(torch.zeros(d_latent))
        self.z0_logvar = nn.Parameter(torch.zeros(d_latent))
        self.v0_mean = nn.Parameter(torch.ones(d_latent) * -2)  # exp(-2) ≈ 0.1

    def get_initial_state(
        self,
        batch_size: int,
        device: torch.device,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample initial state (z0, v0).

        Args:
            batch_size: Number of samples
            device: Target device
            deterministic: If True, return mean without sampling

        Returns:
            z0: Initial semantic state [batch, d_latent]
            v0: Initial variance [batch, d_latent]
        """
        z0_mean = self.z0_mean.unsqueeze(0).expand(batch_size, -1)
        v0_mean = self.v0_mean.exp().unsqueeze(0).expand(batch_size, -1)

        if deterministic:
            return z0_mean.to(device), v0_mean.to(device)

        # Sample z0 from Gaussian
        z0_std = (0.5 * self.z0_logvar).exp()
        z0 = z0_mean + z0_std * torch.randn(batch_size, self.d_latent, device=device)

        # v0 is deterministic (could also be sampled from inverse gamma)
        v0 = v0_mean.to(device)

        return z0, v0

    def euler_maruyama_step(
        self,
        z: torch.Tensor,
        v: torch.Tensor,
        t: torch.Tensor,
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single Euler-Maruyama step for the coupled SDE.

        Args:
            z: Current semantic state [batch, d_latent]
            v: Current variance [batch, d_latent]
            t: Current time
            dt: Time step

        Returns:
            z_next: Next semantic state
            v_next: Next variance
        """
        sqrt_dt = torch.sqrt(torch.tensor(dt, device=z.device))

        # Sample Brownian increments
        dW_z = torch.randn_like(z) * sqrt_dt
        dW_v = torch.randn_like(v) * sqrt_dt

        # Update z: dz = μdt + √v dW_z
        drift_z = self.sde_func.drift_z(z, t)
        diff_z = self.sde_func.diffusion_z(v)
        z_next = z + drift_z * dt + diff_z * dW_z

        # Update v: dv = κ(θ-v)dt + ξ√v dW_v
        drift_v = self.sde_func.drift_v(v)
        diff_v = self.sde_func.diffusion_v(v)
        v_next = v + drift_v * dt + diff_v * dW_v

        # Ensure variance stays positive (reflection at boundary)
        v_next = torch.clamp(v_next, min=self.config.min_variance)

        return z_next, v_next

    def solve(
        self,
        event_times: torch.Tensor,
        event_marks: torch.Tensor,
        T: float,
        dt: float = 0.01,
        batch_size: int = 1,
        return_full_trajectory: bool = False,
    ) -> dict:
        """
        Solve the coupled SDE with jump-diffusion.

        Args:
            event_times: Event timestamps [n_events]
            event_marks: Event semantic vectors [n_events, d_latent]
            T: Terminal time
            dt: Discretization step
            batch_size: Number of sample paths
            return_full_trajectory: Whether to return full trajectory

        Returns:
            Dictionary containing:
                - z_events: States at event times [n_events, batch, d_latent]
                - v_events: Variances at event times [n_events, batch, d_latent]
                - z_trajectory: Full trajectory (if requested)
                - v_trajectory: Full trajectory (if requested)
                - times: Time grid
        """
        device = event_marks.device
        n_events = len(event_times)

        # Initialize
        z, v = self.get_initial_state(batch_size, device)

        # Storage
        z_at_events = []
        v_at_events = []

        if return_full_trajectory:
            trajectory_z = [z.clone()]
            trajectory_v = [v.clone()]
            trajectory_times = [0.0]

        # Sort events by time
        sorted_idx = torch.argsort(event_times)
        event_times = event_times[sorted_idx]
        event_marks = event_marks[sorted_idx]

        t = 0.0
        event_idx = 0

        while t < T:
            next_t = t + dt

            # Check if event occurs before next_t
            if event_idx < n_events and event_times[event_idx] <= next_t:
                # Step to event time
                event_t = event_times[event_idx].item()
                if event_t > t:
                    dt_to_event = event_t - t
                    z, v = self.euler_maruyama_step(z, v, torch.tensor(t, device=z.device), dt_to_event)

                # Store pre-jump state
                z_at_events.append(z.clone())
                v_at_events.append(v.clone())

                # Apply jump
                x_i = event_marks[event_idx].unsqueeze(0).expand(batch_size, -1)
                z = z + self.sde_func.jump(z, x_i)

                t = event_t
                event_idx += 1
            else:
                # Regular diffusion step
                z, v = self.euler_maruyama_step(z, v, torch.tensor(t, device=z.device), dt)
                t = next_t

            if return_full_trajectory:
                trajectory_z.append(z.clone())
                trajectory_v.append(v.clone())
                trajectory_times.append(t)

        result = {
            "z_events": torch.stack(z_at_events) if z_at_events else None,
            "v_events": torch.stack(v_at_events) if v_at_events else None,
            "z_final": z,
            "v_final": v,
        }

        if return_full_trajectory:
            result["z_trajectory"] = torch.stack(trajectory_z)
            result["v_trajectory"] = torch.stack(trajectory_v)
            result["times"] = trajectory_times

        return result

    def sample_paths(
        self,
        n_paths: int,
        T: float,
        dt: float = 0.01,
        device: torch.device = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
        """
        Sample pure diffusion paths without jumps (for visualization).

        Args:
            n_paths: Number of sample paths
            T: Terminal time
            dt: Time step
            device: Target device

        Returns:
            z_paths: Semantic trajectories [n_steps, n_paths, d_latent]
            v_paths: Variance trajectories [n_steps, n_paths, d_latent]
            times: Time grid
        """
        if device is None:
            device = next(self.parameters()).device

        z, v = self.get_initial_state(n_paths, device)

        z_paths = [z]
        v_paths = [v]
        times = [0.0]

        t = 0.0
        while t < T:
            z, v = self.euler_maruyama_step(z, v, torch.tensor(t, device=device), dt)
            z_paths.append(z)
            v_paths.append(v)
            t += dt
            times.append(t)

        return torch.stack(z_paths), torch.stack(v_paths), times
