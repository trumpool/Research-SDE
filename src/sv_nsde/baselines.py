"""
Baseline Models for Comparison

Implements the baseline models from Section 4.2 of the paper:
1. RMTPP (Recurrent Marked Temporal Point Process)
2. Neural Hawkes Process
3. Latent ODE (Neural ODE without stochastic components)
4. Neural Jump SDE (Jia & Benson 2019 - anchor baseline)

Also includes ablation variants:
- SV-NSDE w/o Volatility Channel
- SV-NSDE with Deterministic Volatility
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math


# =============================================================================
# 1. RMTPP - Recurrent Marked Temporal Point Process
# =============================================================================

class RMTPP(nn.Module):
    """
    Recurrent Marked Temporal Point Process (Du et al., 2016).

    Uses RNN to encode history and predicts intensity as:
        λ(t) = exp(v^T h_j + w(t - t_j) + b)

    This is a deterministic state-dependent model.
    """

    def __init__(
        self,
        d_input: int = 32,
        d_hidden: int = 64,
        d_latent: int = 32,
    ):
        super().__init__()
        self.d_hidden = d_hidden
        self.d_latent = d_latent

        # Input projection
        self.input_proj = nn.Linear(d_input, d_hidden)

        # RNN for history encoding
        self.rnn = nn.GRU(
            input_size=d_hidden + 1,  # +1 for time delta
            hidden_size=d_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # Intensity parameters
        self.v = nn.Linear(d_hidden, 1)  # v^T h
        self.w = nn.Parameter(torch.tensor(-0.1))  # temporal decay
        self.b = nn.Parameter(torch.tensor(0.0))  # bias

        # Mark prediction (semantic decoder)
        self.mark_decoder = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_latent),
        )

        # Target projection for mark loss (d_input -> d_latent)
        self.mark_target_proj = nn.Linear(d_input, d_latent)

    def forward(
        self,
        event_times: torch.Tensor,
        event_marks: torch.Tensor,
        T: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            event_times: [n_events]
            event_marks: [n_events, d_input]
            T: Terminal time

        Returns:
            Dictionary with hidden states and predictions
        """
        n_events = len(event_times)
        device = event_marks.device

        # Compute inter-event times
        delta_t = torch.zeros(n_events, device=device)
        delta_t[1:] = event_times[1:] - event_times[:-1]

        # Project marks
        mark_emb = self.input_proj(event_marks)  # [n_events, d_hidden]

        # Concatenate with time deltas
        rnn_input = torch.cat([
            mark_emb,
            delta_t.unsqueeze(-1)
        ], dim=-1).unsqueeze(0)  # [1, n_events, d_hidden+1]

        # RNN forward
        hidden_states, _ = self.rnn(rnn_input)
        hidden_states = hidden_states.squeeze(0)  # [n_events, d_hidden]

        return {
            "hidden_states": hidden_states,
            "event_times": event_times,
            "T": T,
        }

    def intensity(
        self,
        h: torch.Tensor,
        t: torch.Tensor,
        t_last: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute intensity λ(t) = exp(v^T h + w(t - t_last) + b).
        """
        return torch.exp(
            self.v(h).squeeze(-1) +
            self.w * (t - t_last) +
            self.b
        )

    def compute_loss(
        self,
        event_times: torch.Tensor,
        event_marks: torch.Tensor,
        T: float,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """Compute negative log-likelihood loss."""
        outputs = self.forward(event_times, event_marks, T)
        h = outputs["hidden_states"]
        n_events = len(event_times)
        device = event_marks.device

        # Log-likelihood of events: Σ log λ(t_i)
        log_intensity_sum = torch.tensor(0.0, device=device)
        for i in range(1, n_events):
            t_last = event_times[i - 1]
            t_i = event_times[i]
            lam = self.intensity(h[i - 1:i], t_i, t_last)
            log_intensity_sum += torch.log(lam + 1e-8)

        # Integral: ∫λ(t)dt (closed form for exponential intensity)
        integral = torch.tensor(0.0, device=device)
        for i in range(n_events):
            t_start = event_times[i]
            t_end = event_times[i + 1] if i < n_events - 1 else T

            vh = self.v(h[i:i + 1]).squeeze()
            # ∫ exp(vh + w(t-t_start) + b) dt from t_start to t_end
            # = exp(vh + b) * (exp(w*(t_end-t_start)) - 1) / w
            if abs(self.w.item()) > 1e-6:
                integral += torch.exp(vh + self.b) * (
                    torch.exp(self.w * (t_end - t_start)) - 1
                ) / self.w
            else:
                integral += torch.exp(vh + self.b) * (t_end - t_start)

        # Mark prediction loss (project targets to d_latent space)
        predicted_marks = self.mark_decoder(h[:-1])  # [n-1, d_latent]
        target_marks_proj = self.mark_target_proj(event_marks[1:])  # [n-1, d_latent]
        mark_loss = F.mse_loss(predicted_marks, target_marks_proj)

        loss = -log_intensity_sum + integral + mark_loss

        return {
            "loss": loss,
            "log_intensity": log_intensity_sum,
            "integral": integral,
            "mark_loss": mark_loss,
        }

    def predict_next_time(
        self,
        h: torch.Tensor,
        t_last: torch.Tensor,
        n_samples: int = 100,
    ) -> torch.Tensor:
        """Predict next event time using thinning algorithm."""
        device = h.device
        # Simple Monte Carlo estimation
        samples = []
        for _ in range(n_samples):
            t = t_last.clone()
            max_lam = self.intensity(h, t, t_last).item() * 2

            while True:
                dt = torch.distributions.Exponential(max_lam).sample()
                t = t + dt
                lam = self.intensity(h, t, t_last)
                if torch.rand(1).item() < lam.item() / max_lam:
                    samples.append(t.item())
                    break
                if t.item() - t_last.item() > 100:  # Safety bound
                    samples.append(t.item())
                    break

        return torch.tensor(samples).mean()


# =============================================================================
# 2. Neural Hawkes Process
# =============================================================================

class NeuralHawkes(nn.Module):
    """
    Neural Hawkes Process (Mei & Eisner, 2017).

    Uses continuous-time LSTM with intensity:
        λ(t) = softplus(w^T c(t) + b)

    where c(t) is the cell state with exponential decay.
    """

    def __init__(
        self,
        d_input: int = 32,
        d_hidden: int = 64,
        d_latent: int = 32,
    ):
        super().__init__()
        self.d_hidden = d_hidden

        # Input projection
        self.input_proj = nn.Linear(d_input, d_hidden)

        # Continuous-time LSTM components
        self.W_i = nn.Linear(d_hidden, d_hidden)
        self.W_f = nn.Linear(d_hidden, d_hidden)
        self.W_o = nn.Linear(d_hidden, d_hidden)
        self.W_z = nn.Linear(d_hidden, d_hidden)

        self.U_i = nn.Linear(d_hidden, d_hidden, bias=False)
        self.U_f = nn.Linear(d_hidden, d_hidden, bias=False)
        self.U_o = nn.Linear(d_hidden, d_hidden, bias=False)
        self.U_z = nn.Linear(d_hidden, d_hidden, bias=False)

        # Decay parameter
        self.W_d = nn.Linear(d_hidden, d_hidden)
        self.U_d = nn.Linear(d_hidden, d_hidden, bias=False)

        # Intensity
        self.intensity_layer = nn.Linear(d_hidden, 1)
        self.base_intensity = nn.Parameter(torch.tensor(0.1))

        # Mark decoder
        self.mark_decoder = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_latent),
        )

        # Target projection for mark loss (d_input -> d_latent)
        self.mark_target_proj = nn.Linear(d_input, d_latent)

    def forward(
        self,
        event_times: torch.Tensor,
        event_marks: torch.Tensor,
        T: float,
    ) -> Dict[str, torch.Tensor]:
        n_events = len(event_times)
        device = event_marks.device

        # Initialize hidden and cell states
        h = torch.zeros(self.d_hidden, device=device)
        c = torch.zeros(self.d_hidden, device=device)
        c_bar = torch.zeros(self.d_hidden, device=device)

        hidden_states = []
        cell_states = []
        decays = []

        for i in range(n_events):
            x = self.input_proj(event_marks[i])

            # Decay from last event
            if i > 0:
                dt = event_times[i] - event_times[i - 1]
                c = c_bar + (c - c_bar) * torch.exp(-decays[-1] * dt)

            # LSTM update
            i_gate = torch.sigmoid(self.W_i(x) + self.U_i(h))
            f_gate = torch.sigmoid(self.W_f(x) + self.U_f(h))
            o_gate = torch.sigmoid(self.W_o(x) + self.U_o(h))
            z = torch.tanh(self.W_z(x) + self.U_z(h))

            c_new = f_gate * c + i_gate * z
            c_bar = f_gate * c_bar + i_gate * z

            decay = F.softplus(self.W_d(x) + self.U_d(h))

            h = o_gate * torch.tanh(c_new)

            hidden_states.append(h)
            cell_states.append(c_new)
            decays.append(decay)

            c = c_new

        return {
            "hidden_states": torch.stack(hidden_states),
            "cell_states": torch.stack(cell_states),
            "decays": torch.stack(decays),
            "c_bar": c_bar,
        }

    def intensity_at_time(
        self,
        c: torch.Tensor,
        c_bar: torch.Tensor,
        decay: torch.Tensor,
        dt: torch.Tensor,
    ) -> torch.Tensor:
        """Compute intensity at time t_last + dt."""
        c_t = c_bar + (c - c_bar) * torch.exp(-decay * dt)
        return F.softplus(self.intensity_layer(c_t) + self.base_intensity)

    def compute_loss(
        self,
        event_times: torch.Tensor,
        event_marks: torch.Tensor,
        T: float,
        n_mc_samples: int = 20,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.forward(event_times, event_marks, T)
        h = outputs["hidden_states"]
        c = outputs["cell_states"]
        decays = outputs["decays"]
        c_bar = outputs["c_bar"]

        n_events = len(event_times)
        device = event_marks.device

        # Log-likelihood at event times
        log_intensity_sum = torch.tensor(0.0, device=device)
        for i in range(1, n_events):
            dt = event_times[i] - event_times[i - 1]
            lam = self.intensity_at_time(c[i - 1], c_bar, decays[i - 1], dt)
            log_intensity_sum += torch.log(lam.squeeze() + 1e-8)

        # Integral via Monte Carlo
        integral = torch.tensor(0.0, device=device)
        for i in range(n_events):
            t_start = event_times[i]
            t_end = event_times[i + 1] if i < n_events - 1 else T
            interval = t_end - t_start

            if interval > 0:
                mc_times = torch.rand(n_mc_samples, device=device) * interval
                for dt in mc_times:
                    lam = self.intensity_at_time(c[i], c_bar, decays[i], dt)
                    integral += lam.squeeze() * interval / n_mc_samples

        # Mark loss (project targets to d_latent space)
        predicted_marks = self.mark_decoder(h[:-1])  # [n-1, d_latent]
        target_marks_proj = self.mark_target_proj(event_marks[1:])  # [n-1, d_latent]
        mark_loss = F.mse_loss(predicted_marks, target_marks_proj)

        loss = -log_intensity_sum + integral + mark_loss

        return {
            "loss": loss,
            "log_intensity": log_intensity_sum,
            "integral": integral,
            "mark_loss": mark_loss,
        }


# =============================================================================
# 3. Latent ODE (Neural ODE without stochastic components)
# =============================================================================

class LatentODE(nn.Module):
    """
    Latent ODE for point processes.

    Uses deterministic Neural ODE for state evolution:
        dz/dt = f_θ(z, t)

    No stochastic component - serves as ablation for the SDE approach.
    """

    def __init__(
        self,
        d_input: int = 32,
        d_latent: int = 32,
        d_hidden: int = 64,
    ):
        super().__init__()
        self.d_latent = d_latent

        # Input projection
        self.input_proj = nn.Linear(d_input, d_latent)

        # ODE function f_θ(z, t)
        self.ode_func = nn.Sequential(
            nn.Linear(d_latent + 1, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_latent),
        )

        # Jump network
        self.jump_net = nn.Sequential(
            nn.Linear(d_latent * 2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_latent),
            nn.Tanh(),
        )
        self.jump_scale = nn.Parameter(torch.tensor(0.5))

        # Intensity
        self.intensity_net = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )
        self.base_intensity = nn.Parameter(torch.tensor(0.1))

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_latent),
        )

        # Initial state
        self.z0 = nn.Parameter(torch.zeros(d_latent))

    def ode_step(self, z: torch.Tensor, t: torch.Tensor, dt: float) -> torch.Tensor:
        """Euler step for ODE."""
        t_input = t.unsqueeze(-1) if t.dim() == 0 else t
        if t_input.dim() == 1:
            t_input = t_input.unsqueeze(0)
        t_input = t_input.expand(z.shape[0], 1)

        zt = torch.cat([z, t_input], dim=-1)
        dz = self.ode_func(zt)
        return z + dz * dt

    def intensity(self, z: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.intensity_net(z) + self.base_intensity).squeeze(-1)

    def forward(
        self,
        event_times: torch.Tensor,
        event_marks: torch.Tensor,
        T: float,
        dt: float = 0.01,
    ) -> Dict[str, torch.Tensor]:
        device = event_marks.device
        n_events = len(event_times)

        # Project marks
        marks_proj = self.input_proj(event_marks)

        # Initialize
        z = self.z0.unsqueeze(0)  # [1, d_latent]

        z_at_events = []
        trajectory = [z]
        times = [0.0]

        t = 0.0
        event_idx = 0

        while t < T:
            next_t = t + dt

            if event_idx < n_events and event_times[event_idx] <= next_t:
                # Step to event
                event_t = event_times[event_idx].item()
                if event_t > t:
                    z = self.ode_step(z, torch.tensor(t, device=device), event_t - t)

                z_at_events.append(z.clone())

                # Apply jump
                x_i = marks_proj[event_idx:event_idx + 1]
                jump = self.jump_scale * self.jump_net(torch.cat([z, x_i], dim=-1))
                z = z + jump

                t = event_t
                event_idx += 1
            else:
                z = self.ode_step(z, torch.tensor(t, device=device), dt)
                t = next_t

            trajectory.append(z)
            times.append(t)

        return {
            "z_events": torch.stack(z_at_events).squeeze(1) if z_at_events else None,
            "z_trajectory": torch.stack(trajectory).squeeze(1),
            "times": times,
        }

    def compute_loss(
        self,
        event_times: torch.Tensor,
        event_marks: torch.Tensor,
        T: float,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.forward(event_times, event_marks, T)
        z_events = outputs["z_events"]
        z_traj = outputs["z_trajectory"]
        times = outputs["times"]

        device = event_marks.device
        n_events = len(event_times)
        marks_proj = self.input_proj(event_marks)

        # Log intensity at events
        log_intensity_sum = torch.tensor(0.0, device=device)
        if z_events is not None:
            for i in range(n_events):
                lam = self.intensity(z_events[i:i + 1])
                log_intensity_sum += torch.log(lam + 1e-8)

        # Integral
        integral = torch.tensor(0.0, device=device)
        for i in range(len(times) - 1):
            dt = times[i + 1] - times[i]
            lam = self.intensity(z_traj[i:i + 1])
            integral += lam * dt

        # Reconstruction
        recon_loss = torch.tensor(0.0, device=device)
        if z_events is not None:
            decoded = self.decoder(z_events)
            recon_loss = F.mse_loss(decoded, marks_proj)

        loss = -log_intensity_sum + integral + recon_loss

        return {
            "loss": loss,
            "log_intensity": log_intensity_sum,
            "integral": integral,
            "reconstruction": recon_loss,
        }


# =============================================================================
# 4. Neural Jump SDE (Jia & Benson 2019)
# =============================================================================

class NeuralJumpSDE(nn.Module):
    """
    Neural Jump Stochastic Differential Equation (Jia & Benson, 2019).

    Standard Neural SDE with jumps but WITHOUT decoupled volatility:
        dz(t) = f_θ(z,t)dt + g_θ(z,t)dW(t) + J(z,x)dN(t)

    Key difference from SV-NSDE: volatility is state-dependent g(z),
    not a separate stochastic process v(t).
    """

    def __init__(
        self,
        d_input: int = 32,
        d_latent: int = 32,
        d_hidden: int = 64,
    ):
        super().__init__()
        self.d_latent = d_latent

        # Input projection
        self.input_proj = nn.Linear(d_input, d_latent)

        # Drift f_θ(z, t)
        self.drift_net = nn.Sequential(
            nn.Linear(d_latent + 1, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_latent),
        )

        # Diffusion g_θ(z, t) - state-dependent, NOT separate process
        self.diffusion_net = nn.Sequential(
            nn.Linear(d_latent + 1, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_latent),
            nn.Softplus(),  # Ensure positive
        )

        # Jump network
        self.jump_net = nn.Sequential(
            nn.Linear(d_latent * 2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_latent),
            nn.Tanh(),
        )
        self.jump_scale = nn.Parameter(torch.tensor(0.5))

        # Intensity (only depends on z, not separate v)
        self.intensity_net = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )
        self.base_intensity = nn.Parameter(torch.tensor(0.1))

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_latent),
        )

        # Initial state
        self.z0 = nn.Parameter(torch.zeros(d_latent))

    def sde_step(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Euler-Maruyama step."""
        t_input = t.unsqueeze(-1).expand(z.shape[0], 1)
        zt = torch.cat([z, t_input], dim=-1)

        drift = self.drift_net(zt)
        diffusion = self.diffusion_net(zt)

        dW = torch.randn_like(z) * math.sqrt(dt)
        return z + drift * dt + diffusion * dW

    def intensity(self, z: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.intensity_net(z) + self.base_intensity).squeeze(-1)

    def forward(
        self,
        event_times: torch.Tensor,
        event_marks: torch.Tensor,
        T: float,
        dt: float = 0.01,
        n_samples: int = 1,
    ) -> Dict[str, torch.Tensor]:
        device = event_marks.device
        n_events = len(event_times)

        marks_proj = self.input_proj(event_marks)

        # Initialize
        z = self.z0.unsqueeze(0).expand(n_samples, -1)

        z_at_events = []
        trajectory = [z]
        times = [0.0]

        t = 0.0
        event_idx = 0

        while t < T:
            next_t = t + dt

            if event_idx < n_events and event_times[event_idx] <= next_t:
                event_t = event_times[event_idx].item()
                if event_t > t:
                    z = self.sde_step(z, torch.tensor(t, device=device), event_t - t)

                z_at_events.append(z.clone())

                x_i = marks_proj[event_idx:event_idx + 1].expand(n_samples, -1)
                jump = self.jump_scale * self.jump_net(torch.cat([z, x_i], dim=-1))
                z = z + jump

                t = event_t
                event_idx += 1
            else:
                z = self.sde_step(z, torch.tensor(t, device=device), dt)
                t = next_t

            trajectory.append(z)
            times.append(t)

        return {
            "z_events": torch.stack(z_at_events) if z_at_events else None,
            "z_trajectory": torch.stack(trajectory),
            "times": times,
        }

    def compute_loss(
        self,
        event_times: torch.Tensor,
        event_marks: torch.Tensor,
        T: float,
        n_samples: int = 1,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.forward(event_times, event_marks, T, n_samples=n_samples)
        z_events = outputs["z_events"]
        z_traj = outputs["z_trajectory"]
        times = outputs["times"]

        device = event_marks.device
        n_events = len(event_times)
        marks_proj = self.input_proj(event_marks)

        # Log intensity
        log_intensity_sum = torch.tensor(0.0, device=device)
        if z_events is not None:
            for i in range(n_events):
                lam = self.intensity(z_events[i])
                log_intensity_sum += torch.log(lam + 1e-8).mean()

        # Integral
        integral = torch.tensor(0.0, device=device)
        for i in range(len(times) - 1):
            dt = times[i + 1] - times[i]
            lam = self.intensity(z_traj[i])
            integral += lam.mean() * dt

        # Reconstruction
        recon_loss = torch.tensor(0.0, device=device)
        if z_events is not None:
            decoded = self.decoder(z_events.mean(dim=1))
            recon_loss = F.mse_loss(decoded, marks_proj)

        # KL (simplified)
        kl_loss = 0.01 * (z_traj ** 2).mean()

        loss = -log_intensity_sum + integral + recon_loss + kl_loss

        return {
            "loss": loss,
            "log_intensity": log_intensity_sum,
            "integral": integral,
            "reconstruction": recon_loss,
            "kl": kl_loss,
        }


# =============================================================================
# 5. Ablation Variants
# =============================================================================

class SVNSDENoVolatilityChannel(nn.Module):
    """
    SV-NSDE without the volatility channel in intensity.

    Ablation: removes w_vol^T g(v(t)) from the intensity function.
    Tests whether volatility information helps predict bursts.
    """

    def __init__(
        self,
        d_input: int = 32,
        d_latent: int = 32,
        d_hidden: int = 64,
    ):
        super().__init__()
        from .sde import NeuralHestonSDE
        from .decoder import SemanticDecoder

        self.d_latent = d_latent
        self.input_proj = nn.Linear(d_input, d_latent)
        self.sde = NeuralHestonSDE(d_latent=d_latent, d_hidden=d_hidden)
        self.decoder = SemanticDecoder(d_latent=d_latent)

        # Intensity WITHOUT volatility channel
        self.intensity_net = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )
        self.base_intensity = nn.Parameter(torch.tensor(0.1))

        self.dt = 0.01
        self.kl_weight = 0.01

    def intensity(self, z: torch.Tensor, v: torch.Tensor = None) -> torch.Tensor:
        """Intensity depends only on z, ignores v."""
        return F.softplus(self.intensity_net(z) + self.base_intensity).squeeze(-1)

    def compute_loss(
        self,
        event_times: torch.Tensor,
        event_marks: torch.Tensor,
        T: float,
        n_samples: int = 1,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        device = event_marks.device
        marks_proj = self.input_proj(event_marks)
        n_events = len(event_times)

        sde_output = self.sde.solve(
            event_times=event_times,
            event_marks=marks_proj,
            T=T,
            dt=self.dt,
            batch_size=n_samples,
            return_full_trajectory=True,
        )

        z_events = sde_output["z_events"]
        z_traj = sde_output["z_trajectory"]
        times = sde_output["times"]

        # Log intensity (no v channel)
        log_intensity_sum = torch.tensor(0.0, device=device)
        if z_events is not None:
            for i in range(n_events):
                lam = self.intensity(z_events[i])
                log_intensity_sum += torch.log(lam + 1e-8).mean()

        # Integral
        integral = torch.tensor(0.0, device=device)
        for i in range(len(times) - 1):
            dt = times[i + 1] - times[i]
            lam = self.intensity(z_traj[i])
            integral += lam.mean() * dt

        # Reconstruction
        recon_loss = torch.tensor(0.0, device=device)
        if z_events is not None:
            for i in range(n_events):
                x_i = marks_proj[i].unsqueeze(0).expand(n_samples, -1)
                log_p = self.decoder.log_prob(x_i, z_events[i])
                recon_loss += log_p.mean()

        kl_loss = self.kl_weight * (z_traj ** 2).mean()

        elbo = log_intensity_sum - integral + recon_loss - kl_loss
        loss = -elbo

        return {
            "loss": loss,
            "log_intensity": log_intensity_sum,
            "integral": integral,
            "reconstruction": recon_loss,
            "kl": kl_loss,
        }


class SVNSDEDeterministicVolatility(nn.Module):
    """
    SV-NSDE with deterministic volatility.

    Ablation: v(t) = h(z(t)) instead of stochastic CIR process.
    Tests whether stochastic volatility modeling is necessary.
    """

    def __init__(
        self,
        d_input: int = 32,
        d_latent: int = 32,
        d_hidden: int = 64,
    ):
        super().__init__()
        from .intensity import DualChannelIntensity
        from .decoder import SemanticDecoder

        self.d_latent = d_latent
        self.input_proj = nn.Linear(d_input, d_latent)

        # ODE for z (no stochastic term)
        self.drift_net = nn.Sequential(
            nn.Linear(d_latent + 1, d_hidden),
            nn.Tanh(),
            nn.Linear(d_hidden, d_latent),
        )

        # Deterministic volatility: v = h(z)
        self.vol_net = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_latent),
            nn.Softplus(),
        )

        # Jump
        self.jump_net = nn.Sequential(
            nn.Linear(d_latent * 2, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_latent),
            nn.Tanh(),
        )
        self.jump_scale = nn.Parameter(torch.tensor(0.5))

        self.intensity = DualChannelIntensity(d_latent=d_latent, d_hidden=d_hidden)
        self.decoder = SemanticDecoder(d_latent=d_latent)

        self.z0 = nn.Parameter(torch.zeros(d_latent))
        self.dt = 0.01
        self.kl_weight = 0.01

    def compute_loss(
        self,
        event_times: torch.Tensor,
        event_marks: torch.Tensor,
        T: float,
        n_samples: int = 1,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        device = event_marks.device
        marks_proj = self.input_proj(event_marks)
        n_events = len(event_times)

        z = self.z0.unsqueeze(0)
        z_at_events = []
        v_at_events = []
        trajectory_z = [z]
        trajectory_v = [self.vol_net(z)]
        times = [0.0]

        t = 0.0
        event_idx = 0

        while t < T:
            next_t = t + self.dt

            if event_idx < n_events and event_times[event_idx] <= next_t:
                event_t = event_times[event_idx].item()
                if event_t > t:
                    t_tensor = torch.tensor([[t]], device=device)
                    zt = torch.cat([z, t_tensor], dim=-1)
                    z = z + self.drift_net(zt) * (event_t - t)

                z_at_events.append(z.clone())
                v_at_events.append(self.vol_net(z))

                x_i = marks_proj[event_idx:event_idx + 1]
                jump = self.jump_scale * self.jump_net(torch.cat([z, x_i], dim=-1))
                z = z + jump

                t = event_t
                event_idx += 1
            else:
                t_tensor = torch.tensor([[t]], device=device)
                zt = torch.cat([z, t_tensor], dim=-1)
                z = z + self.drift_net(zt) * self.dt
                t = next_t

            trajectory_z.append(z)
            trajectory_v.append(self.vol_net(z))
            times.append(t)

        z_events = torch.stack(z_at_events).squeeze(1) if z_at_events else None
        v_events = torch.stack(v_at_events).squeeze(1) if v_at_events else None
        z_traj = torch.stack(trajectory_z).squeeze(1)
        v_traj = torch.stack(trajectory_v).squeeze(1)

        # Losses
        log_intensity_sum = torch.tensor(0.0, device=device)
        if z_events is not None:
            for i in range(n_events):
                lam = self.intensity(z_events[i:i + 1], v_events[i:i + 1])
                log_intensity_sum += torch.log(lam + 1e-8)

        integral = self.intensity.compute_integral(
            z_traj.unsqueeze(1), v_traj.unsqueeze(1), times
        ).mean()

        recon_loss = torch.tensor(0.0, device=device)
        if z_events is not None:
            for i in range(n_events):
                log_p = self.decoder.log_prob(marks_proj[i:i + 1], z_events[i:i + 1])
                recon_loss += log_p

        kl_loss = self.kl_weight * (z_traj ** 2).mean()

        elbo = log_intensity_sum - integral + recon_loss - kl_loss
        loss = -elbo

        return {
            "loss": loss,
            "log_intensity": log_intensity_sum,
            "integral": integral,
            "reconstruction": recon_loss,
            "kl": kl_loss,
        }


# =============================================================================
# Model Registry
# =============================================================================

BASELINE_MODELS = {
    "rmtpp": RMTPP,
    "neural_hawkes": NeuralHawkes,
    "latent_ode": LatentODE,
    "neural_jump_sde": NeuralJumpSDE,
    "sv_nsde_no_vol": SVNSDENoVolatilityChannel,
    "sv_nsde_det_vol": SVNSDEDeterministicVolatility,
}


def get_baseline(name: str, **kwargs) -> nn.Module:
    """Get a baseline model by name."""
    if name not in BASELINE_MODELS:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(BASELINE_MODELS.keys())}")
    return BASELINE_MODELS[name](**kwargs)
