"""
Semantic Decoder Module

Implements the emission model for reconstructing semantic vectors.
Reference: Section 3.5 of the paper.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
import math


class SemanticDecoder(nn.Module):
    """
    Conditional Gaussian emission model.

    Equation (5):
        p(x|z(t)) = N(x; μ_dec(z(t)), σ²_obs I)

    This decoder ensures the latent SDE trajectory z(t) captures
    meaningful semantic information by reconstructing the original
    text embeddings.

    This extends the model into a VAE-like framework, preventing
    posterior collapse where z(t) might lose correlation with x.
    """

    def __init__(
        self,
        d_latent: int = 32,
        d_hidden: int = 128,
        sigma_obs: float = 0.1,
        learn_variance: bool = False,
    ):
        super().__init__()
        self.d_latent = d_latent
        self.learn_variance = learn_variance

        # Decoder network μ_dec(z(t))
        # Maps latent state back to observation space
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_latent),
        )

        # Observation noise variance
        if learn_variance:
            self.log_sigma_obs = nn.Parameter(torch.tensor(sigma_obs).log())
        else:
            self.register_buffer(
                "log_sigma_obs",
                torch.tensor(sigma_obs).log()
            )

        self._init_weights()

    def _init_weights(self):
        """Initialize decoder weights."""
        for module in self.decoder.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    @property
    def sigma_obs(self) -> torch.Tensor:
        """Observation noise standard deviation."""
        return self.log_sigma_obs.exp()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent state to observation space mean.

        Args:
            z: Latent state [batch, d_latent]

        Returns:
            mu: Predicted observation mean [batch, d_latent]
        """
        return self.decoder(z)

    def log_prob(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        reduce: str = "sum",
    ) -> torch.Tensor:
        """
        Compute log probability log p(x|z(t)).

        Gaussian log-likelihood:
            log p(x|z) = -d/2 log(2π) - d log(σ) - ||x - μ(z)||² / (2σ²)

        Args:
            x: Observed semantic vector [batch, d_latent]
            z: Latent state [batch, d_latent]
            reduce: Reduction mode ("sum", "mean", "none")

        Returns:
            log_prob: Log probability [batch] or [batch, d_latent]
        """
        mu = self.forward(z)
        var = self.sigma_obs ** 2

        # Gaussian log-likelihood per dimension
        log_prob_per_dim = (
            -0.5 * math.log(2 * math.pi)
            - self.log_sigma_obs
            - 0.5 * (x - mu) ** 2 / var
        )

        if reduce == "sum":
            return log_prob_per_dim.sum(dim=-1)
        elif reduce == "mean":
            return log_prob_per_dim.mean(dim=-1)
        else:
            return log_prob_per_dim

    def sample(
        self,
        z: torch.Tensor,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """
        Sample from the emission distribution.

        Args:
            z: Latent state [batch, d_latent]
            n_samples: Number of samples

        Returns:
            samples: Sampled observations [n_samples, batch, d_latent]
        """
        mu = self.forward(z)
        sigma = self.sigma_obs

        if n_samples == 1:
            return mu + sigma * torch.randn_like(mu)

        samples = []
        for _ in range(n_samples):
            samples.append(mu + sigma * torch.randn_like(mu))

        return torch.stack(samples)

    def reconstruction_loss(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss (negative log-likelihood).

        Args:
            x: Observed semantic vectors [batch, d_latent]
            z: Latent states [batch, d_latent]

        Returns:
            loss: Reconstruction loss (scalar)
        """
        return -self.log_prob(x, z, reduce="sum").mean()


class ConditionalDecoder(nn.Module):
    """
    Extended decoder conditioned on both z(t) and v(t).

    This variant allows the decoder to also consider the uncertainty
    state, potentially improving reconstruction during high-volatility
    periods.
    """

    def __init__(
        self,
        d_latent: int = 32,
        d_hidden: int = 128,
        sigma_obs: float = 0.1,
    ):
        super().__init__()
        self.d_latent = d_latent

        # Decoder takes both z and v as input
        self.decoder = nn.Sequential(
            nn.Linear(d_latent * 2, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.SiLU(),
            nn.Linear(d_hidden, d_latent),
        )

        # Variance prediction network
        self.var_net = nn.Sequential(
            nn.Linear(d_latent * 2, d_hidden // 2),
            nn.SiLU(),
            nn.Linear(d_hidden // 2, d_latent),
            nn.Softplus(),
        )

        self.min_var = sigma_obs ** 2

    def forward(
        self,
        z: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode to observation space with heteroscedastic variance.

        Args:
            z: Semantic state [batch, d_latent]
            v: Variance state [batch, d_latent]

        Returns:
            mu: Predicted mean [batch, d_latent]
            var: Predicted variance [batch, d_latent]
        """
        combined = torch.cat([z, v], dim=-1)
        mu = self.decoder(combined)
        var = self.var_net(combined) + self.min_var
        return mu, var

    def log_prob(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability with heteroscedastic variance.

        Args:
            x: Observed semantic vector [batch, d_latent]
            z: Latent state [batch, d_latent]
            v: Variance state [batch, d_latent]

        Returns:
            log_prob: Log probability [batch]
        """
        mu, var = self.forward(z, v)

        log_prob_per_dim = (
            -0.5 * math.log(2 * math.pi)
            - 0.5 * torch.log(var)
            - 0.5 * (x - mu) ** 2 / var
        )

        return log_prob_per_dim.sum(dim=-1)
