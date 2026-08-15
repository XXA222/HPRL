"""PyTorch neural networks for Hedge observations, policy/value learning, and multitask ML."""

from __future__ import annotations

import torch
from torch import nn


class HedgeTemporalEncoder(nn.Module):
    """Encode a flattened causal market window plus account state."""

    def __init__(
        self,
        *,
        market_width: int,
        window_size: int,
        account_width: int = 12,
        hidden_dim: int = 128,
        recurrent_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if min(market_width, window_size, account_width, hidden_dim, recurrent_layers) < 1:
            raise ValueError("network dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be within [0, 1)")
        self.market_width = market_width
        self.window_size = window_size
        self.account_width = account_width
        self.flat_size = market_width * window_size + account_width
        self.market_projection = nn.Sequential(
            nn.LayerNorm(market_width),
            nn.Linear(market_width, hidden_dim),
            nn.GELU(),
        )
        self.temporal = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=recurrent_layers,
            batch_first=True,
            dropout=dropout if recurrent_layers > 1 else 0.0,
        )
        self.account_encoder = nn.Sequential(
            nn.LayerNorm(account_width),
            nn.Linear(account_width, hidden_dim // 2),
            nn.GELU(),
        )
        self.output = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
        )
        self.output_dim = hidden_dim

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim == 1:
            observation = observation.unsqueeze(0)
        if observation.ndim != 2 or observation.shape[-1] != self.flat_size:
            raise ValueError(
                "observation must have shape "
                f"(batch, {self.flat_size}), got {tuple(observation.shape)}"
            )
        market_end = self.market_width * self.window_size
        market = observation[:, :market_end].reshape(
            observation.shape[0], self.window_size, self.market_width
        )
        account = observation[:, market_end:]
        projected = self.market_projection(market)
        temporal, _ = self.temporal(projected)
        market_embedding = temporal[:, -1, :]
        account_embedding = self.account_encoder(account)
        return self.output(torch.cat([market_embedding, account_embedding], dim=-1))


class HedgeActorCriticNetwork(nn.Module):
    def __init__(self, encoder: HedgeTemporalEncoder, action_count: int = 21) -> None:
        super().__init__()
        if action_count < 2:
            raise ValueError("action_count must be at least 2")
        self.encoder = encoder
        self.policy_head = nn.Linear(encoder.output_dim, action_count)
        self.value_head = nn.Linear(encoder.output_dim, 1)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.encoder(observation)
        return self.policy_head(encoded), self.value_head(encoded).squeeze(-1)


class HedgeMultiTaskNetwork(nn.Module):
    """Sequence model predicting directional targets and account risk proxies."""

    OUTPUT_NAMES = (
        "long_score",
        "short_score",
        "target_net_ratio",
        "future_return",
        "future_volatility",
    )

    def __init__(self, encoder: HedgeTemporalEncoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Sequential(
            nn.Linear(encoder.output_dim, encoder.output_dim),
            nn.GELU(),
            nn.Linear(encoder.output_dim, len(self.OUTPUT_NAMES)),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(observation))

    def controls(self, observation: torch.Tensor) -> dict[str, torch.Tensor]:
        raw = self.forward(observation)
        return {
            "long_score": torch.sigmoid(raw[..., 0]),
            "short_score": torch.sigmoid(raw[..., 1]),
            "target_net_ratio": torch.tanh(raw[..., 2]),
            "future_return": raw[..., 3],
            "future_volatility": torch.nn.functional.softplus(raw[..., 4]),
        }


class HedgeMultiTaskMLP(nn.Module):
    """FreqAI-compatible 2D feature model for five Hedge supervision targets."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 5,
        hidden_dim: int = 256,
        n_layer: int = 2,
        dropout_percent: float = 0.1,
    ) -> None:
        super().__init__()
        if min(input_dim, output_dim, hidden_dim, n_layer) < 1:
            raise ValueError("MLP dimensions must be positive")
        blocks: list[nn.Module] = [
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        ]
        for _ in range(n_layer - 1):
            blocks.extend(
                [
                    nn.Dropout(dropout_percent),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                ]
            )
        blocks.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*blocks)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)
