"""Frozen foundation forecast plus an additive, observed-pace-conditioned adjustment."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from deeppace_net.pace_data import PaceBatch

NAME = "two_stage"
VERSION = 1


@dataclass(frozen=True)
class TwoStageConfig:
    hidden_dim: int = 16
    context_dim: int = 6


class TwoStagePaceNet(nn.Module):
    def __init__(self, config: TwoStageConfig | None = None):
        super().__init__()
        config = config or TwoStageConfig()
        self.config = config
        self.local_gru = nn.GRU(3, config.hidden_dim, batch_first=True)
        self.shared_gru = nn.GRU(4, config.hidden_dim, batch_first=True)
        self.readout = nn.Sequential(
            nn.Linear(2 * config.hidden_dim + config.context_dim + 3, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
        )
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    def _encode(self, batch: PaceBatch, device) -> tuple[torch.Tensor, torch.Tensor]:
        """`(local_state, shared_state)` — one row per `batch.rows`, `shared_state` broadcast.

        Split out of `forward` so a diagnostic can read the local state (what the model
        actually conditions its correction on) without duplicating the encoding logic — see
        `deeppace_eval.latents`'s decisive test (design.md §7.4): correlating this state
        against the true `w_group/leisure/lastmin` is only meaningful if it's the exact state
        the readout consumes, not a re-derived approximation of it.
        """
        x = torch.as_tensor(batch.curves, device=device)
        lengths = torch.as_tensor(batch.observed.sum(axis=1), dtype=torch.long)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        _, local = self.local_gru(packed)
        shared = torch.as_tensor(batch.shared, device=device).unsqueeze(0)
        _, state = self.shared_gru(shared)
        return local[0], state[0].expand(len(batch.rows), -1)

    def encode(self, batch: PaceBatch) -> torch.Tensor:
        """The local GRU's final hidden state, one row per `batch.rows` — for probing only,
        never for training (see `_encode`)."""
        device = next(self.parameters()).device
        with torch.no_grad():
            local_state, _ = self._encode(batch, device)
        return local_state

    def forward(self, batch: PaceBatch, base: torch.Tensor) -> torch.Tensor:
        """Return final-room predictions in rooms; labels are never read here.

        Adjustment is signed and capacity-normalized. No log transform, OTB double
        counting, or hard clamp in the training path. Inference bounds are applied separately.
        """
        device = next(self.parameters()).device
        local_state, shared_state = self._encode(batch, device)
        capacity = torch.tensor(batch.rows.capacity.to_numpy(), dtype=torch.float32, device=device)
        otb = torch.tensor(batch.rows.current_otb.to_numpy(), dtype=torch.float32, device=device)
        base = base.to(device).detach()
        context = torch.as_tensor(batch.context, device=device)
        inputs = torch.cat(
            [
                local_state,
                shared_state,
                context,
                torch.stack([base / capacity, otb / capacity, (base - otb) / capacity], dim=1),
            ],
            dim=1,
        )
        return base + capacity * self.readout(inputs).squeeze(-1)


def retention_penalty(
    prediction: torch.Tensor, otb: torch.Tensor, capacity: torch.Tensor, min_retention: float,
) -> torch.Tensor:
    """Soft, capacity-normalized penalty for predictions far below current OTB.

    Cancellations legitimately let `prediction < current_otb` (module docstring) — this is
    *not* a hard floor, and callers should add it to the training loss, never clip inference
    output with it. It exists because nothing else discourages the failure actually observed
    on v3: `base` (Chronos) has no access to `current_otb` at all, so it can forecast a final
    total below what's already booked, and the correction — trained on plain MSE — has no
    signal telling it that's implausible. `min_retention` only penalizes predictions assuming
    a same-day collapse beyond that fraction (default 0.5: lose more than half of current OTB
    before arrival); ordinary, gradual cancellation is well inside the unpenalized region.
    """
    floor = min_retention * otb
    return torch.relu(floor - prediction) / capacity


@dataclass(frozen=True)
class ModelSpec:
    name: str
    version: int
    model: nn.Module


def build(config: TwoStageConfig | None = None) -> ModelSpec:
    return ModelSpec(NAME, VERSION, TwoStagePaceNet(config))
