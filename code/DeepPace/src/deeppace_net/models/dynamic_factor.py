"""`dynamic_factor` — a shared/local state-space encoder walked in booking-date order.

Design: docs/design.md §12 (2026-09-18 entry).

Adapted from Deep Factors (Wang et al.) global/local decomposition: one small **shared** GRU
state, updated once per booking day from a portfolio-wide summary of that day's activity, and
one **local** GRU state per (property, arrival_date) *slot*, updated only on the days that slot
is actually touched. The split matches two separate, already-validated mechanisms in the
generator (module docstring in `booking_stream.py`): the shared state's job is the diagonal
book_dow/promo ripple; the local state's job is the persistent per-stay-date `pace_shock` field.

Prediction, per slot, follows the pickup-method shape the whole design converged on:

    unconstrained_hat = net_so_far + mlp(local_state, shared_state, slot_context)

`net_so_far` (cumulative gross_inc - cancel_inc for that slot) is exact bookkeeping, not
learned — the network only has to learn the *correction*, not reinvent addition. Trained
against every touch event, not just a final snapshot, so accuracy-vs-DTA falls out of
evaluation for free (comparable to `validate.pace_information`'s X6 correlation curve).

**Everything is computed in capacity-normalized units** (rooms / that slot's capacity), not raw
room counts. Properties in `_data/v3` range from capacity 87 to 646 — training on raw counts
means the optimizer has to relearn each property's scale from scratch, which swamps the
signal this architecture is actually supposed to learn (an early, real finding: loss was
essentially flat over 5 epochs on raw counts). Rescaling to room counts is a final
multiplication by capacity, done by the caller (`train.py`), not inside the loss.

Capacity is otherwise absent from the model's mechanics — this variant targets
`unconstrained_demand`; a `min(prediction, capacity)` cap, if wanted, is a separate
post-processing step (docs/design.md §12), never a term in this model's loss or forward pass.
Normalizing *inputs* by capacity is a scale fix, not a cap.

Public surface:
    NAME, VERSION
    DynamicFactorConfig (dataclass)
    DynamicFactorNet (nn.Module) — .forward(stream, chunk_days) -> normalized preds/targets/slot_idx
    build(config) -> ModelSpec
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from deeppace_net.booking_stream import BookingStream

NAME = "dynamic_factor"
VERSION = 2


@dataclass(frozen=True)
class DynamicFactorConfig:
    n_slot_features: int
    hidden_dim: int = 32
    day_input_dim: int = 3  # booking_dow_sin, booking_dow_cos, mean gross_inc/capacity that day
    correction_scale: float = 1.5  # bounds the readout's log-space correction to [-scale, scale]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    version: int
    model: nn.Module


class DynamicFactorNet(nn.Module):
    def __init__(self, config: DynamicFactorConfig) -> None:
        super().__init__()
        self.config = config
        h = config.hidden_dim

        self.shared_gru = nn.GRUCell(config.day_input_dim, h)
        # local input: gross_inc, cancel_inc, dta (log1p), booking_dow_sin, booking_dow_cos — all
        # but dta are capacity-normalized before reaching here
        self.local_gru = nn.GRUCell(5, h)
        self.readout = nn.Sequential(
            nn.Linear(2 * h + config.n_slot_features, h), nn.ReLU(), nn.Linear(h, 1),
        )
        # zero-init the last layer: correction(0) == 0 exactly, so training starts at the
        # bookkeeping-only baseline and can only learn a deviation from there, rather than from
        # an arbitrary random correction that a saturating tanh (below) then traps in place.
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)
        self.shared_init = nn.Parameter(torch.zeros(h))
        # bounds the correction to at most a `correction_scale`x multiplicative move on the
        # bookkeeping term, in log space. Unbounded, a first pass on v3 learned a correction that
        # overwhelmed `net_so_far` and drove predictions negative even at a 7-day horizon, where
        # bookkeeping alone is already accurate — the correction must be a *correction*, not a
        # term that can dominate a signal already known to be reliable.
        self.correction_scale = config.correction_scale

    def forward(
        self, stream: BookingStream, chunk_days: int = 30, device: str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
        """Walk `stream.days` in order, predicting at every touch event.

        Returns `(preds_log, targets_log, slot_idx)`, all aligned, one entry per (day, slot)
        touch — *not* one per slot — so a slot touched on 40 different booking days contributes
        40 supervised points, one per partial-curve state. Both are `log1p` of the
        capacity-normalized quantity; invert with `expm1(.) * capacity` to get room counts.
        `slot_idx` lets the caller do that rescaling and look up other per-slot metadata.
        State is detached every `chunk_days` (truncated BPTT): this keeps memory bounded on the
        full portfolio without limiting how many days the recurrence can span in principle.
        """
        h = self.config.hidden_dim
        n_slots = len(stream.slots)
        slot_context = torch.as_tensor(stream.slot_context, dtype=torch.float32, device=device)
        capacity = torch.as_tensor(stream.capacity, dtype=torch.float32, device=device)
        target_norm_all = (
            torch.as_tensor(stream.target, dtype=torch.float32, device=device) / capacity
        )

        local_state = torch.zeros(n_slots, h, device=device)
        shared_state = self.shared_init.unsqueeze(0).to(device)
        net_so_far = torch.zeros(n_slots, device=device)  # normalized (fraction of capacity)

        preds: list[torch.Tensor] = []
        touched_targets: list[torch.Tensor] = []
        touched_idx: list[np.ndarray] = []

        for i, day in enumerate(stream.days):
            idx = torch.as_tensor(day.slot_idx, dtype=torch.long, device=device)
            cap = capacity[idx]
            gross = torch.as_tensor(day.gross_inc, dtype=torch.float32, device=device) / cap
            cancel = torch.as_tensor(day.cancel_inc, dtype=torch.float32, device=device) / cap
            dta = torch.as_tensor(day.dta, dtype=torch.float32, device=device)

            day_input = torch.tensor(
                [[day.booking_dow_sin, day.booking_dow_cos, float(gross.mean())]],
                dtype=torch.float32, device=device,
            )
            shared_state = self.shared_gru(day_input, shared_state)

            bdow = torch.tensor(
                [day.booking_dow_sin, day.booking_dow_cos], dtype=torch.float32, device=device,
            ).expand(len(idx), 2)
            local_input = torch.stack(
                [gross, cancel, torch.log1p(dta), bdow[:, 0], bdow[:, 1]], dim=1,
            )
            updated = self.local_gru(local_input, local_state[idx])
            local_state = local_state.index_copy(0, idx, updated)

            net_so_far = net_so_far.index_copy(0, idx, net_so_far[idx] + gross - cancel)

            shared_rep = shared_state.expand(len(idx), h)
            readout_input = torch.cat([updated, shared_rep, slot_context[idx]], dim=1)
            correction = self.correction_scale * torch.tanh(self.readout(readout_input).squeeze(-1))
            # log1p space, not linear: unconstrained_demand/capacity is heavy-tailed (measured
            # 99.9th pct ~6x capacity, max ~24x on v3 — a handful of large-shock stay dates would
            # otherwise dominate a linear MSE loss and destabilize training). Every factor in the
            # generator itself is log-additive (demand.py, pace.py), so this is the natural space,
            # not a bolted-on transform.
            preds.append(torch.log1p(net_so_far[idx]) + correction)
            touched_targets.append(torch.log1p(target_norm_all[idx]))
            touched_idx.append(day.slot_idx)

            if (i + 1) % chunk_days == 0:
                local_state = local_state.detach()
                shared_state = shared_state.detach()
                net_so_far = net_so_far.detach()

        return torch.cat(preds), torch.cat(touched_targets), np.concatenate(touched_idx)


def build(config: DynamicFactorConfig) -> ModelSpec:
    return ModelSpec(name=NAME, version=VERSION, model=DynamicFactorNet(config))
