"""Tensorized vector dual-leg trading environment for CPU/CUDA HPRL training."""

from __future__ import annotations

from dataclasses import dataclass

from .action_space import TieredHedgeActionCodec
from .config import HPRLEnvironmentConfig, HPRLMemoryConfig
from .costs import ExecutionCostModel
from .data import TensorMarketDataset
from .device import require_torch, torch_device
from .memory import MarketDatasetAccessor
from .reward import CompositeReward, RewardFactsTensor
from .risk import HedgeActionProjector


@dataclass(frozen=True, slots=True)
class VectorStep:
    observation: object
    reward: object
    terminated: object
    truncated: object
    info: dict[str, object]


class VectorizedHedgeEnv:
    """Batched dual-leg simulator with a clean decision/realization boundary.

    In ``tiered`` mode the policy emits continuous [0, 1] latents but the executed action is an
    exact configurable LONG/SHORT margin-budget grid.  Five levels therefore produce 25 joint
    states per symbol while preserving the existing high-throughput continuous-control agents.
    """

    def __init__(
        self,
        dataset: TensorMarketDataset,
        config: HPRLEnvironmentConfig | None = None,
        *,
        device: str = "auto",
        memory_config: HPRLMemoryConfig | None = None,
    ) -> None:
        torch = require_torch()
        self.torch = torch
        self.config = config or HPRLEnvironmentConfig()
        self.memory_config = memory_config or HPRLMemoryConfig()
        self.device = torch_device(device)
        self.market = MarketDatasetAccessor(dataset, self.device, self.memory_config)
        self.dataset = self.market.dataset
        self.action_mode = self.config.action.mode.strip().lower()
        self.projector = HedgeActionProjector(
            self.config.action, validate_inputs=self.config.runtime_checks
        )
        self.tiered_codec = (
            TieredHedgeActionCodec(
                self.config.action,
                validate_inputs=self.config.runtime_checks,
            )
            if self.action_mode == "tiered"
            else None
        )
        self.cost_model = ExecutionCostModel(
            self.config.costs, validate_inputs=self.config.runtime_checks
        )
        self.reward_model = CompositeReward(
            self.config.reward, validate_inputs=self.config.runtime_checks
        )
        self.envs = self.config.parallel_envs
        self.symbols = self.market.symbols
        self.features = self.market.features
        self._step = 0
        self._equity = torch.empty(0, device=self.device)
        self._peak_equity = torch.empty(0, device=self.device)
        self._position = torch.empty(0, device=self.device)  # notional/equity ratio
        self._margin_position = torch.empty(0, device=self.device)
        self._level_index = torch.empty(0, dtype=torch.int64, device=self.device)
        self._previous_drawdown = torch.empty(0, device=self.device)
        self._tail_window = 128
        self._return_history = torch.empty(0, device=self.device)
        self._return_history_valid = torch.empty(0, dtype=torch.bool, device=self.device)
        self._return_history_cursor = 0
        self._closed = False
        self.reset()

    @property
    def observation_dim(self) -> int:
        return self.symbols * self.features + self.symbols * 2 + 4

    @property
    def action_dim(self) -> int:
        return self.symbols * 2

    @property
    def action_level_count(self) -> int:
        return self.config.action.level_count if self.action_mode == "tiered" else 0

    @property
    def joint_action_states_per_symbol(self) -> int | None:
        return (
            self.config.action.joint_states_per_symbol
            if self.action_mode == "tiered"
            else None
        )

    @property
    def position(self):
        return self._position.clone()

    @property
    def margin_position(self):
        return self._margin_position.clone()

    @property
    def equity(self):
        return self._equity.clone()

    @property
    def _return_history_count(self) -> int:
        """Compatibility diagnostic; not used on the training hot path."""
        if self._return_history_valid.numel() == 0:
            return 0
        return int(self._return_history_valid.sum(dim=0).max().item())

    def _assert_open(self) -> None:
        if self._closed:
            raise RuntimeError("HPRL environment has been closed")

    def _observation(self):
        self._assert_open()
        torch = self.torch
        idx = min(self._step, self.market.time_steps - 1)
        market = self.market.features_at(idx).reshape(1, -1).expand(self.envs, -1)
        # Tiered actions are margin-budget decisions, so the policy observes margin position.
        position_state = self._margin_position if self.action_mode == "tiered" else self._position
        position = position_state.reshape(self.envs, -1)
        gross_margin = self._margin_position.sum(dim=(-2, -1)).unsqueeze(-1)
        net_notional = (
            (self._position[..., 0] - self._position[..., 1]).sum(dim=-1).unsqueeze(-1)
        )
        equity_ratio = (self._equity / self.config.initial_equity).unsqueeze(-1)
        drawdown = (1.0 - self._equity / torch.clamp(self._peak_equity, min=1e-12)).unsqueeze(-1)
        return torch.cat(
            (market, position, gross_margin, net_notional, equity_ratio, drawdown), dim=-1
        )

    def reset(self, *, start_index: int = 0):
        self._assert_open()
        torch = self.torch
        if not 0 <= start_index < self.market.time_steps - 1:
            raise ValueError("start_index must leave at least one realizable transition")
        self._step = int(start_index)
        self._equity = torch.full(
            (self.envs,), self.config.initial_equity, dtype=torch.float32, device=self.device
        )
        self._peak_equity = self._equity.clone()
        shape = (self.envs, self.symbols, 2)
        self._position = torch.zeros(shape, dtype=torch.float32, device=self.device)
        self._margin_position = torch.zeros_like(self._position)
        self._level_index = torch.zeros(shape, dtype=torch.int64, device=self.device)
        self._previous_drawdown = torch.zeros_like(self._equity)
        self._return_history = torch.zeros(
            (self._tail_window, self.envs), dtype=torch.float32, device=self.device
        )
        self._return_history_valid = torch.zeros(
            (self._tail_window, self.envs), dtype=torch.bool, device=self.device
        )
        self._return_history_cursor = 0
        max_k = max(1, int(self._tail_window * float(self.config.cvar_alpha)))
        self._tail_sortable = torch.empty_like(self._return_history)
        self._tail_ranks = torch.arange(max_k, device=self.device).unsqueeze(-1)
        self._zero_env = torch.zeros(self.envs, device=self.device, dtype=torch.float32)
        return self._observation(), {"start_index": self._step}

    def sample_random_action(self):
        """Sample in the policy domain; tiered mode samples exact canonical tier codes."""
        torch = self.torch
        shape = (self.envs, self.symbols, 2)
        if self.action_mode == "tiered":
            index = torch.randint(
                0,
                self.config.action.level_count,
                shape,
                device=self.device,
            )
            return index.to(torch.float32).reshape(self.envs, -1) / float(
                self.config.action.level_count - 1
            )
        raw = torch.rand(shape, device=self.device, dtype=torch.float32)
        max_leg = float(self.config.action.max_leg_exposure)
        return (raw * max_leg).reshape(self.envs, -1)

    def _tail_loss(self, equity_return):
        torch = self.torch
        cursor = self._return_history_cursor
        self._return_history[cursor].copy_(equity_return.detach())
        self._return_history_valid[cursor].fill_(True)
        self._return_history_cursor = (cursor + 1) % self._tail_window

        # Every parallel environment owns an independent valid history.  This matters after
        # per-row autoreset: a newly reset account must not inherit another env's CVaR sample count.
        valid = self._return_history_valid
        counts = valid.sum(dim=0).clamp_min(1)
        k = torch.clamp(
            torch.floor(counts.to(torch.float32) * float(self.config.cvar_alpha)).to(torch.int64),
            min=1,
        )
        self._tail_sortable.copy_(self._return_history)
        self._tail_sortable.masked_fill_(~valid, float("inf"))
        max_k = self._tail_ranks.shape[0]
        worst = torch.topk(self._tail_sortable, k=max_k, dim=0, largest=False).values
        selected = self._tail_ranks < k.unsqueeze(0)
        selected_values = torch.where(selected, worst, torch.zeros_like(worst))
        tail_mean = selected_values.sum(dim=0) / k.to(worst.dtype)
        return torch.clamp(-tail_mean, min=0.0)

    def _decode_action(self, raw_action):
        torch = self.torch
        action = raw_action.to(self.device, dtype=torch.float32)
        if self.action_mode == "tiered":
            decoded = self.tiered_codec.decode(action, self._margin_position)
            return {
                "target_notional": decoded.target_notional,
                "target_margin": decoded.target_margin,
                "level_index": decoded.executed_level_index,
                "joint_action_index": decoded.joint_action_index,
                "executed_policy": decoded.executed_policy,
                "requested_policy": decoded.requested_policy,
                "requested_margin": decoded.requested_margin,
                "quantization_distance": decoded.quantization_distance,
                "constraint_distance": decoded.constraint_distance,
                "projected_mask": decoded.projected_mask,
                "transition_limited": decoded.transition_limited,
                "risk_limited": decoded.risk_limited,
            }
        projection = self.projector.project(action, self._position)
        zeros = torch.zeros(self.envs, device=self.device, dtype=torch.float32)
        return {
            "target_notional": projection.target,
            "target_margin": projection.target,
            "level_index": self._level_index,
            "joint_action_index": torch.full(
                self._level_index.shape[:-1], -1, device=self.device, dtype=torch.int64
            ),
            "executed_policy": projection.target,
            "requested_policy": action,
            "requested_margin": action,
            "quantization_distance": zeros,
            "constraint_distance": torch.abs(projection.target - action).mean(dim=(-2, -1)),
            "projected_mask": projection.projected_mask,
            "transition_limited": projection.projected_mask,
            "risk_limited": projection.projected_mask,
        }

    def step(self, raw_action) -> VectorStep:
        self._assert_open()
        torch = self.torch
        if raw_action.shape == (self.envs, self.action_dim):
            raw_action = raw_action.reshape(self.envs, self.symbols, 2)
        expected = (self.envs, self.symbols, 2)
        if tuple(raw_action.shape) != expected:
            raise ValueError(f"action shape must be {expected}, got {tuple(raw_action.shape)}")
        if self._step >= self.market.time_steps - 1:
            raise RuntimeError("episode is exhausted; call reset()")

        pre_equity = self._equity
        decoded = self._decode_action(raw_action)
        target = decoded["target_notional"]
        target_margin = decoded["target_margin"]
        delta = target - self._position
        turnover_by_symbol = delta.abs().sum(dim=-1)
        turnover_ratio = turnover_by_symbol.sum(dim=-1)
        turnover_notional = turnover_by_symbol * pre_equity.unsqueeze(-1)

        available_row = self.market.available_notional_at(self._step)
        available = None
        if available_row is not None:
            available = available_row.unsqueeze(0).expand(self.envs, -1)
        costs_by_symbol = self.cost_model.evaluate(
            turnover_notional=turnover_notional,
            equity=pre_equity.unsqueeze(-1),
            available_notional=available,
        )
        fees = costs_by_symbol.fees.sum(dim=-1)
        slippage = costs_by_symbol.slippage.sum(dim=-1)
        market_impact = costs_by_symbol.market_impact.sum(dim=-1)
        total_cost = fees + slippage + market_impact
        self._position = target
        self._margin_position = target_margin
        self._level_index = decoded["level_index"]

        returns = self.market.forward_returns_at(self._step)
        directional = self._position[..., 0] - self._position[..., 1]
        market_pnl_ratio = (directional * returns.unsqueeze(0)).sum(dim=-1)
        funding_pnl_ratio = self._zero_env
        funding_row = self.market.funding_rates_at(self._step)
        if funding_row is not None:
            funding = funding_row.unsqueeze(0)
            funding_side = -self._position[..., 0] + self._position[..., 1]
            funding_pnl_ratio = (funding_side * funding).sum(dim=-1)

        pnl = pre_equity * (market_pnl_ratio + funding_pnl_ratio) - total_cost
        self._equity = torch.clamp(pre_equity + pnl, min=0.0)
        self._peak_equity = torch.maximum(self._peak_equity, self._equity)
        drawdown = 1.0 - self._equity / torch.clamp(self._peak_equity, min=1e-12)
        drawdown_increase = torch.clamp(drawdown - self._previous_drawdown, min=0.0)
        self._previous_drawdown = drawdown
        equity_return = (self._equity - pre_equity) / torch.clamp(pre_equity, min=1e-12)
        tail_loss = self._tail_loss(equity_return)
        gross_margin_ratio = self._margin_position.sum(dim=(-2, -1))
        hedge_overlap_ratio = torch.minimum(
            self._margin_position[..., 0], self._margin_position[..., 1]
        ).sum(dim=-1)
        flatness = torch.clamp(1.0 - gross_margin_ratio / 0.05, 0.0, 1.0)
        opportunity_miss = (
            flatness * returns.abs().max().expand(self.envs)
            if self.config.reward.opportunity_cost != 0.0
            else self._zero_env
        )
        bankrupt = self._equity <= self.config.initial_equity * self.config.terminate_equity_ratio

        reward, components = self.reward_model.evaluate_tensor(
            RewardFactsTensor(
                equity_return=equity_return,
                drawdown_increase=drawdown_increase,
                downside_return=equity_return,
                cvar_loss=tail_loss,
                turnover_ratio=turnover_ratio,
                fee_ratio=(
                    fees / torch.clamp(pre_equity, min=1e-12)
                    if self.config.reward.fees != 0.0 else self._zero_env
                ),
                slippage_ratio=(
                    slippage / torch.clamp(pre_equity, min=1e-12)
                    if self.config.reward.slippage != 0.0 else self._zero_env
                ),
                impact_ratio=(
                    market_impact / torch.clamp(pre_equity, min=1e-12)
                    if self.config.reward.market_impact != 0.0 else self._zero_env
                ),
                funding_ratio=funding_pnl_ratio,
                quantization_distance=decoded["quantization_distance"],
                constraint_distance=decoded["constraint_distance"],
                gross_margin_ratio=gross_margin_ratio,
                hedge_overlap_ratio=hedge_overlap_ratio,
                opportunity_miss=opportunity_miss,
                terminal=bankrupt,
            ),
            return_components=self.config.info_mode == "full",
        )

        self._step += 1
        time_done = self._step >= self.market.time_steps - 1
        terminated = bankrupt
        truncated = torch.full_like(bankrupt, time_done, dtype=torch.bool)

        final_equity = self._equity
        final_position = self._position
        final_margin = self._margin_position
        final_level = self._level_index
        final_drawdown = drawdown
        final_executed_policy = decoded["executed_policy"].reshape(self.envs, self.action_dim)
        reset_float = bankrupt.to(dtype=self._equity.dtype)
        keep_float = 1.0 - reset_float
        self._equity = self._equity * keep_float + self.config.initial_equity * reset_float
        self._peak_equity = (
            self._peak_equity * keep_float + self.config.initial_equity * reset_float
        )
        position_keep = keep_float.reshape(self.envs, 1, 1)
        self._position = self._position * position_keep
        self._margin_position = self._margin_position * position_keep
        self._level_index = self._level_index * position_keep.to(torch.int64)
        self._previous_drawdown = self._previous_drawdown * keep_float
        self._return_history.mul_(keep_float.unsqueeze(0))
        self._return_history_valid &= (~bankrupt).unsqueeze(0)

        if self.config.info_mode == "training":
            info = {
                "equity": final_equity,
                "executed_action": final_executed_policy,
                "time_done": time_done,
            }
        else:
            info = {
                "equity": final_equity,
                "position": final_position,
                "margin_position": final_margin,
                "level_index": final_level,
                "joint_action_index": decoded["joint_action_index"],
                "drawdown": final_drawdown,
                "turnover_ratio": turnover_ratio,
                "fee_cost": fees,
                "slippage_cost": slippage,
                "market_impact_cost": market_impact,
                "executed_action": final_executed_policy,
                "target_notional": final_position,
                "target_margin": final_margin,
                "requested_policy_action": decoded["requested_policy"].reshape(
                    self.envs, self.action_dim
                ),
                "quantization_distance": decoded["quantization_distance"],
                "constraint_distance": decoded["constraint_distance"],
                "transition_limited": decoded["transition_limited"],
                "risk_limited": decoded["risk_limited"],
                "funding_pnl_ratio": funding_pnl_ratio,
                "projected": decoded["projected_mask"],
                "gross_margin_ratio": gross_margin_ratio,
                "hedge_overlap_ratio": hedge_overlap_ratio,
                "autoreset_mask": bankrupt,
                "time_done": time_done,
                "liquidation_distance_modeled": False,
                "reward_components": components,
                "step_index": self._step,
            }
        return VectorStep(self._observation(), reward, terminated, truncated, info)

    def close(self, *, aggressive: bool = False) -> None:
        """Release market history, vector state and CUDA staging buffers."""
        if self._closed:
            return
        self.market.release(aggressive=False, release_dataset=True)
        self.dataset = None
        self._equity = self.torch.empty(0, device=self.device)
        self._peak_equity = self.torch.empty(0, device=self.device)
        self._position = self.torch.empty(0, device=self.device)
        self._margin_position = self.torch.empty(0, device=self.device)
        self._level_index = self.torch.empty(0, dtype=self.torch.int64, device=self.device)
        self._previous_drawdown = self.torch.empty(0, device=self.device)
        self._return_history = self.torch.empty(0, device=self.device)
        self._return_history_valid = self.torch.empty(
            0, dtype=self.torch.bool, device=self.device
        )
        self._return_history_cursor = 0
        self._closed = True
        if aggressive:
            from .memory import phase_boundary_cleanup

            phase_boundary_cleanup(self.device, enabled=True)
