from __future__ import annotations

import pytest
import torch

from freqtrade.hedge.hprl.ensemble import GaussianStateBoundary, RiskAwareEnsembleRouter
from freqtrade.hedge.hprl.regime import AdaptiveRegimeDetector, PolicyLibrary, RegimeSignature


class ConstantPolicy:
    def __init__(self, value: float, action_dim: int = 2) -> None:
        self.value = value
        self.action_dim = action_dim

    def act(self, obs, *, deterministic: bool = True):
        return torch.full((obs.shape[0], self.action_dim), self.value, dtype=obs.dtype)


def test_regime_detector_labels_trend() -> None:
    detector = AdaptiveRegimeDetector(trend_threshold=0.2, high_vol_threshold=10.0)
    assert detector.label(torch.tensor([0.01, 0.02, 0.01, 0.02])) == "trend_up"
    assert detector.label(torch.tensor([-0.01, -0.02, -0.01, -0.02])) == "trend_down"


def test_policy_library_weights_sum_to_one() -> None:
    library = PolicyLibrary()
    library.register("up", ConstantPolicy(0.8), RegimeSignature(1.0, 0.5, 0.1))
    library.register("down", ConstantPolicy(0.2), RegimeSignature(-1.0, 0.5, 0.8))
    weights = library.weights(RegimeSignature(0.8, 0.5, 0.2))
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["up"] > weights["down"]


def test_policy_composition_stays_in_bounds() -> None:
    library = PolicyLibrary()
    library.register("a", ConstantPolicy(0.9), RegimeSignature(1.0, 1.0, 0.0))
    library.register("b", ConstantPolicy(0.1), RegimeSignature(-1.0, 1.0, 1.0))
    action = library.compose_action(torch.zeros((4, 3)), RegimeSignature(0.0, 1.0, 0.5))
    assert action.shape == (4, 2)
    assert torch.all((0 <= action) & (action <= 1))


def test_ood_boundary_separates_extreme_state() -> None:
    torch.manual_seed(1)
    states = torch.randn((500, 4)) * 0.2
    boundary = GaussianStateBoundary(quantile=0.99).fit(states)
    score = boundary.score(torch.tensor([[0.0, 0.0, 0.0, 0.0], [10.0, 10.0, 10.0, 10.0]]))
    assert bool(score.in_distribution[0])
    assert not bool(score.in_distribution[1])


def test_ensemble_routes_ood_to_conservative_policy() -> None:
    torch.manual_seed(2)
    boundary = GaussianStateBoundary(quantile=0.99).fit(torch.randn((300, 3)) * 0.1)
    router = RiskAwareEnsembleRouter(ConstantPolicy(0.0))
    router.register("special", ConstantPolicy(0.8), boundary, profitability=1.0)
    obs = torch.tensor([[0.0, 0.0, 0.0], [9.0, 9.0, 9.0]])
    action = router.act(obs)
    assert torch.allclose(action[0], torch.tensor([0.8, 0.8]))
    assert torch.allclose(action[1], torch.tensor([0.0, 0.0]))
