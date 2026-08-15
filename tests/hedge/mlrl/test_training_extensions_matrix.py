from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from freqtrade.freqai.hedge_rl.actions import DEFAULT_ACTION_CATALOG, HedgeActions
from freqtrade.freqai.hedge_rl.config import HedgeRLConfig
from freqtrade.freqai.hedge_rl.training_extensions import (
    AuxiliaryRiskHead,
    CheckpointCompatibility,
    DistributionalValueHead,
    OptimizerConfig,
    RecurrentStateManager,
    clip_gradients,
    fail_closed_policy_decision,
    finite_multitask_loss,
    mask_action_logits,
    orthogonal_initialize,
)


def test_round91_action_logit_mask_blocks_invalid_actions():
    logits = torch.arange(len(DEFAULT_ACTION_CATALOG), dtype=torch.float32).unsqueeze(0)
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask[:, HedgeActions.HOLD] = True
    masked = mask_action_logits(logits, mask)
    assert masked.argmax(dim=-1).item() == HedgeActions.HOLD
    assert masked[0, HedgeActions.LONG_OPEN_SMALL] < -1e8


def test_round92_recurrent_state_manager_resets_selected_batch_rows():
    manager = RecurrentStateManager(layers=2, batch_size=3, hidden_size=4)
    manager.update(torch.ones_like(manager.state))
    manager.reset([False, True, False])
    assert torch.all(manager.state[:, 0, :] == 1)
    assert torch.all(manager.state[:, 1, :] == 0)
    assert torch.all(manager.state[:, 2, :] == 1)
    manager.reset()
    assert torch.count_nonzero(manager.state) == 0


def test_round93_orthogonal_initialization_zeros_linear_biases():
    model = nn.Sequential(nn.Linear(4, 4), nn.GELU(), nn.Linear(4, 2))
    orthogonal_initialize(model, gain=1.0)
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            assert torch.count_nonzero(layer.bias) == 0
            gram = layer.weight @ layer.weight.T
            assert torch.allclose(gram, torch.eye(layer.out_features), atol=1e-5)


def test_round94_distributional_value_head_outputs_probability_expectation():
    head = DistributionalValueHead(4, atoms=11, minimum=-5, maximum=5)
    logits, expectation = head(torch.zeros(3, 4))
    assert logits.shape == (3, 11) and expectation.shape == (3,)
    probabilities = torch.softmax(logits, dim=-1)
    assert torch.allclose(probabilities.sum(dim=-1), torch.ones(3))
    assert torch.all(expectation >= -5) and torch.all(expectation <= 5)


def test_round95_auxiliary_risk_head_outputs_bounded_named_risks():
    head = AuxiliaryRiskHead(8, hidden_dim=4)
    output = head(torch.randn(2, 8))
    assert tuple(output) == head.OUTPUTS
    assert all(value.shape == (2,) for value in output.values())
    assert all(torch.all((0 <= value) & (value <= 1)) for value in output.values())


def test_round96_finite_multitask_loss_checks_shape_values_and_weights():
    prediction = torch.zeros(4, 5, requires_grad=True)
    target = torch.ones(4, 5)
    loss = finite_multitask_loss(prediction, target, weights=torch.ones(5))
    loss.backward()
    assert torch.isfinite(loss) and prediction.grad is not None
    with pytest.raises(ValueError):
        finite_multitask_loss(torch.tensor([[float("nan")]]), torch.zeros(1, 1))


def test_round97_gradient_clipping_reports_and_caps_large_gradient():
    model = nn.Linear(2, 1)
    output = model(torch.full((8, 2), 1000.0)).sum()
    output.backward()
    report = clip_gradients(model.parameters(), maximum_norm=0.5)
    assert report.clipped and report.total_norm_before > report.maximum_norm
    post_norm = torch.sqrt(sum(parameter.grad.square().sum() for parameter in model.parameters()))
    assert post_norm <= 0.50001


def test_round98_optimizer_config_builds_adamw_with_exact_hyperparameters():
    model = nn.Linear(2, 1)
    config = OptimizerConfig(learning_rate=1e-3, weight_decay=0.01, epsilon=1e-7, betas=(0.8, 0.9))
    optimizer = config.build(model.parameters())
    group = optimizer.param_groups[0]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert group["lr"] == 1e-3 and group["weight_decay"] == 0.01
    assert group["eps"] == 1e-7 and group["betas"] == (0.8, 0.9)


def test_round99_checkpoint_compatibility_lists_exact_mismatches():
    contract = CheckpointCompatibility("clean-mainline-v2", "obs", "actions", "gru-ac")
    compatible, mismatches = contract.validate(
        {
            "source_version": "clean-mainline-v2",
            "observation_signature": "obs",
            "action_signature": "actions",
            "architecture": "gru-ac",
        }
    )
    assert compatible and mismatches == ()
    compatible, mismatches = contract.validate(
        {
            "source_version": "clean-mainline-v1",
            "observation_signature": "obs",
            "action_signature": "wrong",
            "architecture": "gru-ac",
        }
    )
    assert not compatible and mismatches == ("source_version", "action_signature")


def test_round100_fail_closed_policy_blocks_incompatible_or_stale_state():
    logits = np.zeros(len(DEFAULT_ACTION_CATALOG))
    logits[HedgeActions.LONG_OPEN_SMALL] = 10
    mask = np.ones(len(DEFAULT_ACTION_CATALOG), dtype=bool)
    blocked = fail_closed_policy_decision(
        logits,
        action_mask=mask,
        feature_age_steps=0,
        config=HedgeRLConfig(confidence_threshold=0),
        model_compatible=False,
        account_projection_fresh=True,
    )
    assert blocked.executed_action is HedgeActions.HOLD
    assert "MODEL_INCOMPATIBLE" in blocked.reasons
    allowed = fail_closed_policy_decision(
        logits,
        action_mask=mask,
        feature_age_steps=0,
        config=HedgeRLConfig(confidence_threshold=0),
        model_compatible=True,
        account_projection_fresh=True,
    )
    assert allowed.executed_action is HedgeActions.LONG_OPEN_SMALL
