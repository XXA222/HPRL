from types import SimpleNamespace

from freqtrade.hedge.hprl.closure_acceptance import evaluate_hprl_closure


def test_closure_requires_all_final_pass_evidence():
    report = evaluate_hprl_closure(
        SimpleNamespace(verdict="PASS"),
        SimpleNamespace(host_fit=True, cuda_fit=True),
        SimpleNamespace(verdict="PASS"),
    )
    assert report.verdict == "PASS"
    assert report.final_release_ready is True
    assert report.reasons == ()


def test_closure_never_upgrades_provisional_or_inconclusive_evidence():
    report = evaluate_hprl_closure(
        SimpleNamespace(verdict="INCONCLUSIVE"),
        SimpleNamespace(host_fit=True, cuda_fit=True),
        SimpleNamespace(verdict="PROVISIONAL"),
    )
    assert report.verdict == "BLOCKED"
    assert report.final_release_ready is False
    assert "risk_learning_inconclusive" in report.reasons
    assert "two_year_runtime_provisional" in report.reasons
