from datetime import UTC,datetime
from decimal import Decimal
from freqtrade.hedge.operations.risk import PortfolioRiskMonitor,RiskLeg

def test_portfolio_risk_evaluates_gross_net_and_margin():
    m=PortfolioRiskMonitor(max_gross_ratio=Decimal("0.8"),max_margin_ratio=Decimal("0.5"),max_net_ratio=Decimal("0.4"));s=m.snapshot(timestamp=datetime(2026,8,5,tzinfo=UTC),equity=Decimal("1000"),legs=(RiskLeg("BTC",Decimal("700"),Decimal("100"),Decimal("600")),));assert not s.ready and set(s.reasons)=={"MARGIN_LIMIT","NET_LIMIT"}
