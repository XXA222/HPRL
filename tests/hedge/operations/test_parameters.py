from datetime import UTC,datetime
from decimal import Decimal
import pytest
from freqtrade.hedge.operations.parameters import ParameterBounds,StrategyParameterRegistry

def test_parameter_versions_activate_and_rollback():
    r=StrategyParameterRegistry({"risk":ParameterBounds(Decimal("0"),Decimal("1"))});t=datetime(2026,8,5,tzinfo=UTC);a=r.create({"risk":Decimal("0.5")},actor="a",at=t);b=r.create({"risk":Decimal("0.3")},actor="a",at=t);r.activate(a.version_id);r.activate(b.version_id);assert r.rollback()==a
    with pytest.raises(ValueError):r.create({"risk":Decimal("2")},actor="a",at=t)
