from freqtrade.hedge.operations.warmup import StrategyWarmupGate,WarmupRequirement

def test_warmup_requires_base_and_informative_frames():
    gate=StrategyWarmupGate(WarmupRequirement(100,(("5m",20),("1h",5))));d=gate.evaluate(base_available=80,informative_available={"5m":20,"1h":4});assert not d.ready and d.progress==0.8 and d.missing==("BASE:20","1h:1")
    assert gate.evaluate(base_available=100,informative_available={"5m":20,"1h":5}).ready
