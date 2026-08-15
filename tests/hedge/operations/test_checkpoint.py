from datetime import UTC,datetime,timedelta
from pathlib import Path
import pytest
from freqtrade.hedge.operations.checkpoint import RuntimeCheckpointManager

def test_checkpoint_checksum_retention_and_restore(tmp_path:Path):
    m=RuntimeCheckpointManager(tmp_path,retention=2);t=datetime(2026,8,5,tzinfo=UTC);ids=[]
    for i in range(3):ids.append(m.create({"n":i},at=t+timedelta(seconds=i)).checkpoint_id)
    assert len(m.list_ids())==2 and m.restore(ids[-1]).payload=={"n":2};path=tmp_path/f"{ids[-1]}.json";path.write_text(path.read_text().replace('"n":2','"n":9'),encoding="utf-8")
    with pytest.raises(ValueError):m.restore(ids[-1])
