from datetime import UTC,datetime
from pathlib import Path
import zipfile
from freqtrade.hedge.operations.readiness import DryRunReadinessBuilder,ReadinessCheck,SupportBundleBuilder

def test_readiness_certificate_and_support_bundle(tmp_path:Path):
    c=DryRunReadinessBuilder().build(session_id="s",checks=(ReadinessCheck("a",True),ReadinessCheck("b",True)),at=datetime(2026,8,5,tzinfo=UTC));assert c.ready and c.mainnet_writes_locked;f=tmp_path/"x.txt";f.write_text("ok");z=SupportBundleBuilder().create(destination=tmp_path/"bundle.zip",certificate=c,files=(f,));assert zipfile.is_zipfile(z) and {"readiness-certificate.json","manifest.json","evidence/001-x.txt"}.issubset(zipfile.ZipFile(z).namelist())
