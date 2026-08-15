"""Dry-run readiness certificate and reproducible support bundle."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from datetime import datetime
import json
from pathlib import Path
import zipfile
from .common import atomic_write_text,canonical_json,ensure_aware,sha256_value
@dataclass(frozen=True,slots=True)
class ReadinessCheck:name:str;passed:bool;reason:str=""
@dataclass(frozen=True,slots=True)
class DryRunReadinessCertificate:
    certificate_id:str;created_at:datetime;session_id:str;ready:bool;checks:tuple[ReadinessCheck,...];source_version: str = "clean-mainline";mainnet_writes_locked:bool=True
class DryRunReadinessBuilder:
    def build(self,*,session_id:str,checks:tuple[ReadinessCheck,...],at:datetime)->DryRunReadinessCertificate:
        ts=ensure_aware(at);ready=bool(checks) and all(x.passed for x in checks);body={"session_id":session_id,"checks":checks,"created_at":ts,"ready":ready};return DryRunReadinessCertificate(sha256_value(body)[:24],ts,session_id,ready,checks)
class SupportBundleBuilder:
    def create(self,*,destination:str|Path,certificate:DryRunReadinessCertificate,files:tuple[str|Path,...]=())->Path:
        target=Path(destination);target.parent.mkdir(parents=True,exist_ok=True);manifest=[]
        with zipfile.ZipFile(target,"w",zipfile.ZIP_DEFLATED) as z:
            cert_text=canonical_json(asdict(certificate));z.writestr("readiness-certificate.json",cert_text);manifest.append({"path":"readiness-certificate.json","sha256":sha256_value(json.loads(cert_text))})
            for index,item in enumerate(files,1):
                path=Path(item)
                if not path.exists() or not path.is_file() or path.is_symlink():continue
                safe_name=path.name.replace("/","_").replace("\\","_")
                arc=f"evidence/{index:03d}-{safe_name}";z.write(path,arc);manifest.append({"path":arc,"size":path.stat().st_size})
            z.writestr("manifest.json",canonical_json({"files":manifest,"mainnet_writes_locked":True}))
        return target
