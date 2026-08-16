#!/usr/bin/env python3
"""Operational CLI for the Production Readiness R1 evidence spine."""
from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))

from freqtrade.hedge.production.contracts import EvidenceKind, EvidenceStatus, ProductionStage, cumulative_requirements
from freqtrade.hedge.production.evidence import EvidenceLedger, EvidenceLedgerStore
from freqtrade.hedge.production.policy import StageEvaluator


def parse_time(value: str | None) -> datetime:
    if not value: return datetime.now(UTC)
    parsed=datetime.fromisoformat(value.replace('Z','+00:00'))
    if parsed.tzinfo is None: raise ValueError('timestamp must include timezone')
    return parsed.astimezone(UTC)


def load_or_new(path: Path) -> EvidenceLedger:
    return EvidenceLedger.load(path) if path.exists() else EvidenceLedger()


def cmd_init(args) -> int:
    if args.ledger.exists() and not args.force: raise FileExistsError(f"ledger already exists: {args.ledger}")
    EvidenceLedger().save_atomic(args.ledger); print(json.dumps({"status":"PASS","ledger":str(args.ledger),"digest":EvidenceLedger().digest()},sort_keys=True)); return 0


def cmd_record(args) -> int:
    store=EvidenceLedgerStore(args.ledger); ledger, expected_digest=store.load(); artifact=Path(args.artifact)
    raw=artifact.read_bytes(); metadata=json.loads(args.metadata) if args.metadata else {}
    if not isinstance(metadata,dict): raise ValueError('metadata must be a JSON object')
    rec=store.append_record(kind=EvidenceKind(args.kind),status=EvidenceStatus(args.status),observed_at=parse_time(args.observed_at),ttl=timedelta(hours=args.ttl_hours),artifact_sha256=sha256(raw).hexdigest(),producer=args.producer,metadata=metadata,expected_digest=expected_digest)
    updated,_=store.load(); print(json.dumps({"status":"PASS","record_sha256":rec.record_sha256,"ledger_digest":updated.digest(),"kind":rec.kind.value,"evidence_status":rec.status.value},sort_keys=True)); return 0


def cmd_verify(args) -> int:
    ledger=EvidenceLedger.load(args.ledger); ok=ledger.verify_chain(); print(json.dumps({"status":"PASS" if ok else "FAIL","records":len(ledger.records),"digest":ledger.digest()},sort_keys=True)); return 0 if ok else 1


def cmd_status(args) -> int:
    ledger=EvidenceLedger.load(args.ledger); stage=ProductionStage(args.stage); result=StageEvaluator(ledger).evaluate(stage,now=parse_time(args.now))
    payload={"status":"PASS" if result.passed else "FAIL","stage":stage.value,"passed":result.passed,"missing":[x.value for x in result.missing],"failed":[x.value for x in result.failed],"stale":[x.value for x in result.stale],"reasons":list(result.reasons),"evidence_digest":result.evidence_digest}
    print(json.dumps(payload,sort_keys=True)); return 0 if result.passed else 2


def cmd_requirements(args) -> int:
    stage=ProductionStage(args.stage); print(json.dumps({"stage":stage.value,"requirements":sorted(x.value for x in cumulative_requirements(stage))},sort_keys=True)); return 0


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('init'); q.add_argument('--ledger',type=Path,required=True); q.add_argument('--force',action='store_true'); q.set_defaults(fn=cmd_init)
    q=sub.add_parser('record'); q.add_argument('--ledger',type=Path,required=True); q.add_argument('--kind',choices=[x.value for x in EvidenceKind],required=True); q.add_argument('--status',choices=[x.value for x in EvidenceStatus],required=True); q.add_argument('--artifact',required=True); q.add_argument('--producer',required=True); q.add_argument('--ttl-hours',type=float,required=True); q.add_argument('--observed-at'); q.add_argument('--metadata'); q.set_defaults(fn=cmd_record)
    q=sub.add_parser('verify'); q.add_argument('--ledger',type=Path,required=True); q.set_defaults(fn=cmd_verify)
    q=sub.add_parser('status'); q.add_argument('--ledger',type=Path,required=True); q.add_argument('--stage',choices=[x.value for x in ProductionStage],required=True); q.add_argument('--now'); q.set_defaults(fn=cmd_status)
    q=sub.add_parser('requirements'); q.add_argument('--stage',choices=[x.value for x in ProductionStage],required=True); q.set_defaults(fn=cmd_requirements)
    return p


def main()->int:
    args=parser().parse_args(); return args.fn(args)

if __name__=='__main__': raise SystemExit(main())
