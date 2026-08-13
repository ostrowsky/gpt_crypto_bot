from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import ml_candidate_ranker


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "critic_dataset.jsonl"


def build(dataset: Path) -> dict[str, Any]:
    coverage = ml_candidate_ranker.training_provenance_coverage(dataset)
    first_ts = None
    last_ts = None
    malformed_rows = 0
    total_rows = 0
    for raw in dataset.open("r", encoding="utf-8", errors="ignore") if dataset.exists() else ():
        if not raw.strip():
            continue
        total_rows += 1
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            malformed_rows += 1
            continue
        if not isinstance(row, dict):
            malformed_rows += 1
            continue
        ts = str(row.get("ts_signal") or "")
        if ts:
            first_ts = ts if first_ts is None else min(first_ts, ts)
            last_ts = ts if last_ts is None else max(last_ts, ts)
    return {
        "status": "ready" if int(coverage.get("verified_rows") or 0) >= 500 else "accumulating_verified_cohort",
        "decision": "eligible_for_training" if int(coverage.get("verified_rows") or 0) >= 500 else "do_not_train_or_promote",
        "dataset": str(dataset),
        "scope": {
            "maximum_available": True,
            "first_signal_time": first_ts,
            "last_signal_time": last_ts,
            "total_rows": total_rows,
            "malformed_rows": malformed_rows,
        },
        "provenance": coverage,
        "claims": {
            "trading_policy_changed": False,
            "causal_pnl_or_capture_uplift": None,
            "reason": "measurement-governance audit; no trading hypothesis was relaxed",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit immutable policy/label provenance over the full critic dataset")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = build(args.dataset)
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
