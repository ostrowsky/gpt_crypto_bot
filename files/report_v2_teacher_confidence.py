from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from v2.lifecycle_labeling import LifecycleLabel
from v2.state import SymbolState
from v2.teacher_confidence import score_label


ROOT = Path(__file__).resolve().parent
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_teacher_confidence_audit_15m.json"


def build(labels_path: Path, output: Path) -> dict:
    labels = []
    if labels_path.exists():
        for line in labels_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            payload["state"] = SymbolState(payload["state"])
            labels.append(LifecycleLabel(**payload))
    day_sizes = defaultdict(int)
    for label in labels:
        day_sizes[(label.symbol, label.local_day)] += 1
    buckets = defaultdict(list)
    rows = []
    for label in labels:
        conf = score_label(label, bars_in_day=day_sizes[(label.symbol, label.local_day)])
        buckets[label.state.value].append(conf.value)
        rows.append({**asdict(label), "state": label.state.value, **asdict(conf)})
    summary = {}
    for state, values in buckets.items():
        summary[state] = {
            "rows": len(values),
            "min": min(values),
            "max": max(values),
            "avg": round(sum(values) / len(values), 6),
        }
    payload = {
        "rows": len(rows),
        "state_confidence": summary,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.labels, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
