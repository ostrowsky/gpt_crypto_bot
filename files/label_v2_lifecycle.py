from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from v2.history_store import LocalHistoryStore
from v2.lifecycle_labeling import LifecycleThresholds, label_bars, summarize_labels, thresholds_dict


ROOT = Path(__file__).resolve().parent
DEFAULT_STORE = ROOT / ".runtime" / "v2_history"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_AUDIT = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_label_audit_15m.json"


def build(store_root: Path, output: Path, audit_path: Path, timeframe: str = "15m") -> dict:
    store = LocalHistoryStore(store_root)
    labels = []
    for symbol, tf in store.keys():
        if tf != timeframe:
            continue
        slice_ = store.load(symbol, tf)
        if not slice_.is_contiguous:
            continue
        labels.extend(label_bars(slice_.bars))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(asdict(label), ensure_ascii=False, default=str) for label in labels)
        + ("\n" if labels else ""),
        encoding="utf-8",
    )
    audit = {
        "timeframe": timeframe,
        "thresholds": thresholds_dict(LifecycleThresholds()),
        "summary": summarize_labels(labels),
        "files": {"labels": str(output), "audit": str(audit_path)},
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.store_root, args.output, args.audit, args.timeframe)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
