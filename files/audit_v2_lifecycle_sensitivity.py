from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from v2.history_store import LocalHistoryStore
from v2.lifecycle_labeling import LifecycleThresholds, label_bars, summarize_labels


ROOT = Path(__file__).resolve().parent
DEFAULT_STORE = ROOT / ".runtime" / "v2_history"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_sensitivity_audit_15m.json"


def build(store_root: Path, output: Path, timeframe: str = "15m") -> dict:
    store = LocalHistoryStore(store_root)
    slices = [
        store.load(symbol, tf)
        for symbol, tf in store.keys()
        if tf == timeframe and store.load(symbol, tf).is_contiguous
    ]
    baseline = LifecycleThresholds()
    variants = []
    for min_move in (3.0, 4.0, 5.0):
        for confirm in (1.5, 2.0, 2.5):
            for exhaustion in (0.30, 0.35, 0.40):
                thresholds = replace(
                    baseline,
                    min_favorable_move_pct=min_move,
                    confirmed_move_pct=confirm,
                    exhaustion_giveback_ratio=exhaustion,
                )
                labels = []
                for slice_ in slices:
                    labels.extend(label_bars(slice_.bars, thresholds=thresholds))
                summary = summarize_labels(labels)
                variants.append(
                    {
                        "thresholds": {
                            "min_favorable_move_pct": min_move,
                            "confirmed_move_pct": confirm,
                            "exhaustion_giveback_ratio": exhaustion,
                        },
                        "summary": summary,
                    }
                )
    base = next(
        item
        for item in variants
        if item["thresholds"]
        == {
            "min_favorable_move_pct": 4.0,
            "confirmed_move_pct": 2.0,
            "exhaustion_giveback_ratio": 0.35,
        }
    )
    base_days = int(base["summary"]["qualifying_days"])
    base_states = base["summary"]["state_counts"]
    for item in variants:
        summary = item["summary"]
        item["delta_vs_baseline"] = {
            "qualifying_days": int(summary["qualifying_days"]) - base_days,
            "state_counts": {
                state: int(summary["state_counts"].get(state, 0)) - int(base_states.get(state, 0))
                for state in sorted(set(summary["state_counts"]) | set(base_states))
            },
        }
    payload = {
        "timeframe": timeframe,
        "baseline": base["thresholds"],
        "variants": variants,
        "compact_summary": {
            "qualifying_days_min": min(int(item["summary"]["qualifying_days"]) for item in variants),
            "qualifying_days_max": max(int(item["summary"]["qualifying_days"]) for item in variants),
            "all_invalid_transition_counts_zero": all(
                not item["summary"]["invalid_transition_counts"] for item in variants
            ),
            "by_min_favorable_move_pct": {
                str(value): sorted(
                    {
                        int(item["summary"]["qualifying_days"])
                        for item in variants
                        if item["thresholds"]["min_favorable_move_pct"] == value
                    }
                )
                for value in (3.0, 4.0, 5.0)
            },
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-root", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.store_root, args.output, args.timeframe)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else {"variants": len(payload["variants"])})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
