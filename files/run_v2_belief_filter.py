from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_v2_state_reconstruction import _build_confidence, _load_labels
from v2.belief_filter import filter_rows
from v2.history_store import LocalHistoryStore
from v2.state_reconstruction import (
    build_rows,
    chronological_split,
    evaluate,
    fit_centroids,
    fit_scaler,
    predict_centroid,
    scale_features,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_belief_filter_v1_15m.json"


def build(history_root: Path, labels_path: Path, output: Path) -> dict:
    labels, day_sizes = _load_labels(labels_path)
    confidence = _build_confidence(labels, day_sizes)
    store = LocalHistoryStore(history_root)
    rows = []
    for symbol, tf in store.keys():
        if tf != "15m":
            continue
        slice_ = store.load(symbol, tf)
        if not slice_.is_contiguous:
            continue
        labels_by_ts = {ts: label for (sym, ts), label in labels.items() if sym == symbol}
        conf_by_ts = {ts: conf for (sym, ts), conf in confidence.items() if sym == symbol}
        rows.extend(build_rows(slice_.bars, labels_by_ts, conf_by_ts))
    train, test = chronological_split(rows)
    means, stds = fit_scaler(train)
    scaled_train = [_scaled(row, means, stds) for row in train]
    scaled_test = [_scaled(row, means, stds) for row in test]
    centroids = fit_centroids(scaled_train)
    isolated_preds = [predict_centroid(row.features, centroids) for row in scaled_test]
    filtered = filter_rows(scaled_test, centroids, self_bias=0.70, temperature=1.0)
    payload = {
        "rows": len(rows),
        "train_rows": len(train),
        "test_rows": len(test),
        "parameters": {"self_bias": 0.70, "temperature": 1.0},
        "models": {
            "isolated_nearest_centroid": evaluate(scaled_test, isolated_preds),
            "belief_filter_v1": evaluate([item.row for item in filtered], [item.prediction for item in filtered]),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _scaled(row, means, stds):
    return row.__class__(
        row.symbol,
        row.local_day,
        row.ts_ms,
        scale_features(row.features, means, stds),
        row.label,
        row.confidence,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.history_root, args.labels, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
