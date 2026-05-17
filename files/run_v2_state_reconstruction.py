from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from v2.history_store import LocalHistoryStore
from v2.lifecycle_labeling import LifecycleLabel
from v2.state import SymbolState
from v2.state_reconstruction import (
    build_rows,
    chronological_split,
    evaluate,
    fit_centroids,
    fit_majority,
    fit_scaler,
    predict_centroid,
    predict_shadow_rule,
    scale_features,
)
from v2.teacher_confidence import TeacherConfidence


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_state_reconstruction_baseline_15m.json"


def _load_labels(path: Path):
    labels = {}
    by_day = defaultdict(int)
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        payload["state"] = SymbolState(payload["state"])
        label = LifecycleLabel(**payload)
        labels[(label.symbol, label.open_ts_ms)] = label
        by_day[(label.symbol, label.local_day)] += 1
    return labels, by_day


def _build_confidence(labels, by_day):
    from v2.teacher_confidence import score_label

    return {
        key: score_label(label, bars_in_day=by_day[(label.symbol, label.local_day)])
        for key, label in labels.items()
    }


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
    majority = fit_majority(train)
    means, stds = fit_scaler(train)
    scaled_train = [
        row.__class__(
            row.symbol,
            row.local_day,
            row.ts_ms,
            scale_features(row.features, means, stds),
            row.label,
            row.confidence,
        )
        for row in train
    ]
    scaled_test = [scale_features(row.features, means, stds) for row in test]
    centroids = fit_centroids(scaled_train)
    result = {
        "rows": len(rows),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_days": len({row.local_day for row in train}),
        "test_days": len({row.local_day for row in test}),
        "models": {
            "majority_class": evaluate(test, [majority for _ in test]),
            "nearest_centroid": evaluate(test, [predict_centroid(features, centroids) for features in scaled_test]),
            "provisional_shadow_rules": evaluate(test, [predict_shadow_rule(row) for row in test]),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


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
