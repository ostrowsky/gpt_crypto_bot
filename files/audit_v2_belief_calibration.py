from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_v2_belief_filter import _scaled
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
)


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_belief_calibration_audit_15m.json"


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
    isolated = evaluate(scaled_test, [predict_centroid(row.features, centroids) for row in scaled_test])
    variants = []
    for self_bias in (0.55, 0.65, 0.70, 0.75, 0.85):
        for temperature in (0.50, 0.75, 1.00, 1.25, 1.50):
            filtered = filter_rows(scaled_test, centroids, self_bias=self_bias, temperature=temperature)
            metrics = evaluate([item.row for item in filtered], [item.prediction for item in filtered])
            variants.append(
                {
                    "self_bias": self_bias,
                    "temperature": temperature,
                    "metrics": metrics,
                    "delta_vs_isolated": {
                        "macro_f1": round(metrics["macro_f1"] - isolated["macro_f1"], 6),
                        "weighted_accuracy": round(metrics["weighted_accuracy"] - isolated["weighted_accuracy"], 6),
                        "emerging_move_recall": round(
                            metrics["recall_by_state"]["emerging_move"]
                            - isolated["recall_by_state"]["emerging_move"],
                            6,
                        ),
                        "reversal_recall": round(
                            metrics["recall_by_state"]["reversal"]
                            - isolated["recall_by_state"]["reversal"],
                            6,
                        ),
                    },
                }
            )
    best_macro = max(variants, key=lambda item: item["metrics"]["macro_f1"])
    best_emerging = max(variants, key=lambda item: item["metrics"]["recall_by_state"]["emerging_move"])
    balanced = max(
        variants,
        key=lambda item: (
            item["metrics"]["recall_by_state"]["emerging_move"]
            if item["metrics"]["recall_by_state"]["reversal"] >= 0.40
            else -1.0,
            item["metrics"]["macro_f1"],
        ),
    )
    payload = {
        "rows": len(rows),
        "train_rows": len(train),
        "test_rows": len(test),
        "isolated_baseline": isolated,
        "variants": variants,
        "selection": {
            "best_macro_f1": _key(best_macro),
            "best_emerging_recall": _key(best_emerging),
            "balanced_candidate_reversal_ge_0_40": _key(balanced),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _key(item: dict) -> dict:
    return {
        "self_bias": item["self_bias"],
        "temperature": item["temperature"],
        "metrics": item["metrics"],
        "delta_vs_isolated": item["delta_vs_isolated"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.history_root, args.labels, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["selection"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
