from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_v2_v1_market_structure_features import _evaluate_horizon, _nearest_centroid_prediction


ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLES = ROOT.parent / ".runtime" / "reports" / "v2_market_breadth_observation_store_15m.json"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_selected_feature_market_switch_replay_15m.json"
SELECTED = {
    "1h": ("market_ret4_positive_share", "prefix_projected_leader_score_trend"),
    "2h": ("market_btc_ret4_pct", "market_volume_gt_mean20_share"),
}


def build(samples_path: Path, output: Path) -> dict:
    source = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = source["samples"]
    horizons = {horizon: _replay_horizon(samples, horizon, list(names)) for horizon, names in SELECTED.items()}
    payload = {"source_samples": str(samples_path), "selected_features": {k: list(v) for k, v in SELECTED.items()}, "horizons": horizons}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _replay_horizon(samples: list[dict], horizon: str, names: list[str]) -> dict:
    selected = []
    for sample in samples:
        if sample["horizon"] != horizon:
            continue
        features = sample.get("combined_features", {})
        selected.append(
            {
                "day": sample["day"],
                "anchor_ts_ms": int(sample["anchor_ts_ms"]),
                "label": sample["label"],
                "reward_delta": float(sample["reward_delta"]),
                "features": {name: float(features.get(name, 0.0)) for name in names},
            }
        )

    history = []
    predictions = []
    for sample in sorted(selected, key=lambda s: (s["day"], s["anchor_ts_ms"])):
        labels = {item["label"] for item in history}
        if labels == {"candidate_favorable", "candidate_unfavorable"}:
            pred, confidence = _nearest_centroid_prediction(history, sample["features"])
            choose_candidate = pred == "candidate_favorable"
            predictions.append(
                {
                    "day": sample["day"],
                    "anchor_ts_ms": sample["anchor_ts_ms"],
                    "actual": sample["label"],
                    "predicted": pred,
                    "confidence": confidence,
                    "correct": pred == sample["label"],
                    "reward_delta": sample["reward_delta"],
                    "switched_delta_vs_base": sample["reward_delta"] if choose_candidate else 0.0,
                    "oracle_delta_vs_base": max(sample["reward_delta"], 0.0),
                }
            )
        history.append(sample)

    fixed_candidate_delta = sum(p["reward_delta"] for p in predictions)
    switched_delta = sum(p["switched_delta_vs_base"] for p in predictions)
    oracle_delta = sum(p["oracle_delta_vs_base"] for p in predictions)
    classifier = _evaluate_horizon([{k: v for k, v in s.items() if k != "reward_delta"} for s in selected])
    pred_counts = Counter(p["predicted"] for p in predictions)
    wrong_confident_loss = sum(p["reward_delta"] for p in predictions if not p["correct"] and p["predicted"] == "candidate_favorable")
    return {
        "features": names,
        "classifier": classifier,
        "prediction_counts": dict(sorted(pred_counts.items())),
        "fixed_base_delta": 0.0,
        "fixed_candidate_delta_vs_base": round(fixed_candidate_delta, 6),
        "switched_delta_vs_base": round(switched_delta, 6),
        "switched_delta_vs_candidate": round(switched_delta - fixed_candidate_delta, 6),
        "oracle_delta_vs_base": round(oracle_delta, 6),
        "oracle_gap": round(oracle_delta - switched_delta, 6),
        "wrong_confident_candidate_loss": round(wrong_confident_loss, 6),
        "verdict": _verdict(classifier, switched_delta, fixed_candidate_delta),
        "predictions": predictions,
    }


def _verdict(classifier: dict, switched_delta: float, fixed_candidate_delta: float) -> str:
    if classifier["accuracy_edge"] <= 0.03:
        return "reject_classifier_gate"
    if switched_delta <= 0:
        return "reject_loses_to_base"
    if switched_delta <= fixed_candidate_delta:
        return "reject_loses_to_candidate"
    return "promising_next_full_replay_gate"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.samples, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["horizons"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
