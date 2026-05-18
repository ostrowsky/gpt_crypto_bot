from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_dataset_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_exhaustion_discrimination_audit_15m.json"
TARGET_STATES = {"mature_trend", "exhaustion"}


def build(dataset_path: Path, output: Path) -> dict:
    rows = [row for row in _load_rows(dataset_path) if row["true_state"] in TARGET_STATES]
    features = [_feature_row(row) for row in rows]
    grouped = defaultdict(list)
    for row, feat in zip(rows, features):
        grouped[row["true_state"]].append(feat)
    feature_names = sorted(features[0].keys()) if features else []
    means = {
        state: {name: round(_mean([row[name] for row in feats]), 6) for name in feature_names}
        for state, feats in grouped.items()
    }
    ranked = []
    for name in feature_names:
        mature_values = [row[name] for row in grouped["mature_trend"]]
        exhaustion_values = [row[name] for row in grouped["exhaustion"]]
        effect = _standardized_mean_difference(exhaustion_values, mature_values)
        ranked.append(
            {
                "feature": name,
                "mature_mean": round(_mean(mature_values), 6),
                "exhaustion_mean": round(_mean(exhaustion_values), 6),
                "standardized_mean_difference": round(effect, 6),
                "abs_effect": round(abs(effect), 6),
            }
        )
    ranked.sort(key=lambda item: item["abs_effect"], reverse=True)
    payload = {
        "rows": len(rows),
        "class_counts": {state: len(values) for state, values in sorted(grouped.items())},
        "feature_means": means,
        "ranked_features": ranked,
        "top_features": ranked[:10],
        "separation_verdict": _verdict(ranked),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _load_rows(path: Path) -> list[dict]:
    out = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _feature_row(row: dict) -> dict[str, float]:
    belief = row["belief"]
    projected = row.get("v1_projected_structural") or {}
    late_mass = float(belief.get("exhaustion", 0.0)) + float(belief.get("reversal", 0.0))
    # The current admission dataset does not yet carry bar-by-bar realized peak
    # context, so this first audit is intentionally limited to state + projected
    # structural evidence already available on every row.
    return {
        "belief_mature": float(belief.get("mature_trend", 0.0)),
        "belief_exhaustion": float(belief.get("exhaustion", 0.0)),
        "belief_reversal": float(belief.get("reversal", 0.0)),
        "belief_late_mass": late_mass,
        "belief_entropy": float(row.get("belief_entropy", 0.0)),
        "projected_forecast_proxy": float(projected.get("projected_forecast_proxy_pct", 0.0)),
        "projected_leader_score_trend": float(projected.get("projected_leader_score_trend", 0.0)),
        "slope": float(projected.get("slope", 0.0)),
        "adx": float(projected.get("adx", 0.0)),
        "rsi": float(projected.get("rsi", 0.0)),
        "vol_x": float(projected.get("vol_x", 0.0)),
        "daily_range_pct": float(projected.get("daily_range_pct", 0.0)),
        "price_vs_ema20_pct": float(projected.get("price_vs_ema20_pct", 0.0)),
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _standardized_mean_difference(a: list[float], b: list[float]) -> float:
    mean_a = _mean(a)
    mean_b = _mean(b)
    var_a = _variance(a, mean_a)
    var_b = _variance(b, mean_b)
    pooled = math.sqrt((var_a + var_b) / 2.0)
    return 0.0 if pooled <= 1e-12 else (mean_a - mean_b) / pooled


def _variance(values: list[float], mean: float) -> float:
    return sum((value - mean) ** 2 for value in values) / len(values) if values else 0.0


def _verdict(ranked: list[dict]) -> dict:
    strong = [item for item in ranked if item["abs_effect"] >= 0.80]
    moderate = [item for item in ranked if item["abs_effect"] >= 0.50]
    if strong:
        label = "interpretable_rule_candidate"
    elif moderate:
        label = "weak_to_moderate_separation"
    else:
        label = "weak_separation"
    return {
        "label": label,
        "strong_feature_count": len(strong),
        "moderate_feature_count": len(moderate),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.dataset, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["separation_verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
