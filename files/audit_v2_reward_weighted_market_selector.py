from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from audit_v2_selected_feature_market_switch_replay import SELECTED, _replay_horizon as _unweighted_replay_horizon
from audit_v2_v1_market_structure_features import _evaluate_horizon, _nearest_centroid_prediction


ROOT = Path(__file__).resolve().parent
DEFAULT_SAMPLES = ROOT.parent / ".runtime" / "reports" / "v2_market_breadth_observation_store_15m.json"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_reward_weighted_market_selector_15m.json"
K_VALUES = (3, 5, 8)
DOWNSIDE_MULTIPLIERS = (1.0, 2.0, 3.0)
EDGE_THRESHOLDS = (0.0, 1.0, 2.0, 5.0)
CONF_THRESHOLDS = (0.55, 0.60, 0.65)
MIN_CANDIDATE_SHARE = 0.05


def build(samples_path: Path, output: Path) -> dict:
    source = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = source["samples"]
    horizons = {}
    for horizon, names in SELECTED.items():
        selected = _select_samples(samples, horizon, list(names))
        unweighted = _unweighted_replay_horizon(samples, horizon, list(names))
        horizons[horizon] = _evaluate_selector_family(selected, list(names), unweighted)
    payload = {
        "source_samples": str(samples_path),
        "selected_features": {horizon: list(names) for horizon, names in SELECTED.items()},
        "parameters": {
            "k_values": list(K_VALUES),
            "downside_multipliers": list(DOWNSIDE_MULTIPLIERS),
            "edge_thresholds": list(EDGE_THRESHOLDS),
            "confidence_thresholds": list(CONF_THRESHOLDS),
            "hybrid_grid_enabled": False,
            "min_candidate_share": MIN_CANDIDATE_SHARE,
        },
        "horizons": horizons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _select_samples(samples: list[dict], horizon: str, names: list[str]) -> list[dict]:
    out = []
    for sample in samples:
        if sample["horizon"] != horizon:
            continue
        features = sample.get("combined_features", {})
        out.append(
            {
                "day": sample["day"],
                "anchor_ts_ms": int(sample["anchor_ts_ms"]),
                "label": sample["label"],
                "reward_delta": float(sample["reward_delta"]),
                "features": {name: float(features.get(name, 0.0)) for name in names},
            }
        )
    return sorted(out, key=lambda s: (s["day"], s["anchor_ts_ms"]))


def _evaluate_selector_family(samples: list[dict], names: list[str], unweighted: dict) -> dict:
    fixed_candidate_delta = _fixed_candidate_delta(samples)
    oracle_delta = sum(max(0.0, sample["reward_delta"]) for sample in samples[2:])
    classifier = _evaluate_horizon([{k: v for k, v in s.items() if k != "reward_delta"} for s in samples])
    candidates = []
    for k in K_VALUES:
        for downside in DOWNSIDE_MULTIPLIERS:
            for threshold in EDGE_THRESHOLDS:
                candidates.append(_replay_knn(samples, names, k=k, downside_multiplier=downside, edge_threshold=threshold))
    for confidence in CONF_THRESHOLDS:
        candidates.append(_replay_confidence_centroid(samples, names, confidence_threshold=confidence))
    candidates.sort(key=lambda item: (item["verdict_rank"], item["switched_delta_vs_base"], -item["wrong_candidate_loss"]), reverse=True)
    best = candidates[0] if candidates else None
    return {
        "features": names,
        "classifier": classifier,
        "fixed_base_delta": 0.0,
        "fixed_candidate_delta_vs_base": round(fixed_candidate_delta, 6),
        "unweighted_selected_switch": _compact_unweighted(unweighted),
        "oracle_delta_vs_base": round(oracle_delta, 6),
        "best": best,
        "top_candidates": candidates[:10],
        "decision": _decision(best, fixed_candidate_delta, unweighted),
    }


def _replay_knn(samples: list[dict], names: list[str], *, k: int, downside_multiplier: float, edge_threshold: float) -> dict:
    def decide(history: list[dict], sample: dict) -> tuple[bool, dict]:
        expected, neighbors = _reward_weighted_knn_expected_delta(history, sample["features"], names, k, downside_multiplier)
        return expected > edge_threshold, {"expected_delta": expected, "neighbors": neighbors}

    return _replay(samples, f"knn_k{k}_down{downside_multiplier}_thr{edge_threshold}", decide, {"k": k, "downside_multiplier": downside_multiplier, "edge_threshold": edge_threshold})


def _replay_confidence_centroid(samples: list[dict], names: list[str], *, confidence_threshold: float) -> dict:
    def decide(history: list[dict], sample: dict) -> tuple[bool, dict]:
        pred, confidence = _nearest_centroid_prediction(history, sample["features"])
        return pred == "candidate_favorable" and confidence >= confidence_threshold, {"predicted": pred, "confidence": confidence}

    return _replay(samples, f"centroid_conf_{confidence_threshold}", decide, {"confidence_threshold": confidence_threshold})


def _replay_hybrid(samples: list[dict], names: list[str], *, k: int, downside_multiplier: float, edge_threshold: float, confidence_threshold: float) -> dict:
    def decide(history: list[dict], sample: dict) -> tuple[bool, dict]:
        pred, confidence = _nearest_centroid_prediction(history, sample["features"])
        expected, neighbors = _reward_weighted_knn_expected_delta(history, sample["features"], names, k, downside_multiplier)
        choose = pred == "candidate_favorable" and confidence >= confidence_threshold and expected > edge_threshold
        return choose, {"predicted": pred, "confidence": confidence, "expected_delta": expected, "neighbors": neighbors}

    return _replay(samples, f"hybrid_k{k}_down{downside_multiplier}_thr{edge_threshold}_conf{confidence_threshold}", decide, {"k": k, "downside_multiplier": downside_multiplier, "edge_threshold": edge_threshold, "confidence_threshold": confidence_threshold})


def _replay(samples: list[dict], name: str, decide, params: dict) -> dict:
    history = []
    rows = []
    for sample in samples:
        labels = {item["label"] for item in history}
        if labels == {"candidate_favorable", "candidate_unfavorable"}:
            choose_candidate, meta = decide(history, sample)
            reward_delta = sample["reward_delta"] if choose_candidate else 0.0
            rows.append(
                {
                    "day": sample["day"],
                    "anchor_ts_ms": sample["anchor_ts_ms"],
                    "actual": sample["label"],
                    "choose_candidate": choose_candidate,
                    "reward_delta": sample["reward_delta"],
                    "switched_delta_vs_base": reward_delta,
                    **meta,
                }
            )
        history.append(sample)
    selected = [row for row in rows if row["choose_candidate"]]
    wrong_candidate = [row for row in selected if row["reward_delta"] <= 0.0]
    missed_positive = [row for row in rows if not row["choose_candidate"] and row["reward_delta"] > 0.0]
    switched = sum(row["switched_delta_vs_base"] for row in rows)
    fixed_candidate = sum(row["reward_delta"] for row in rows)
    candidate_share = len(selected) / len(rows) if rows else 0.0
    result = {
        "name": name,
        "params": params,
        "prediction_coverage": len(rows),
        "candidate_count": len(selected),
        "candidate_share": round(candidate_share, 6),
        "wrong_candidate_count": len(wrong_candidate),
        "wrong_candidate_loss": round(sum(row["reward_delta"] for row in wrong_candidate), 6),
        "missed_positive_count": len(missed_positive),
        "missed_positive_reward": round(sum(row["reward_delta"] for row in missed_positive), 6),
        "switched_delta_vs_base": round(switched, 6),
        "switched_delta_vs_candidate": round(switched - fixed_candidate, 6),
    }
    result["verdict_rank"] = _verdict_rank(result, fixed_candidate)
    result["verdict"] = _selector_verdict(result, fixed_candidate)
    return result


def _reward_weighted_knn_expected_delta(history: list[dict], features: dict, names: list[str], k: int, downside_multiplier: float) -> tuple[float, list[dict]]:
    stats = _history_stats(history, names)
    current = _standardize(features, names, stats)
    neighbors = []
    for item in history:
        distance = _distance(current, _standardize(item["features"], names, stats), names)
        weight = 1.0 / max(distance, 1e-9)
        raw_delta = float(item["reward_delta"])
        adjusted = raw_delta if raw_delta > 0 else raw_delta * downside_multiplier
        neighbors.append({"distance": distance, "weight": weight, "raw_delta": raw_delta, "adjusted_delta": adjusted})
    neighbors.sort(key=lambda item: item["distance"])
    top = neighbors[:k]
    denom = sum(item["weight"] for item in top) or 1.0
    expected = sum(item["adjusted_delta"] * item["weight"] for item in top) / denom
    return round(expected, 6), [
        {"distance": round(n["distance"], 6), "raw_delta": round(n["raw_delta"], 6), "adjusted_delta": round(n["adjusted_delta"], 6)}
        for n in top
    ]


def _history_stats(history: list[dict], names: list[str]) -> dict[str, tuple[float, float]]:
    stats = {}
    for name in names:
        values = [float(item["features"].get(name, 0.0)) for item in history]
        mean = sum(values) / len(values) if values else 0.0
        var = sum((v - mean) ** 2 for v in values) / len(values) if values else 0.0
        stats[name] = (mean, max(math.sqrt(var), 1e-9))
    return stats


def _standardize(features: dict, names: list[str], stats: dict[str, tuple[float, float]]) -> dict:
    return {name: (float(features.get(name, 0.0)) - stats[name][0]) / stats[name][1] for name in names}


def _distance(left: dict, right: dict, names: list[str]) -> float:
    return math.sqrt(sum((float(left[name]) - float(right[name])) ** 2 for name in names))


def _fixed_candidate_delta(samples: list[dict]) -> float:
    history = []
    total = 0.0
    for sample in samples:
        labels = {item["label"] for item in history}
        if labels == {"candidate_favorable", "candidate_unfavorable"}:
            total += sample["reward_delta"]
        history.append(sample)
    return total


def _compact_unweighted(result: dict) -> dict:
    return {key: result[key] for key in ("fixed_candidate_delta_vs_base", "switched_delta_vs_base", "switched_delta_vs_candidate", "oracle_delta_vs_base", "wrong_confident_candidate_loss", "verdict")}


def _verdict_rank(result: dict, fixed_candidate_delta: float) -> int:
    if result["candidate_share"] < MIN_CANDIDATE_SHARE:
        return 0
    if result["switched_delta_vs_base"] <= 0:
        return 1
    if result["switched_delta_vs_base"] <= fixed_candidate_delta:
        return 2
    return 3


def _selector_verdict(result: dict, fixed_candidate_delta: float) -> str:
    if result["candidate_share"] < MIN_CANDIDATE_SHARE:
        return "reject_trivial_candidate_count"
    if result["switched_delta_vs_base"] <= 0:
        return "reject_loses_to_base"
    if result["switched_delta_vs_base"] <= fixed_candidate_delta:
        return "reject_loses_to_candidate"
    return "promising_next_full_replay_gate"


def _decision(best: dict | None, fixed_candidate_delta: float, unweighted: dict) -> dict:
    if not best:
        return {"promotion_gate_passed": False, "recommendation": "no selector produced output"}
    beats_unweighted = best["switched_delta_vs_base"] > unweighted["switched_delta_vs_base"]
    passed = best["verdict"] == "promising_next_full_replay_gate" and beats_unweighted
    return {
        "promotion_gate_passed": passed,
        "beats_unweighted": beats_unweighted,
        "recommendation": "run full offline environment replay" if passed else "keep research-only; improve downside calibration",
    }


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
