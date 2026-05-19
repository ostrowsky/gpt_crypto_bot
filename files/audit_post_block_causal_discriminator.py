from __future__ import annotations

import argparse
import json
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / ".runtime" / "reports" / "post_block_causal_discriminator_dataset_15m.jsonl"
DEFAULT_OUTPUT = ROOT / ".runtime" / "reports" / "post_block_causal_discriminator_audit_15m.json"

FEATURES = [
    "rel_ret_15m_pct",
    "rel_ret_30m_pct",
    "rel_ret_60m_pct",
    "rel_ret_120m_pct",
    "ret_60m_pct",
    "ret_120m_pct",
    "max_high_60m_pct",
    "max_high_120m_pct",
    "volume_x_60m",
    "volume_x_120m",
    "range_x_60m",
    "range_x_120m",
]
RETURN_FEATURES = ["rel_ret_15m_pct", "rel_ret_30m_pct", "rel_ret_60m_pct", "rel_ret_120m_pct"]
EXPANSION_FEATURES = ["volume_x_60m", "volume_x_120m", "range_x_60m", "range_x_120m"]


def build(dataset_path: Path = DEFAULT_DATASET, output: Path = DEFAULT_OUTPUT, train_fraction: float = 0.70) -> dict:
    rows = _load_rows(dataset_path)
    train, holdout = _split_by_day(rows, train_fraction)
    variants = []
    variants.extend(_rule_variants(holdout))
    variants.extend(_binned_variants(train, holdout))
    variants = sorted(variants, key=_rank_key, reverse=True)
    best = variants[0] if variants else None
    payload = {
        "dataset": str(dataset_path),
        "rows": len(rows),
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "train_days": _day_range(train),
        "holdout_days": _day_range(holdout),
        "base_rates": {"train": _base_rates(train), "holdout": _base_rates(holdout)},
        "best_variant": best,
        "top_variants": variants[:30],
        "decision": _decision(best, holdout),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return sorted(rows, key=lambda r: (str(r.get("local_day")), str(r.get("candidate_ts")), str(r.get("symbol"))))


def _split_by_day(rows: list[dict], train_fraction: float) -> tuple[list[dict], list[dict]]:
    days = sorted({str(r.get("local_day")) for r in rows})
    split = max(1, min(len(days) - 1, int(len(days) * train_fraction)))
    train_days = set(days[:split])
    return [r for r in rows if r.get("local_day") in train_days], [r for r in rows if r.get("local_day") not in train_days]


def _rule_variants(rows: list[dict]) -> list[dict]:
    variants = []
    ret_thresholds = [0.5, 1.0, 2.0, 3.0, 5.0]
    exp_thresholds = [1.2, 1.5, 2.0, 3.0, 5.0]
    for feature in RETURN_FEATURES:
        for threshold in ret_thresholds:
            variants.append(_evaluate(rows, "rule_return", [feature], lambda r, f=feature, t=threshold: _feat(r, f) >= t, {"threshold": threshold}))
    for feature in EXPANSION_FEATURES:
        for threshold in exp_thresholds:
            variants.append(_evaluate(rows, "rule_expansion", [feature], lambda r, f=feature, t=threshold: _feat(r, f) >= t, {"threshold": threshold}))
    for ret_feature, exp_feature in product(RETURN_FEATURES, EXPANSION_FEATURES):
        for ret_t in [0.5, 1.0, 2.0, 3.0]:
            for exp_t in [1.5, 2.0, 3.0]:
                variants.append(
                    _evaluate(
                        rows,
                        "rule_return_and_expansion",
                        [ret_feature, exp_feature],
                        lambda r, rf=ret_feature, ef=exp_feature, rt=ret_t, et=exp_t: _feat(r, rf) >= rt and _feat(r, ef) >= et,
                        {"return_threshold": ret_t, "expansion_threshold": exp_t},
                    )
                )
    return variants


def _binned_variants(train: list[dict], holdout: list[dict]) -> list[dict]:
    variants = []
    for feature in FEATURES:
        variants.append(_binned_variant(train, holdout, [feature], bins=5, min_support=20))
    for pair in combinations(FEATURES[:8], 2):
        variants.append(_binned_variant(train, holdout, list(pair), bins=4, min_support=15))
    return [v for v in variants if v is not None]


def _binned_variant(train: list[dict], holdout: list[dict], features: list[str], bins: int, min_support: int) -> dict | None:
    edges = {feature: _edges([_feat(row, feature) for row in train], bins) for feature in features}
    stats = defaultdict(lambda: [0, 0])
    for row in train:
        key = tuple(_bin(_feat(row, feature), edges[feature]) for feature in features)
        stats[key][0] += int(row.get("label_useful_missed_winner"))
        stats[key][1] += 1
    useful_rates = {key: pos / total for key, (pos, total) in stats.items() if total >= min_support}
    if not useful_rates:
        return None
    train_base = _base_rates(train)["useful_rate"]
    selected_bins = {key for key, rate in useful_rates.items() if rate > train_base * 1.5 and rate >= 0.05}
    if not selected_bins:
        selected_bins = {max(useful_rates, key=useful_rates.get)}
    return _evaluate(
        holdout,
        "binned_lookup",
        features,
        lambda r: tuple(_bin(_feat(r, feature), edges[feature]) for feature in features) in selected_bins,
        {"bins": bins, "min_support": min_support, "selected_bins": [list(k) for k in sorted(selected_bins)], "train_base_useful_rate": round(train_base, 6)},
    )


def _evaluate(rows: list[dict], family: str, features: list[str], predicate, params: dict) -> dict:
    selected = [row for row in rows if predicate(row)]
    useful_total = sum(1 for row in rows if row.get("label_useful_missed_winner"))
    top_total = sum(1 for row in rows if row.get("label_top15"))
    useful = sum(1 for row in selected if row.get("label_useful_missed_winner"))
    top = sum(1 for row in selected if row.get("label_top15"))
    bad = sum(1 for row in selected if row.get("label_bad_candidate"))
    return {
        "family": family,
        "features": features,
        "params": params,
        "candidate_count": len(selected),
        "useful_missed_winners": useful,
        "top15_candidates": top,
        "bad_candidates": bad,
        "useful_precision": round(useful / len(selected), 6) if selected else 0.0,
        "top15_precision": round(top / len(selected), 6) if selected else 0.0,
        "bad_ratio": round(bad / len(selected), 6) if selected else 0.0,
        "useful_recall": round(useful / useful_total, 6) if useful_total else 0.0,
        "top15_recall": round(top / top_total, 6) if top_total else 0.0,
        "top_examples": sorted([_compact(row) for row in selected if row.get("label_useful_missed_winner")], key=lambda r: float(r.get("opportunity_from_first_block_pct") or 0.0), reverse=True)[:10],
    }


def _rank_key(item: dict) -> tuple:
    return (
        _passes_gate(item),
        item["useful_precision"],
        item["useful_missed_winners"],
        item["top15_precision"],
        -item["candidate_count"],
    )


def _passes_gate(item: dict) -> bool:
    return (
        item["candidate_count"] >= 10
        and item["candidate_count"] <= 250
        and item["useful_missed_winners"] >= 5
        and item["useful_precision"] >= 0.10
        and item["top15_precision"] >= 0.30
        and item["bad_ratio"] <= 0.70
    )


def _decision(best: dict | None, holdout: list[dict]) -> str:
    if not best:
        return "no_candidate"
    if _passes_gate(best):
        return "advance_to_behavior_replay_candidate"
    return "research_only_rejected_discriminator_gate"


def _feat(row: dict, name: str) -> float:
    value = (row.get("features") or {}).get(name)
    try:
        value = float(value)
        return value if value == value else 0.0
    except Exception:
        return 0.0


def _edges(values: list[float], bins: int) -> list[float]:
    if not values:
        return []
    arr = np.array(values, dtype=float)
    qs = [i / bins for i in range(1, bins)]
    edges = []
    for value in np.quantile(arr, qs):
        value = float(value)
        if not edges or abs(value - edges[-1]) > 1e-12:
            edges.append(value)
    return edges


def _bin(value: float, edges: list[float]) -> int:
    return int(np.searchsorted(np.array(edges, dtype=float), value, side="right"))


def _base_rates(rows: list[dict]) -> dict:
    n = len(rows) or 1
    return {
        "rows": len(rows),
        "useful_rate": round(sum(1 for r in rows if r.get("label_useful_missed_winner")) / n, 6),
        "top15_rate": round(sum(1 for r in rows if r.get("label_top15")) / n, 6),
        "bad_rate": round(sum(1 for r in rows if r.get("label_bad_candidate")) / n, 6),
    }


def _day_range(rows: list[dict]) -> dict:
    days = sorted({str(r.get("local_day")) for r in rows})
    return {"count": len(days), "start": days[0] if days else None, "end": days[-1] if days else None}


def _compact(row: dict) -> dict:
    return {
        "local_day": row.get("local_day"),
        "symbol": row.get("symbol"),
        "candidate_ts": row.get("candidate_ts"),
        "reason_code": row.get("reason_code"),
        "opportunity_from_first_block_pct": row.get("opportunity_from_first_block_pct"),
        "features": {k: row.get("features", {}).get(k) for k in ("rel_ret_60m_pct", "rel_ret_120m_pct", "volume_x_60m", "range_x_60m")},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.dataset, args.output, args.train_fraction)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else {"best_variant": payload["best_variant"], "decision": payload["decision"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
