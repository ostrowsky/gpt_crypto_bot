from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

from audit_post_block_causal_discriminator import _feat, _load_rows, _split_by_day


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = ROOT / ".runtime" / "reports" / "post_block_causal_discriminator_dataset_15m.jsonl"
DEFAULT_OUTPUT = ROOT / ".runtime" / "reports" / "post_block_experiment_suite_15m.json"

RETURN_FEATURES = ["rel_ret_15m_pct", "rel_ret_30m_pct", "rel_ret_60m_pct", "rel_ret_120m_pct"]
UPSIDE_FEATURES = ["max_high_60m_pct", "max_high_120m_pct"]
EXPANSION_FEATURES = ["volume_x_60m", "volume_x_120m", "range_x_60m", "range_x_120m"]


def build(dataset: Path = DEFAULT_DATASET, output: Path = DEFAULT_OUTPUT, train_fraction: float = 0.70) -> dict:
    rows = _load_rows(dataset)
    train, holdout = _split_by_day(rows, train_fraction)
    targets = _target_defs()
    experiments = []
    for name, target_fn in targets.items():
        experiments.append(_run_target(name, target_fn, train, holdout))
    experiments = sorted(experiments, key=_target_rank, reverse=True)
    selected = experiments[0] if experiments else None
    payload = {
        "dataset": str(dataset),
        "rows": len(rows),
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "experiments": experiments,
        "selected_direction": selected,
        "decision": _decision(selected),
        "next_action": _next_action(selected),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _target_defs():
    return {
        "useful_missed_winner": lambda r: bool(r.get("label_useful_missed_winner")),
        "final_top15": lambda r: bool(r.get("label_top15")),
        "tradable_60m": lambda r: _feat(r, "max_high_60m_pct") >= 2.0 and _feat(r, "min_low_60m_pct") >= -2.5,
        "tradable_120m": lambda r: _feat(r, "max_high_120m_pct") >= 3.0 and _feat(r, "min_low_120m_pct") >= -3.0,
        "top15_and_tradable_120m": lambda r: bool(r.get("label_top15")) and _feat(r, "max_high_120m_pct") >= 3.0 and _feat(r, "min_low_120m_pct") >= -3.0,
    }


def _run_target(name: str, target_fn, train: list[dict], holdout: list[dict]) -> dict:
    variants = _rule_variants(holdout, target_fn)
    variants = sorted(variants, key=lambda item: (item["passes_gate"], item["precision_lift"], item["positives"], item["precision"], -item["candidate_count"]), reverse=True)
    best = variants[0] if variants else None
    return {
        "target": name,
        "train_base": _base(train, target_fn),
        "holdout_base": _base(holdout, target_fn),
        "best_variant": best,
        "top_variants": variants[:15],
    }


def _rule_variants(rows: list[dict], target_fn) -> list[dict]:
    variants = []
    for feature in RETURN_FEATURES + UPSIDE_FEATURES:
        for threshold in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]:
            variants.append(_evaluate(rows, target_fn, "single_return_or_upside", [feature], {"threshold": threshold}, lambda r, f=feature, t=threshold: _feat(r, f) >= t))
    for feature in EXPANSION_FEATURES:
        for threshold in [1.2, 1.5, 2.0, 3.0, 5.0]:
            variants.append(_evaluate(rows, target_fn, "single_expansion", [feature], {"threshold": threshold}, lambda r, f=feature, t=threshold: _feat(r, f) >= t))
    for ret_feature, exp_feature in product(RETURN_FEATURES + UPSIDE_FEATURES, EXPANSION_FEATURES):
        for ret_t in [0.5, 1.0, 2.0, 3.0]:
            for exp_t in [1.5, 2.0, 3.0]:
                variants.append(_evaluate(rows, target_fn, "return_or_upside_and_expansion", [ret_feature, exp_feature], {"value_threshold": ret_t, "expansion_threshold": exp_t}, lambda r, rf=ret_feature, ef=exp_feature, rt=ret_t, et=exp_t: _feat(r, rf) >= rt and _feat(r, ef) >= et))
    return variants


def _evaluate(rows: list[dict], target_fn, family: str, features: list[str], params: dict, predicate) -> dict:
    selected = [row for row in rows if predicate(row)]
    total_pos = sum(1 for row in rows if target_fn(row))
    positives = sum(1 for row in selected if target_fn(row))
    top15 = sum(1 for row in selected if row.get("label_top15"))
    bad = sum(1 for row in selected if row.get("label_bad_candidate"))
    base = total_pos / len(rows) if rows else 0.0
    precision = positives / len(selected) if selected else 0.0
    return {
        "family": family,
        "features": features,
        "params": params,
        "candidate_count": len(selected),
        "positives": positives,
        "precision": round(precision, 6),
        "precision_lift": round((precision / base), 6) if base > 0 and selected else 0.0,
        "recall": round(positives / total_pos, 6) if total_pos else 0.0,
        "top15_precision": round(top15 / len(selected), 6) if selected else 0.0,
        "bad_ratio": round(bad / len(selected), 6) if selected else 0.0,
        "passes_gate": _passes_gate(len(selected), positives, precision, base),
        "examples": [_compact(row) for row in selected if target_fn(row)][:8],
    }


def _passes_gate(candidate_count: int, positives: int, precision: float, base: float) -> bool:
    return candidate_count >= 10 and candidate_count <= 250 and positives >= 5 and precision >= max(base * 2.0, 0.10)


def _base(rows: list[dict], target_fn) -> dict:
    n = len(rows) or 1
    positives = sum(1 for row in rows if target_fn(row))
    return {"rows": len(rows), "positives": positives, "rate": round(positives / n, 6)}


def _target_rank(item: dict) -> tuple:
    best = item.get("best_variant") or {}
    return (bool(best.get("passes_gate")), best.get("precision_lift", 0.0), best.get("positives", 0), best.get("precision", 0.0))


def _decision(selected: dict | None) -> str:
    if not selected or not (selected.get("best_variant") or {}).get("passes_gate"):
        return "all_directions_rejected_or_need_better_labels"
    return f"continue_{selected['target']}"


def _next_action(selected: dict | None) -> str:
    if not selected or not (selected.get("best_variant") or {}).get("passes_gate"):
        return "Stop threshold/model iteration; inspect labels/features manually or add v1 score/rank trajectory."
    target = selected["target"]
    if target.startswith("tradable"):
        return "Continue with tradability-first discriminator and then intersect with top-mover objective in replay."
    return f"Continue target {target} with a focused behavior replay candidate."


def _compact(row: dict) -> dict:
    return {
        "day": row.get("local_day"),
        "symbol": row.get("symbol"),
        "top15": row.get("label_top15"),
        "useful_missed": row.get("label_useful_missed_winner"),
        "features": {k: row.get("features", {}).get(k) for k in ["max_high_60m_pct", "max_high_120m_pct", "min_low_60m_pct", "rel_ret_60m_pct", "volume_x_60m", "range_x_60m"]},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.dataset, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else {"decision": payload["decision"], "next_action": payload["next_action"], "selected": payload["selected_direction"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
