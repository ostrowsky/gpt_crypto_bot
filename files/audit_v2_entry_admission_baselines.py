from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_dataset_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_baselines_15m.json"
POSITIVE_STATES = {"emerging_move", "confirmed_trend"}


def build(dataset_path: Path, output: Path) -> dict:
    rows = _load_rows(dataset_path)
    variants = []
    for belief_threshold in (0.50, 0.60, 0.70):
        variants.append(
            _evaluate(
                rows,
                family="belief_only",
                belief_threshold=belief_threshold,
                leader_threshold=None,
                require_temporal=False,
            )
        )
    for leader_threshold in (3.0, 5.0, 8.0):
        variants.append(
            _evaluate(
                rows,
                family="projected_v1_only",
                belief_threshold=None,
                leader_threshold=leader_threshold,
                require_temporal=False,
            )
        )
    for belief_threshold in (0.50, 0.60, 0.70):
        for leader_threshold in (3.0, 5.0, 8.0):
            variants.append(
                _evaluate(
                    rows,
                    family="belief_plus_projected_v1",
                    belief_threshold=belief_threshold,
                    leader_threshold=leader_threshold,
                    require_temporal=False,
                )
            )
            variants.append(
                _evaluate(
                    rows,
                    family="belief_plus_projected_v1_plus_temporal",
                    belief_threshold=belief_threshold,
                    leader_threshold=leader_threshold,
                    require_temporal=True,
                )
            )
    belief_only_best = max(
        [item for item in variants if item["family"] == "belief_only"],
        key=lambda item: (item["metrics"]["admission_precision"], item["metrics"]["emerging_move_recall"]),
    )
    recall_floor = belief_only_best["metrics"]["emerging_move_recall"] * 0.90
    recall_preserving = [
        item
        for item in variants
        if item["metrics"]["emerging_move_recall"] >= recall_floor
    ]
    best_recall_preserving = max(
        recall_preserving,
        key=lambda item: (
            item["metrics"]["admission_precision"],
            -item["metrics"]["noise_admission_rate"],
        ),
    )
    best_balanced = max(
        variants,
        key=lambda item: (
            item["metrics"]["admission_precision"],
            item["metrics"]["emerging_move_recall"],
            -item["metrics"]["noise_admission_rate"],
        ),
    )
    payload = {
        "rows": len(rows),
        "variants": variants,
        "selection": {
            "belief_only_best": _key(belief_only_best),
            "best_recall_preserving": _key(best_recall_preserving),
            "best_balanced": _key(best_balanced),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _load_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _evaluate(
    rows: list[dict],
    *,
    family: str,
    belief_threshold: float | None,
    leader_threshold: float | None,
    require_temporal: bool,
) -> dict:
    admitted = [row for row in rows if _admit(row, belief_threshold, leader_threshold, require_temporal)]
    state_counts = Counter(row["true_state"] for row in admitted)
    positives_total = sum(1 for row in rows if row["true_state"] in POSITIVE_STATES)
    emerging_total = sum(1 for row in rows if row["true_state"] == "emerging_move")
    true_positive = sum(1 for row in admitted if row["true_state"] in POSITIVE_STATES)
    noise_admitted = state_counts.get("noise", 0)
    emerging_admitted = state_counts.get("emerging_move", 0)
    metrics = {
        "admitted_rows": len(admitted),
        "admitted_state_counts": dict(sorted(state_counts.items())),
        "admission_precision": round(true_positive / len(admitted), 6) if admitted else 0.0,
        "positive_recall": round(true_positive / positives_total, 6) if positives_total else 0.0,
        "emerging_move_recall": round(emerging_admitted / emerging_total, 6) if emerging_total else 0.0,
        "noise_admission_rate": round(noise_admitted / len(admitted), 6) if admitted else 0.0,
    }
    return {
        "family": family,
        "belief_threshold": belief_threshold,
        "leader_threshold": leader_threshold,
        "require_temporal": require_temporal,
        "metrics": metrics,
    }


def _admit(row: dict, belief_threshold: float | None, leader_threshold: float | None, require_temporal: bool) -> bool:
    belief = row["belief"]
    projected = row.get("v1_projected_structural") or {}
    temporal = row.get("v1_temporal") or {}
    belief_ok = (
        True
        if belief_threshold is None
        else float(belief.get("emerging_move", 0.0)) + float(belief.get("confirmed_trend", 0.0)) >= belief_threshold
    )
    leader_ok = (
        True
        if leader_threshold is None
        else float(projected.get("projected_leader_score_trend", 0.0)) >= leader_threshold
    )
    temporal_ok = (not require_temporal) or bool(temporal.get("prior_structural_scout"))
    return belief_ok and leader_ok and temporal_ok


def _key(item: dict) -> dict:
    return {
        "family": item["family"],
        "belief_threshold": item["belief_threshold"],
        "leader_threshold": item["leader_threshold"],
        "require_temporal": item["require_temporal"],
        "metrics": item["metrics"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.dataset, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["selection"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
