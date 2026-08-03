from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "files" / "ml_candidate_ranker_report.json"
DEFAULT_OUTPUT = ROOT / ".runtime" / "reports" / "portfolio_ev_ranker_pre_gate_latest.json"
DEFAULT_TEXT_OUTPUT = ROOT / ".runtime" / "reports" / "portfolio_ev_ranker_pre_gate_latest.txt"


@dataclass(frozen=True)
class PreGateConfig:
    required_top_n: tuple[int, ...] = (1, 3, 5)
    min_eligible_groups: int = 100
    min_target_return_delta_pct: float = 0.05
    min_win_rate_delta: float = 0.0
    max_drawdown_delta_pct: float = 0.0
    min_top_gainer_rate_delta: float = 0.0
    min_capture_ratio_delta: float = 0.0


def build_report(payload: dict[str, Any], cfg: PreGateConfig = PreGateConfig()) -> dict[str, Any]:
    grouped = payload.get("test_group_ranking") or {}
    rows_by_top_n = {
        int(row.get("top_n") or 0): row
        for row in (grouped.get("top_n") or [])
        if isinstance(row, dict)
    }
    slices = [_evaluate_slice(rows_by_top_n.get(top_n), top_n, cfg) for top_n in cfg.required_top_n]
    coverage_ok = (
        int(payload.get("test_rows") or 0) > 0
        and int(grouped.get("grouped_competitions") or 0) >= cfg.min_eligible_groups
        and all(item["coverage_ok"] for item in slices)
    )
    passed = coverage_ok and all(item["passed"] for item in slices)
    if not coverage_ok:
        decision = "insufficient_chronological_test_coverage"
    elif passed:
        decision = "advance_to_full_ten_slot_portfolio_replay"
    else:
        decision = "reject_current_ranker_for_capacity_ranking"
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "complete" if coverage_ok else "partial",
        "decision": decision,
        "config": asdict(cfg),
        "coverage": {
            "rows_total": int(payload.get("rows_total") or 0),
            "train_rows": int(payload.get("train_rows") or 0),
            "validation_rows": int(payload.get("val_rows") or 0),
            "test_rows": int(payload.get("test_rows") or 0),
            "grouped_test_competitions": int(grouped.get("grouped_competitions") or 0),
            "chosen_model": str(payload.get("chosen_model") or ""),
        },
        "slices": slices,
        "failed_checks": [
            check
            for item in slices
            for check in item["failed_checks"]
        ],
    }


def _evaluate_slice(row: dict[str, Any] | None, top_n: int, cfg: PreGateConfig) -> dict[str, Any]:
    if not row:
        return {
            "top_n": top_n,
            "eligible_groups": 0,
            "coverage_ok": False,
            "passed": False,
            "delta": {},
            "failed_checks": [f"top{top_n}:missing_slice"],
        }
    delta = row.get("delta") or {}
    eligible = int(row.get("eligible_groups") or 0)
    checks = {
        "eligible_groups": eligible >= cfg.min_eligible_groups,
        "avg_target_return": _num(delta.get("avg_target_return")) >= cfg.min_target_return_delta_pct,
        "win_rate": _num(delta.get("win_rate")) >= cfg.min_win_rate_delta,
        "avg_drawdown": _num(delta.get("avg_drawdown")) <= cfg.max_drawdown_delta_pct,
        "teacher_top_gainer_rate": _num(delta.get("teacher_top_gainer_rate")) >= cfg.min_top_gainer_rate_delta,
        "teacher_capture_ratio": _num(delta.get("teacher_capture_ratio")) >= cfg.min_capture_ratio_delta,
    }
    failed = [f"top{top_n}:{name}" for name, ok in checks.items() if not ok]
    return {
        "top_n": top_n,
        "eligible_groups": eligible,
        "coverage_ok": checks["eligible_groups"],
        "passed": not failed,
        "baseline": row.get("baseline") or {},
        "ranker": row.get("ranker") or {},
        "delta": delta,
        "overlap_ratio": row.get("overlap_ratio"),
        "checks": checks,
        "failed_checks": failed,
    }


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def render_text(report: dict[str, Any]) -> str:
    coverage = report.get("coverage") or {}
    lines = [
        "Portfolio EV/opportunity ranker pre-gate",
        f"status: {report.get('status')}",
        f"decision: {report.get('decision')}",
        (
            "coverage: "
            f"rows={coverage.get('rows_total')} train={coverage.get('train_rows')} "
            f"validation={coverage.get('validation_rows')} test={coverage.get('test_rows')} "
            f"groups={coverage.get('grouped_test_competitions')}"
        ),
        "",
    ]
    for item in report.get("slices") or []:
        delta = item.get("delta") or {}
        lines.append(
            f"top-{item.get('top_n')}: groups={item.get('eligible_groups')} passed={item.get('passed')} "
            f"ret={_fmt(delta.get('avg_target_return'))} "
            f"win={_fmt(delta.get('win_rate'))} "
            f"dd={_fmt(delta.get('avg_drawdown'))} "
            f"top={_fmt(delta.get('teacher_top_gainer_rate'))} "
            f"capture={_fmt(delta.get('teacher_capture_ratio'))}"
        )
    if report.get("failed_checks"):
        lines.extend(["", "failed: " + ", ".join(report["failed_checks"])])
    return "\n".join(lines) + "\n"


def _fmt(value: Any) -> str:
    try:
        return f"{float(value):+.4f}"
    except (TypeError, ValueError):
        return "n/a"


def main() -> int:
    parser = argparse.ArgumentParser(description="Chronological pre-gate for portfolio EV/opportunity ranking")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--text-output", type=Path, default=DEFAULT_TEXT_OUTPUT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    raw = args.input.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    report = build_report(payload)
    report["input"] = {
        "path": str(args.input),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    text = render_text(report)
    if not args.no_save:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        args.text_output.write_text(text, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
