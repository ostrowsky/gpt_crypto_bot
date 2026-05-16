from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / ".runtime" / "reports"


def _latest(pattern: str) -> Path | None:
    rows = sorted(REPORTS.glob(pattern))
    return rows[-1] if rows else None


def _load(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _metric_median(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, dict):
        raw = value.get("median")
        return None if raw is None else float(raw)
    return None


def build() -> dict[str, Any]:
    quality_path = _latest("signal_quality_*_final.json")
    critic_path = _latest("top_gainer_critic_*_final.json")
    quality = _load(quality_path)
    critic = _load(critic_path)
    q = quality.get("summary") or {}
    c = critic.get("summary") or {}
    hypotheses: list[dict[str, Any]] = []

    top_mover_missed = int(q.get("top_mover_missed_trends") or 0)
    miss_rate = q.get("miss_rate")
    false_positive_rate = q.get("false_positive_rate")
    early_exits = int(q.get("early_exits") or 0)
    exit_eff = _metric_median(q, "exit_efficiency")
    giveback = _metric_median(q, "giveback_pct")

    if top_mover_missed >= 10 and (miss_rate or 0) >= 0.65:
        hypotheses.append({
            "id": "trend_start_recall",
            "priority": 1,
            "problem": "Too many final top-mover trends are still missed.",
            "evidence": {
                "top_mover_missed_trends": top_mover_missed,
                "miss_rate": miss_rate,
                "false_positive_rate": false_positive_rate,
                "blocked_winner_reason_counts": c.get("blocked_winner_reason_counts") or {},
            },
            "proposal": "Audit first blocker chain for missed top movers and test one narrow trend-start bypass conditioned on wake-up scout evidence.",
            "required_backtest": "baseline vs one-factor bypass on 7d and 30d replay; require no worse capture and better early capture / precision tradeoff.",
            "auto_apply_allowed": False,
        })

    if early_exits > 0 and ((exit_eff is not None and exit_eff < 0.35) or (giveback is not None and giveback > 0.70)):
        hypotheses.append({
            "id": "exit_lifecycle",
            "priority": 2,
            "problem": "The bot is still giving back too much trend after profitable entry.",
            "evidence": {
                "early_exits": early_exits,
                "exit_efficiency_median": exit_eff,
                "giveback_pct_median": giveback,
            },
            "proposal": "Collect peak-risk shadow events and test exit changes only after shadow precision is measured.",
            "required_backtest": "exit variants against giveback_pct, exit_efficiency, total PnL, and capture ratio.",
            "auto_apply_allowed": False,
        })

    hypotheses.append({
        "id": "portfolio_replacement",
        "priority": 3,
        "problem": "Unified top-10 can still hide opportunity cost when a stronger candidate arrives after capacity is full.",
        "evidence": {
            "watchlist_top_bought": c.get("watchlist_top_bought"),
            "watchlist_top_count": c.get("watchlist_top_count"),
        },
        "proposal": "Run replacement grid only after enough 10/10 live windows are collected.",
        "required_backtest": "portfolio replacement grid under fixed top-10 cap; require improved objective capture without worse 30d PnL.",
        "auto_apply_allowed": False,
    })

    return {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": {
            "signal_quality": "" if quality_path is None else str(quality_path.relative_to(ROOT)),
            "top_gainer_critic": "" if critic_path is None else str(critic_path.relative_to(ROOT)),
        },
        "hypotheses": hypotheses,
    }


def render(report: dict[str, Any]) -> str:
    lines = ["Quality hypotheses", f"generated_at_utc: {report['generated_at_utc']}"]
    for row in report["hypotheses"]:
        lines.extend([
            "",
            f"{row['priority']}. {row['id']}",
            f"problem: {row['problem']}",
            f"proposal: {row['proposal']}",
            f"backtest: {row['required_backtest']}",
        ])
    return "\n".join(lines) + "\n"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = build()
    (REPORTS / "quality_hypotheses_latest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS / "quality_hypotheses_latest.txt").write_text(render(report), encoding="utf-8")


if __name__ == "__main__":
    main()
