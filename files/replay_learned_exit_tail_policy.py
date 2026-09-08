from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import replay_trailing_tail_after_partial_exit as tail_replay
import report_exit_failure_discriminator as discriminator


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / ".runtime" / "reports"
CACHE_DIR = ROOT / ".runtime" / "signal_quality_cache"
DEFAULT_OUTPUT = REPORTS / "learned_exit_tail_policy_replay_latest.json"
DEFAULT_TEXT_OUTPUT = REPORTS / "learned_exit_tail_policy_replay_latest.txt"


@dataclass(frozen=True)
class LearnedTailConfig:
    days: int = 0
    continuation_margin_pct: float = 0.75
    min_train_days: int = 3
    train_risk_quantile: float = 0.80
    policies: tuple[tail_replay.TailPolicy, ...] = (
        tail_replay.TailPolicy("learned_tail50_h5", 0.50, 5, 1.00),
        tail_replay.TailPolicy("learned_tail50_h10", 0.50, 10, 1.00),
        tail_replay.TailPolicy("learned_tail30_h5", 0.70, 5, 1.00),
        tail_replay.TailPolicy("learned_tail30_h10", 0.70, 10, 1.00),
    )


def _quantile(values: Iterable[float], q: float) -> float | None:
    vals = sorted(float(value) for value in values if value is not None)
    if not vals:
        return None
    q = min(max(float(q), 0.0), 1.0)
    pos = (len(vals) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return round(vals[lo], 6)
    weight = pos - lo
    return round(vals[lo] * (1.0 - weight) + vals[hi] * weight, 6)


def _p10(values: Iterable[float]) -> float | None:
    return _quantile(values, 0.10)


def _prepare_rows(
    reports_dir: Path,
    cache_dir: Path,
    cfg: LearnedTailConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    rows = discriminator._load_cases(
        cfg.days,
        reports_dir,
        continuation_margin_pct=cfg.continuation_margin_pct,
    )
    raw_cfg = tail_replay.hold_replay.ReplayConfig(days=cfg.days)
    raw_cases = tail_replay.hold_replay._load_cases(reports_dir, raw_cfg)
    raw_by_key = {_case_key(row): row for row in raw_cases}
    for row in rows:
        source = raw_by_key.get(_case_key(row))
        if source:
            row["entry_price"] = source.get("entry_price")
            row["exit_price"] = source.get("exit_price")
    train, test, split_status = discriminator._split_by_day(rows, cfg.min_train_days)
    baseline = discriminator._rate(train) or 0.0
    feature_rates = discriminator._train_feature_rates(train)
    train_scored = [
        {**row, "risk_score": discriminator._score(row, feature_rates, baseline)}
        for row in train
    ]
    test_scored = [
        {**row, "risk_score": discriminator._score(row, feature_rates, baseline)}
        for row in test
    ]
    max_horizon = max(policy.max_horizon for policy in cfg.policies)
    candle_cache = _build_candle_cache(cache_dir, test_scored)
    for row in test_scored:
        tail_replay._attach_candle_path(
            row,
            cache_dir,
            max_horizon,
            candle_cache=candle_cache,
        )
    return train_scored, test_scored, split_status


def _build_candle_cache(
    cache_dir: Path,
    rows: list[dict[str, Any]],
) -> dict[tuple[str, str], tuple[list[dict[str, Any]], list[int]]]:
    """Index the 20k+ cache directory once instead of globbing once per symbol."""
    wanted_times: dict[tuple[str, str], list[int]] = {}
    for row in rows:
        key = (str(row.get("sym") or ""), str(row.get("tf") or "15m"))
        ts = tail_replay.hold_replay._parse_ts_ms(row.get("exit_ts"))
        if ts is not None:
            wanted_times.setdefault(key, []).append(ts)
    paths: dict[tuple[str, str], list[Path]] = {key: [] for key in wanted_times}
    prior_buffer_ms = 4 * 24 * 60 * 60 * 1000
    future_buffer_ms = 24 * 60 * 60 * 1000
    for path in cache_dir.iterdir() if cache_dir.exists() else ():
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        parts = path.stem.split("_")
        if len(parts) < 4:
            continue
        key = (parts[0], parts[1])
        if key not in paths:
            continue
        try:
            start_ms, end_ms = int(parts[-2]), int(parts[-1])
        except ValueError:
            paths[key].append(path)
            continue
        if any(
            end_ms >= exit_ms - prior_buffer_ms and start_ms <= exit_ms + future_buffer_ms
            for exit_ms in wanted_times[key]
        ):
            paths[key].append(path)
    out: dict[tuple[str, str], tuple[list[dict[str, Any]], list[int]]] = {}
    for key, key_paths in paths.items():
        candles_by_ts: dict[int, dict[str, Any]] = {}
        for path in key_paths:
            data = tail_replay.hold_replay._read_json(path)
            if not isinstance(data, list):
                continue
            for candle in data:
                if not isinstance(candle, dict) or candle.get("t") is None:
                    continue
                candles_by_ts[int(candle["t"])] = candle
        timestamps = sorted(candles_by_ts)
        out[key] = ([candles_by_ts[ts] for ts in timestamps], timestamps)
    return out


def _policy_metrics(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    usable = [row for row in rows if row.get(f"{name}_pnl_pct") is not None]
    baseline = [_num(row.get("pnl_pct")) for row in usable]
    policy = [_num(row.get(f"{name}_pnl_pct")) for row in usable]
    deltas = [_num(row.get(f"{name}_delta_pct")) for row in usable]
    base_vals = [value for value in baseline if value is not None]
    policy_vals = [value for value in policy if value is not None]
    delta_vals = [value for value in deltas if value is not None]
    selected_deltas = [
        _num(row.get(f"{name}_delta_pct"))
        for row in usable
        if row.get("selected_by_train_threshold")
    ]
    selected_vals = [value for value in selected_deltas if value is not None]
    worse = [value for value in selected_vals if value < 0.0]
    return {
        "population_n": len(rows),
        "test_n": len(usable),
        "action_missing_n": len(rows) - len(usable),
        "selected_action_missing_n": sum(
            1
            for row in rows
            if row.get("selected_by_train_threshold") and row.get(f"{name}_pnl_pct") is None
        ),
        "selected_n": len(selected_vals),
        "selected_rate_pct": round(len(selected_vals) / len(usable) * 100.0, 2) if usable else 0.0,
        "baseline_avg_pnl_pct": _avg(base_vals),
        "baseline_median_pnl_pct": _median(base_vals),
        "baseline_win_rate_pct": _win_rate(base_vals),
        "policy_avg_pnl_pct": _avg(policy_vals),
        "policy_median_pnl_pct": _median(policy_vals),
        "policy_win_rate_pct": _win_rate(policy_vals),
        "overall_avg_delta_pct": _avg(delta_vals),
        "overall_median_delta_pct": _median(delta_vals),
        "overall_total_delta_pct": round(sum(delta_vals), 4) if delta_vals else None,
        "selected_avg_delta_pct": _avg(selected_vals),
        "selected_median_delta_pct": _median(selected_vals),
        "selected_p10_delta_pct": _p10(selected_vals),
        "selected_worse_rate_pct": round(len(worse) / len(selected_vals) * 100.0, 2) if selected_vals else 0.0,
    }


def _passes_gate(metrics: dict[str, Any]) -> bool:
    selected_avg = _num(metrics.get("selected_avg_delta_pct"))
    selected_median = _num(metrics.get("selected_median_delta_pct"))
    selected_worse = _num(metrics.get("selected_worse_rate_pct"))
    selected_p10 = _num(metrics.get("selected_p10_delta_pct"))
    overall_avg = _num(metrics.get("overall_avg_delta_pct"))
    policy_median = _num(metrics.get("policy_median_pnl_pct"))
    baseline_median = _num(metrics.get("baseline_median_pnl_pct"))
    policy_win = _num(metrics.get("policy_win_rate_pct"))
    baseline_win = _num(metrics.get("baseline_win_rate_pct"))
    return bool(
        int(metrics.get("action_missing_n") or 0) == 0
        and int(metrics.get("selected_action_missing_n") or 0) == 0
        and int(metrics.get("test_n") or 0) >= 50
        and int(metrics.get("selected_n") or 0) >= 20
        and selected_avg is not None and selected_avg > 0.10
        and selected_median is not None and selected_median >= 0.0
        and selected_worse is not None and selected_worse <= 35.0
        and selected_p10 is not None and selected_p10 >= -0.75
        and overall_avg is not None and overall_avg > 0.02
        and policy_median is not None and baseline_median is not None and policy_median >= baseline_median
        and policy_win is not None and baseline_win is not None and policy_win >= baseline_win
    )


def _case_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("day"),
        row.get("sym"),
        row.get("tf") or "15m",
        row.get("source"),
        row.get("entry_ts"),
        row.get("exit_ts"),
        _num(row.get("pnl_pct")),
    )


def build_replay(
    *,
    reports_dir: Path = REPORTS,
    cache_dir: Path = CACHE_DIR,
    cfg: LearnedTailConfig = LearnedTailConfig(),
    output: Path = DEFAULT_OUTPUT,
    text_output: Path = DEFAULT_TEXT_OUTPUT,
    save: bool = True,
) -> dict[str, Any]:
    train, test, split_status = _prepare_rows(reports_dir, cache_dir, cfg)
    threshold = _quantile((row["risk_score"] for row in train), cfg.train_risk_quantile)
    ready = [row for row in test if row.get("tail_path_status") == "ready"]
    for row in ready:
        row["selected_by_train_threshold"] = threshold is not None and float(row.get("risk_score") or 0.0) >= threshold

    policies: dict[str, Any] = {}
    for policy in cfg.policies:
        selected = [row for row in ready if row.get("selected_by_train_threshold")]
        tail_replay._apply_tail_policy(selected, policy)
        for row in ready:
            if not row.get("selected_by_train_threshold"):
                row[f"{policy.name}_pnl_pct"] = row.get("pnl_pct")
                row[f"{policy.name}_delta_pct"] = 0.0
        metrics = _policy_metrics(ready, policy.name)
        metrics["passes_gate"] = _passes_gate(metrics)
        policies[policy.name] = metrics

    passing = [name for name, metrics in policies.items() if metrics.get("passes_gate")]
    decision = (
        f"advance_{passing[0]}_to_portfolio_replay_not_production"
        if passing
        else "terminal_reject_registered_learned_tail_family"
    )
    if split_status != "chronological" or not ready:
        decision = "insufficient_chronological_candle_evidence"
    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "status": "research_only",
        "production_effect": "none",
        "config": {
            "days": cfg.days,
            "continuation_margin_pct": cfg.continuation_margin_pct,
            "min_train_days": cfg.min_train_days,
            "train_risk_quantile": cfg.train_risk_quantile,
            "policies": [policy.__dict__ for policy in cfg.policies],
        },
        "coverage": {
            "split_status": split_status,
            "train_cases": len(train),
            "test_cases": len(test),
            "train_days": len({row.get("day") for row in train}),
            "test_days": len({row.get("day") for row in test}),
            "test_path_ready": len(ready),
            "test_path_missing": len(test) - len(ready),
            "data_through_day": max((str(row.get("day")) for row in train + test), default=None),
        },
        "selector": {
            "threshold_source": "train_only",
            "risk_threshold": threshold,
            "selected_test": sum(1 for row in ready if row.get("selected_by_train_threshold")),
            "selected_test_rate_pct": round(sum(1 for row in ready if row.get("selected_by_train_threshold")) / len(ready) * 100.0, 2) if ready else 0.0,
        },
        "policies": policies,
        "decision": decision,
        "recommendation": "Do not change live SELL or cooldown. Advance only a passing candidate to portfolio replay.",
    }
    if save:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        text_output.write_text(render_text(payload), encoding="utf-8")
        payload["files"] = {"json": str(output), "txt": str(text_output)}
    return payload


def render_text(report: dict[str, Any]) -> str:
    coverage = report.get("coverage") or {}
    selector = report.get("selector") or {}
    lines = [
        "Learned exit-tail policy replay (research-only)",
        f"decision: {report.get('decision')}",
        f"coverage: train={coverage.get('train_cases')} test={coverage.get('test_cases')} path_ready={coverage.get('test_path_ready')} through={coverage.get('data_through_day')}",
        f"selector: train_q80_threshold={selector.get('risk_threshold')} selected={selector.get('selected_test')} ({selector.get('selected_test_rate_pct')}%)",
    ]
    for name, metrics in (report.get("policies") or {}).items():
        lines.append(
            f"{name}: selected={metrics.get('selected_n')} avg_delta={metrics.get('selected_avg_delta_pct')} "
            f"median_delta={metrics.get('selected_median_delta_pct')} p10={metrics.get('selected_p10_delta_pct')} "
            f"worse={metrics.get('selected_worse_rate_pct')}% overall_avg_delta={metrics.get('overall_avg_delta_pct')} "
            f"gate={metrics.get('passes_gate')}"
        )
    return "\n".join(lines) + "\n"


def _num(value: Any) -> float | None:
    return discriminator._num(value)


def _avg(values: Iterable[float]) -> float | None:
    vals = [float(value) for value in values if value is not None]
    return round(mean(vals), 4) if vals else None


def _median(values: Iterable[float]) -> float | None:
    vals = [float(value) for value in values if value is not None]
    return round(median(vals), 4) if vals else None


def _win_rate(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if value is not None]
    return round(sum(value > 0.0 for value in vals) / len(vals) * 100.0, 2) if vals else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Maximum-period learned exit-tail policy replay")
    parser.add_argument("--days", type=int, default=0)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--text-output", type=Path, default=DEFAULT_TEXT_OUTPUT)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = build_replay(
        reports_dir=args.reports_dir,
        cache_dir=args.cache_dir,
        cfg=LearnedTailConfig(days=args.days),
        output=args.output,
        text_output=args.text_output,
        save=not args.no_save,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
