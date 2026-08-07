from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np

import replay_early_rsi_weak_exit as base
from strategy import check_exit_conditions


ROOT = Path(__file__).resolve().parent.parent
EVENTS = ROOT / "files" / "bot_events.jsonl"
CACHE = ROOT / ".runtime" / "signal_quality_cache"
REPORTS = ROOT / ".runtime" / "reports"
DEFAULT_OUTPUT = REPORTS / "impulse_expansion_profit_tail_replay_latest.json"
DEFAULT_TEXT_OUTPUT = REPORTS / "impulse_expansion_profit_tail_replay_latest.txt"


@dataclass(frozen=True)
class ReplayConfig:
    horizon_bars: int = 10
    min_baseline_pnl_pct: float = 1.0
    policy_change_cost_bps: float = 5.0


@dataclass(frozen=True)
class TailPolicy:
    name: str
    min_score: int = 8
    tail_fraction: float = 0.25
    profit_lock_fraction: float = 0.90
    trail_atr_k: float = 1.8
    decay_score: int | None = 4
    decay_bars: int = 2


POLICIES = (
    TailPolicy("exp8_tail25_lock90_k18_decay4"),
    TailPolicy("exp7_tail25_lock90_k18_decay4", min_score=7),
    TailPolicy("exp9_tail25_lock90_k18_decay4", min_score=9),
    TailPolicy("exp8_tail25_lock90_k18_no_decay", decay_score=None),
    TailPolicy("exp8_tail50_lock90_k18_decay4", tail_fraction=0.50),
    TailPolicy("exp8_tail25_lock75_k18_decay4", profit_lock_fraction=0.75),
    TailPolicy("exp8_tail25_lock90_k14_decay4", trail_atr_k=1.4),
    TailPolicy("exp8_tail25_lock90_k22_decay4", trail_atr_k=2.2),
)


def build_replay(
    *,
    events_path: Path = EVENTS,
    cache_dir: Path = CACHE,
    cfg: ReplayConfig = ReplayConfig(),
    output: Path = DEFAULT_OUTPUT,
    text_output: Path = DEFAULT_TEXT_OUTPUT,
    save: bool = True,
) -> dict[str, Any]:
    events = load_events(events_path)
    cache_index = base.build_cache_index(cache_dir)
    path_cfg = base.ReplayConfig(
        horizon_bars=cfg.horizon_bars,
        warmup_bars=240,
        policy_change_cost_bps=cfg.policy_change_cost_bps,
        max_decision_price_error_pct=0.25,
    )
    missing = Counter()
    rows: list[dict[str, Any]] = []
    processing = sorted(
        events,
        key=lambda event: (
            str(event.get("sym") or ""),
            str(event.get("ts") or "")[:10],
            str(event.get("ts") or ""),
        ),
    )
    for event in processing:
        path, reason = base.build_case_path(event, cache_index, path_cfg)
        if path is None:
            missing[reason] += 1
            continue
        features = expansion_features(path, path.decision_idx)
        row = _base_row(path, features)
        for policy in POLICIES:
            row[policy.name] = simulate_tail(path, features, policy, cfg)
        rows.append(row)

    rows.sort(key=lambda row: str(row.get("exit_ts") or ""))
    split_map = base.chronological_day_splits([str(row["day"]) for row in rows])
    for row in rows:
        row["split"] = split_map.get(str(row["day"]), "holdout")

    policies = {policy.name: policy_report(rows, policy) for policy in POLICIES}
    passing = [name for name, payload in policies.items() if payload["portfolio_replay_gate"] == "pass"]
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "builder": "impulse_expansion_profit_tail_replay_v1",
        "status": "research_only",
        "decision": (
            f"advance_{passing[0]}_to_max_period_portfolio_replay"
            if passing else "reject_tested_expansion_tail_profiles"
        ),
        "coverage": {
            "events_total": len(events),
            "labeled": len(rows),
            "missing": len(events) - len(rows),
            "missing_reasons": dict(missing),
            "first_event": events[0].get("ts") if events else None,
            "last_event": events[-1].get("ts") if events else None,
            "labeled_first": rows[0].get("exit_ts") if rows else None,
            "labeled_last": rows[-1].get("exit_ts") if rows else None,
            "split_days": dict(Counter(split_map.values())),
        },
        "config": {
            "horizon_bars": cfg.horizon_bars,
            "min_baseline_pnl_pct": cfg.min_baseline_pnl_pct,
            "policy_change_cost_bps": cfg.policy_change_cost_bps,
            "policies": [policy.__dict__ for policy in POLICIES],
        },
        "baseline": baseline_report(rows),
        "score_distribution": dict(sorted(Counter(int(row["expansion_score"]) for row in rows).items())),
        "policies": policies,
        "case_rows": rows,
    }
    if save:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        text_output.write_text(render_text(report), encoding="utf-8")
        report["files"] = {"json": str(output), "txt": str(text_output)}
    return report


def load_events(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if '"event": "exit"' not in line or "WEAK:" not in line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get("event") != "exit" or row.get("tf") != "15m" or row.get("mode") != "impulse_speed":
                continue
            if (_num(row.get("entry_price")) or 0.0) <= 0 or (_num(row.get("exit_price")) or 0.0) <= 0:
                continue
            key = (
                row.get("sym"), row.get("ts"), row.get("entry_price"),
                row.get("exit_price"), row.get("bars_held"),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: str(row.get("ts") or ""))
    return rows


def expansion_features(path: base.CasePath, idx: int) -> dict[str, Any]:
    feat = path.feat
    data = path.candles
    atr = _finite(feat["atr"][idx]) or 0.0
    adx = _finite(feat["adx"][idx])
    slope = _finite(feat["slope"][idx])
    macd_hist = _finite(feat["macd_hist"][idx])
    ema20 = _finite(feat["ema_fast"][idx])
    ema50 = _finite(feat["ema_slow"][idx])
    close = float(data["c"][idx])
    adx_delta3 = _delta(feat["adx"], idx, 3)
    slope_delta3 = _delta(feat["slope"], idx, 3)
    values = {
        "adx": adx,
        "adx_delta3": adx_delta3,
        "slope_pct": slope,
        "slope_delta3": slope_delta3,
        "macd_atr": (macd_hist / atr) if macd_hist is not None and atr > 0 else None,
        "efficiency_ratio10": _efficiency_ratio(data["c"], idx, 10),
        "directional_ratio14": _directional_ratio(data["h"], data["l"], idx, 14),
        "donchian_position20": _donchian_position(data["h"], data["l"], close, idx, 20),
        "cmf20": _cmf(data, idx, 20),
        "ema_distance_atr": ((close - ema20) / atr) if ema20 is not None and atr > 0 else None,
        "ema20_above_ema50": bool(ema20 is not None and ema50 is not None and ema20 >= ema50),
    }
    components = {
        "adx_high": _ge(adx, 40.0),
        "adx_rising": _ge(adx_delta3, 0.0),
        "slope_high": _ge(slope, 0.50),
        "slope_rising": _ge(slope_delta3, 0.0),
        "macd_positive": _gt(values["macd_atr"], 0.0),
        "efficient_trend": _ge(values["efficiency_ratio10"], 0.45),
        "directional_dominance": _ge(values["directional_ratio14"], 1.8),
        "near_donchian_high": _ge(values["donchian_position20"], 0.80),
        "positive_money_flow": _ge(values["cmf20"], 0.0),
        "ema_extension": _ge(values["ema_distance_atr"], 0.75),
        "ema_alignment": bool(values["ema20_above_ema50"]),
    }
    return {**values, "components": components, "score": sum(1 for active in components.values() if active)}


def simulate_tail(
    path: base.CasePath,
    decision_features: dict[str, Any],
    policy: TailPolicy,
    cfg: ReplayConfig,
) -> dict[str, Any]:
    event = path.event
    baseline_pnl = _event_pnl(event)
    baseline_price = float(event["exit_price"])
    entry_price = float(event["entry_price"])
    score = int(decision_features.get("score") or 0)
    if baseline_pnl < cfg.min_baseline_pnl_pct:
        return _unchanged("pnl_below_floor", baseline_price, baseline_pnl)
    if score < policy.min_score:
        return _unchanged("score_below_floor", baseline_price, baseline_pnl)

    idx = path.decision_idx
    highest_close = baseline_price
    locked_pnl = baseline_pnl * policy.profit_lock_fraction
    trail_stop = entry_price * (1.0 + locked_pnl / 100.0)
    decay_streak = 0
    min_low = baseline_price
    exit_price = baseline_price
    exit_step = 0
    exit_reason = "baseline"
    for step in range(1, cfg.horizon_bars + 1):
        bar = idx + step
        close = float(path.candles["c"][bar])
        min_low = min(min_low, float(path.candles["l"][bar]))
        highest_close = max(highest_close, close)
        atr = _finite(path.feat["atr"][bar])
        if atr is not None and atr > 0:
            trail_stop = max(trail_stop, highest_close - policy.trail_atr_k * atr)
        if close < trail_stop:
            exit_price, exit_step, exit_reason = close, step, "protected_trail"
            break

        reason = check_exit_conditions(
            path.feat,
            bar,
            path.candles["c"].astype(float),
            mode="impulse_speed",
            bars_elapsed=int(event.get("bars_held") or 0) + step,
            tf="15m",
        )
        if reason and not base._is_weak_reason(reason):
            exit_price, exit_step, exit_reason = close, step, "hard_exit"
            break

        if policy.decay_score is not None:
            current_score = int(expansion_features(path, bar).get("score") or 0)
            decay_streak = decay_streak + 1 if current_score < policy.decay_score else 0
            if decay_streak >= policy.decay_bars:
                exit_price, exit_step, exit_reason = close, step, "expansion_decay"
                break
    else:
        exit_step = cfg.horizon_bars
        exit_price = float(path.candles["c"][idx + cfg.horizon_bars])
        exit_reason = "horizon"

    tail_pnl = (exit_price / entry_price - 1.0) * 100.0
    policy_pnl = (1.0 - policy.tail_fraction) * baseline_pnl + policy.tail_fraction * tail_pnl
    delta = policy_pnl - baseline_pnl
    net_delta = delta - cfg.policy_change_cost_bps / 100.0 * policy.tail_fraction
    return {
        "applicable": True,
        "exit_price": round(exit_price, 10),
        "exit_step": exit_step,
        "exit_reason": exit_reason,
        "tail_pnl_pct": round(tail_pnl, 4),
        "policy_pnl_pct": round(policy_pnl, 4),
        "delta_pct": round(delta, 4),
        "net_delta_pct": round(net_delta, 4),
        "post_exit_adverse_pct": round(
            (min_low / baseline_price - 1.0) * 100.0 * policy.tail_fraction,
            4,
        ),
        "profit_retention_pct": round(policy_pnl / baseline_pnl * 100.0, 2) if baseline_pnl > 0 else None,
    }


def policy_report(rows: list[dict[str, Any]], policy: TailPolicy) -> dict[str, Any]:
    splits = {
        split: policy_metrics([row for row in rows if row.get("split") == split], policy.name)
        for split in ("train", "validation", "holdout")
    }
    selected_rows = [row for row in rows if (row.get(policy.name) or {}).get("applicable")]
    by_reason = {
        name: policy_metrics(items, policy.name)
        for name, items in _groups(selected_rows, "reason_bucket").items()
    }
    by_symbol = Counter(str(row.get("sym")) for row in selected_rows)
    validation = splits["validation"]
    holdout = splits["holdout"]
    concentration_ok = not by_symbol or max(by_symbol.values()) / max(1, sum(by_symbol.values())) <= 0.25
    reason_positive = sum(1 for metrics in by_reason.values() if (_num(metrics.get("avg_net_delta_pct")) or 0.0) > 0)
    gate = "pass" if all((
        int(validation.get("n") or 0) >= 10,
        int(holdout.get("n") or 0) >= 10,
        _metric_gt(validation, "avg_net_delta_pct", 0.0),
        _metric_gt(holdout, "avg_net_delta_pct", 0.0),
        _metric_ge(validation, "median_net_delta_pct", 0.0),
        _metric_ge(holdout, "median_net_delta_pct", 0.0),
        _metric_le(holdout, "worse_rate_pct", 45.0),
        _metric_ge(holdout, "p10_net_delta_pct", -0.50),
        concentration_ok,
        reason_positive >= min(2, len(by_reason)),
    )) else "fail"
    return {
        "spec": policy.__dict__,
        "splits": splits,
        "all_selected": policy_metrics(rows, policy.name),
        "by_reason": by_reason,
        "top_symbols": dict(by_symbol.most_common(10)),
        "symbol_concentration_ok": concentration_ok,
        "portfolio_replay_gate": gate,
    }


def policy_metrics(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    outcomes = [row.get(name) or {} for row in rows]
    outcomes = [outcome for outcome in outcomes if outcome.get("applicable")]
    net = [_num(outcome.get("net_delta_pct")) for outcome in outcomes]
    delta = [_num(outcome.get("delta_pct")) for outcome in outcomes]
    pnl = [_num(outcome.get("policy_pnl_pct")) for outcome in outcomes]
    adverse = [_num(outcome.get("post_exit_adverse_pct")) for outcome in outcomes]
    retention = [_num(outcome.get("profit_retention_pct")) for outcome in outcomes]
    n = [value for value in net if value is not None]
    d = [value for value in delta if value is not None]
    p = [value for value in pnl if value is not None]
    return {
        "n": len(n),
        "avg_delta_pct": _avg(d),
        "median_delta_pct": _median(d),
        "avg_net_delta_pct": _avg(n),
        "median_net_delta_pct": _median(n),
        "p10_net_delta_pct": _percentile(n, 10),
        "worse_rate_pct": _rate(value < 0 for value in n),
        "avg_policy_pnl_pct": _avg(p),
        "median_policy_pnl_pct": _median(p),
        "win_rate_pct": _rate(value > 0 for value in p),
        "median_adverse_pct": _median(value for value in adverse if value is not None),
        "median_profit_retention_pct": _median(value for value in retention if value is not None),
    }


def baseline_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [_num(row.get("baseline_pnl_pct")) for row in rows]
    values = [value for value in pnls if value is not None]
    return {
        "n": len(values),
        "avg_pnl_pct": _avg(values),
        "median_pnl_pct": _median(values),
        "win_rate_pct": _rate(value > 0 for value in values),
        "pnl_ge_1_count": sum(1 for value in values if value >= 1.0),
        "by_reason": dict(Counter(str(row.get("reason_bucket")) for row in rows)),
    }


def _base_row(path: base.CasePath, features: dict[str, Any]) -> dict[str, Any]:
    event = path.event
    return {
        "day": str(event.get("ts"))[:10],
        "exit_ts": event.get("ts"),
        "sym": event.get("sym"),
        "tf": event.get("tf"),
        "mode": event.get("mode"),
        "bars_held": int(event.get("bars_held") or 0),
        "entry_price": _num(event.get("entry_price")),
        "exit_price": _num(event.get("exit_price")),
        "baseline_pnl_pct": round(_event_pnl(event), 4),
        "reason_bucket": _reason_bucket(event.get("reason")),
        "expansion_score": int(features.get("score") or 0),
        "expansion_features": features,
    }


def _unchanged(reason: str, price: float, pnl: float) -> dict[str, Any]:
    return {
        "applicable": False,
        "exit_price": price,
        "exit_step": 0,
        "exit_reason": reason,
        "tail_pnl_pct": pnl,
        "policy_pnl_pct": pnl,
        "delta_pct": 0.0,
        "net_delta_pct": 0.0,
        "post_exit_adverse_pct": 0.0,
        "profit_retention_pct": 100.0,
    }


def _reason_bucket(value: Any) -> str:
    text = str(value or "").lower()
    if "rsi" in text or "диверг" in text or "diverg" in text:
        return "rsi_divergence"
    if "объ" in text or "volume" in text:
        return "volume_exhaustion"
    if "fan" in text or "веер" in text:
        return "ema_fan_collapse"
    return "other_weak"


def _event_pnl(event: dict[str, Any]) -> float:
    value = _num(event.get("pnl_pct"))
    return value if value is not None else (float(event["exit_price"]) / float(event["entry_price"]) - 1.0) * 100.0


def _efficiency_ratio(close: np.ndarray, idx: int, lookback: int) -> float | None:
    if idx < lookback:
        return None
    values = np.asarray(close[idx - lookback: idx + 1], dtype=float)
    movement = float(np.sum(np.abs(np.diff(values))))
    return abs(float(values[-1] - values[0])) / movement if movement > 0 else 0.0


def _directional_ratio(high: np.ndarray, low: np.ndarray, idx: int, lookback: int) -> float | None:
    if idx < lookback:
        return None
    h = np.asarray(high[idx - lookback: idx + 1], dtype=float)
    l = np.asarray(low[idx - lookback: idx + 1], dtype=float)
    up = np.diff(h)
    down = -np.diff(l)
    plus = float(np.sum(np.where((up > down) & (up > 0), up, 0.0)))
    minus = float(np.sum(np.where((down > up) & (down > 0), down, 0.0)))
    if minus <= 1e-12:
        return 10.0 if plus > 0 else 1.0
    return min(10.0, plus / minus)


def _donchian_position(high: np.ndarray, low: np.ndarray, close: float, idx: int, lookback: int) -> float | None:
    if idx + 1 < lookback:
        return None
    highest = float(np.max(high[idx - lookback + 1: idx + 1]))
    lowest = float(np.min(low[idx - lookback + 1: idx + 1]))
    return (close - lowest) / (highest - lowest) if highest > lowest else 0.5


def _cmf(data: np.ndarray, idx: int, lookback: int) -> float | None:
    if idx + 1 < lookback:
        return None
    rows = data[idx - lookback + 1: idx + 1]
    span = rows["h"] - rows["l"]
    numerator = (rows["c"] - rows["l"]) - (rows["h"] - rows["c"])
    multiplier = np.zeros_like(span, dtype=float)
    np.divide(numerator, span, out=multiplier, where=span > 0)
    volume = float(np.sum(rows["v"]))
    return float(np.sum(multiplier * rows["v"]) / volume) if volume > 0 else 0.0


def _delta(values: np.ndarray, idx: int, bars: int) -> float | None:
    if idx < bars:
        return None
    current = _finite(values[idx])
    previous = _finite(values[idx - bars])
    return current - previous if current is not None and previous is not None else None


def _groups(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        out.setdefault(str(row.get(key) or "unknown"), []).append(row)
    return out


def render_text(report: dict[str, Any]) -> str:
    coverage = report.get("coverage") or {}
    lines = [
        "Impulse expansion profit-tail replay (research-only)",
        f"decision: {report.get('decision')}",
        f"coverage: events={coverage.get('events_total')} labeled={coverage.get('labeled')} missing={coverage.get('missing')}",
        f"period: {coverage.get('labeled_first')} -> {coverage.get('labeled_last')}",
        "",
        "Policies:",
    ]
    for name, payload in (report.get("policies") or {}).items():
        validation = ((payload.get("splits") or {}).get("validation") or {})
        holdout = ((payload.get("splits") or {}).get("holdout") or {})
        lines.append(
            f"  {name}: gate={payload.get('portfolio_replay_gate')} "
            f"val n={validation.get('n')} avg={validation.get('avg_net_delta_pct')} med={validation.get('median_net_delta_pct')} | "
            f"holdout n={holdout.get('n')} avg={holdout.get('avg_net_delta_pct')} med={holdout.get('median_net_delta_pct')} "
            f"worse={holdout.get('worse_rate_pct')}% p10={holdout.get('p10_net_delta_pct')}"
        )
    return "\n".join(lines) + "\n"


def _metric_gt(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _num(row.get(key))
    return value is not None and value > threshold


def _metric_ge(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _num(row.get(key))
    return value is not None and value >= threshold


def _metric_le(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _num(row.get(key))
    return value is not None and value <= threshold


def _ge(value: Any, threshold: float) -> bool:
    number = _num(value)
    return number is not None and number >= threshold


def _gt(value: Any, threshold: float) -> bool:
    number = _num(value)
    return number is not None and number > threshold


def _finite(value: Any) -> float | None:
    number = _num(value)
    return number if number is not None and math.isfinite(number) else None


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _avg(values: Iterable[float]) -> float | None:
    rows = [float(value) for value in values]
    return round(mean(rows), 4) if rows else None


def _median(values: Iterable[float]) -> float | None:
    rows = [float(value) for value in values]
    return round(median(rows), 4) if rows else None


def _percentile(values: Iterable[float], percentile: float) -> float | None:
    rows = [float(value) for value in values]
    return round(float(np.percentile(rows, percentile)), 4) if rows else None


def _rate(flags: Iterable[bool]) -> float:
    rows = list(flags)
    return round(sum(1 for flag in rows if flag) / len(rows) * 100.0, 2) if rows else 0.0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Causal replay for protected tails in active impulse expansion")
    parser.add_argument("--events", type=Path, default=EVENTS)
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--text-output", type=Path, default=DEFAULT_TEXT_OUTPUT)
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    report = build_replay(
        events_path=args.events,
        cache_dir=args.cache_dir,
        output=args.output,
        text_output=args.text_output,
        save=not args.no_save,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else render_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
