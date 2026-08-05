from __future__ import annotations

import argparse
import bisect
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from statistics import mean, median
from typing import Any, Callable, Iterable

import numpy as np

from indicators import compute_features
from strategy import check_exit_conditions


ROOT = Path(__file__).resolve().parent.parent
EVENTS = ROOT / "files" / "bot_events.jsonl"
CACHE = ROOT / ".runtime" / "signal_quality_cache"
REPORTS = ROOT / ".runtime" / "reports"
DEFAULT_OUTPUT = REPORTS / "early_rsi_weak_exit_causal_replay_latest.json"
DEFAULT_TEXT_OUTPUT = REPORTS / "early_rsi_weak_exit_causal_replay_latest.txt"
BAR_MS = {"15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000}


@dataclass(frozen=True)
class ReplayConfig:
    horizon_bars: int = 10
    warmup_bars: int = 240
    policy_change_cost_bps: float = 5.0
    profit_floor_pct: float = 0.05
    max_decision_price_error_pct: float = 0.25


@dataclass(frozen=True)
class PolicySpec:
    name: str
    kind: str
    scope: str = "all"
    value: float = 0.0
    tail_fraction: float = 1.0


@dataclass
class CasePath:
    event: dict[str, Any]
    candles: np.ndarray
    feat: dict[str, np.ndarray]
    decision_idx: int
    context_1h: dict[str, float | bool] | None = None


POLICIES = (
    PolicySpec("tighten_all_k090", "tighten", "all", 0.9),
    PolicySpec("tighten_all_k120", "tighten", "all", 1.2),
    PolicySpec("tighten_all_k140", "tighten", "all", 1.4),
    PolicySpec("tighten_retest15_k090", "tighten", "retest15", 0.9),
    PolicySpec("tighten_retest15_k120", "tighten", "retest15", 1.2),
    PolicySpec("tighten_retest15_k140", "tighten", "retest15", 1.4),
    PolicySpec("grace_retest15_3", "grace", "retest15", 3.0),
    PolicySpec("grace_retest15_4", "grace", "retest15", 4.0),
    PolicySpec("grace_retest15_5", "grace", "retest15", 5.0),
    PolicySpec("confirm2_retest15", "confirm2", "retest15", 2.0),
    PolicySpec("structure_retest15_k090", "structure", "retest15", 0.9),
    PolicySpec("structure_retest15_k120", "structure", "retest15", 1.2),
    PolicySpec("structure_retest15_k140", "structure", "retest15", 1.4),
    PolicySpec("mtf_retest15_k090", "mtf", "retest15", 0.9),
    PolicySpec("mtf_retest15_k120", "mtf", "retest15", 1.2),
    PolicySpec("mtf_retest15_k140", "mtf", "retest15", 1.4),
    PolicySpec("partial25_mtf_retest15_k140", "partial_mtf", "retest15", 1.4, 0.25),
    PolicySpec("partial50_mtf_retest15_k140", "partial_mtf", "retest15", 1.4, 0.50),
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
    events = load_exit_events(events_path)
    cache_index = build_cache_index(cache_dir)
    labeled: list[dict[str, Any]] = []
    missing = Counter()
    processing_events = sorted(
        events,
        key=lambda event: (
            str(event.get("sym") or ""),
            str(event.get("tf") or ""),
            str(event.get("ts") or "")[:10],
            str(event.get("ts") or ""),
        ),
    )
    for event in processing_events:
        path, reason = build_case_path(event, cache_index, cfg)
        if path is None:
            missing[reason] += 1
            continue
        row = _base_row(path)
        for policy in POLICIES:
            outcome = simulate_policy(path, policy, cfg)
            row[policy.name] = outcome
        labeled.append(row)

    labeled.sort(key=lambda row: str(row.get("exit_ts") or ""))

    split_days = chronological_day_splits([str(row["day"]) for row in labeled])
    for row in labeled:
        row["split"] = split_days.get(str(row["day"]), "holdout")

    policies = {
        policy.name: _policy_report(labeled, policy, cfg)
        for policy in POLICIES
    }
    passing = [
        (name, data)
        for name, data in policies.items()
        if data.get("portfolio_replay_gate") == "pass"
    ]
    passing.sort(
        key=lambda item: (
            _num(((item[1].get("splits") or {}).get("validation") or {}).get("avg_net_delta_pct")) or -999.0,
            _num(((item[1].get("splits") or {}).get("holdout") or {}).get("avg_net_delta_pct")) or -999.0,
        ),
        reverse=True,
    )
    decision = (
        f"advance_{passing[0][0]}_to_max_period_portfolio_replay"
        if passing
        else "reject_tested_policies_keep_production_sell_unchanged"
    )
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "builder": "early_rsi_weak_exit_causal_replay_v1",
        "status": "research_only",
        "decision": decision,
        "coverage": {
            "events_total": len(events),
            "labeled": len(labeled),
            "missing": len(events) - len(labeled),
            "missing_reasons": dict(missing),
            "first_event": events[0].get("ts") if events else None,
            "last_event": events[-1].get("ts") if events else None,
            "labeled_first": labeled[0].get("exit_ts") if labeled else None,
            "labeled_last": labeled[-1].get("exit_ts") if labeled else None,
            "split_days": dict(Counter(split_days.values())),
        },
        "config": {
            "horizon_bars": cfg.horizon_bars,
            "warmup_bars": cfg.warmup_bars,
            "policy_change_cost_bps": cfg.policy_change_cost_bps,
            "profit_floor_pct": cfg.profit_floor_pct,
            "max_decision_price_error_pct": cfg.max_decision_price_error_pct,
            "policies": [policy.__dict__ for policy in POLICIES],
        },
        "baseline": _baseline_report(labeled),
        "policies": policies,
        "case_rows": labeled,
        "top_improvements": _top_cases(labeled, passing[0][0] if passing else None, best=True),
        "top_harms": _top_cases(labeled, passing[0][0] if passing else None, best=False),
    }
    if save:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        text_output.write_text(render_text(report), encoding="utf-8")
        report["files"] = {"json": str(output), "txt": str(text_output)}
    return report


def load_exit_events(path: Path) -> list[dict[str, Any]]:
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
            reason = str(row.get("reason") or "").lower()
            if row.get("event") != "exit" or not ("rsi" in reason or "диверг" in reason):
                continue
            if str(row.get("tf")) not in BAR_MS:
                continue
            if (_num(row.get("entry_price")) or 0.0) <= 0 or (_num(row.get("exit_price")) or 0.0) <= 0:
                continue
            key = (
                row.get("sym"), row.get("tf"), row.get("ts"), row.get("entry_price"),
                row.get("exit_price"), row.get("bars_held"),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    rows.sort(key=lambda row: str(row.get("ts") or ""))
    return rows


def build_cache_index(cache_dir: Path) -> dict[tuple[str, str], list[tuple[int, int, Path]]]:
    out: dict[tuple[str, str], list[tuple[int, int, Path]]] = defaultdict(list)
    pattern = re.compile(r"^(.+?)_(15m|1h)_(\d+)_(\d+)\.json$")
    for path in cache_dir.glob("*.json"):
        match = pattern.match(path.name)
        if not match:
            continue
        sym, tf, start, end = match.groups()
        out[(sym, tf)].append((int(start), int(end), path))
    for rows in out.values():
        rows.sort(key=lambda item: (item[0], item[1], item[2].name))
    return dict(out)


def build_case_path(
    event: dict[str, Any],
    cache_index: dict[tuple[str, str], list[tuple[int, int, Path]]],
    cfg: ReplayConfig,
) -> tuple[CasePath | None, str]:
    tf = str(event.get("tf"))
    bar_ms = BAR_MS[tf]
    exit_ms = _parse_ts_ms(event.get("ts"))
    if exit_ms is None:
        return None, "bad_exit_ts"
    fragments = cache_index.get((str(event.get("sym") or ""), tf), [])
    if not fragments:
        return None, "no_cache_key"
    day_start = exit_ms - exit_ms % (24 * 60 * 60 * 1000)
    wanted_start = day_start - (cfg.warmup_bars + 5) * bar_ms
    wanted_end = day_start + 24 * 60 * 60 * 1000 + (cfg.horizon_bars + 2) * bar_ms
    paths = tuple(
        item[2]
        for item in fragments
        if item[1] >= wanted_start and item[0] <= wanted_end
    )
    if not paths:
        return None, "no_overlapping_cache"
    candles, feat = _pack_for_paths(tuple(str(path) for path in paths))
    if len(candles) == 0:
        return None, "empty_cache"
    timestamps = [int(value) for value in candles["t"]]
    decision_idx = bisect.bisect_right(timestamps, exit_ms - bar_ms) - 1
    if decision_idx < 0:
        return None, "decision_candle_missing"
    if decision_idx + cfg.horizon_bars >= len(candles):
        return None, "future_candles_missing"
    if decision_idx < 60:
        return None, "indicator_warmup_missing"
    required = ("atr", "ema_fast", "ema_slow", "adx", "slope", "macd_hist")
    if any(not np.isfinite(float(feat[name][decision_idx])) for name in required):
        return None, "decision_features_missing"
    decision_close = float(candles["c"][decision_idx])
    exit_price = float(event["exit_price"])
    price_error_pct = abs((decision_close / exit_price - 1.0) * 100.0)
    if price_error_pct > cfg.max_decision_price_error_pct:
        return None, "decision_price_mismatch"
    context_1h = _build_1h_context(event, exit_ms, cache_index, cfg) if tf == "15m" else None
    return CasePath(
        event=event,
        candles=candles,
        feat=feat,
        decision_idx=decision_idx,
        context_1h=context_1h,
    ), "ok"


def _build_1h_context(
    event: dict[str, Any],
    exit_ms: int,
    cache_index: dict[tuple[str, str], list[tuple[int, int, Path]]],
    cfg: ReplayConfig,
) -> dict[str, float | bool] | None:
    fragments = cache_index.get((str(event.get("sym") or ""), "1h"), [])
    if not fragments:
        return None
    bar_ms = BAR_MS["1h"]
    day_start = exit_ms - exit_ms % (24 * 60 * 60 * 1000)
    wanted_start = day_start - (cfg.warmup_bars + 5) * bar_ms
    wanted_end = day_start + 2 * 24 * 60 * 60 * 1000
    paths = tuple(
        item[2]
        for item in fragments
        if item[1] >= wanted_start and item[0] <= wanted_end
    )
    if not paths:
        return None
    candles, feat = _pack_for_paths(tuple(str(path) for path in paths))
    if len(candles) == 0:
        return None
    idx = bisect.bisect_right([int(value) for value in candles["t"]], exit_ms - bar_ms) - 1
    if idx < 0:
        return None
    values = {
        "close": _finite(candles["c"][idx]),
        "ema20": _finite(feat["ema_fast"][idx]),
        "ema50": _finite(feat["ema_slow"][idx]),
        "adx": _finite(feat["adx"][idx]),
        "slope": _finite(feat["slope"][idx]),
        "macd_hist": _finite(feat["macd_hist"][idx]),
    }
    intact = all(value is not None for value in values.values()) and bool(
        float(values["close"]) >= float(values["ema20"])
        and float(values["ema20"]) >= float(values["ema50"])
        and float(values["adx"]) >= 22.0
        and float(values["slope"]) >= 0.0
        and float(values["macd_hist"]) > 0.0
    )
    return {**values, "intact": intact}  # type: ignore[dict-item]


@lru_cache(maxsize=2048)
def _read_candle_path(path_text: str) -> tuple[tuple[float, ...], ...]:
    try:
        data = json.loads(Path(path_text).read_text(encoding="utf-8-sig"))
    except Exception:
        return ()
    out = []
    for row in data if isinstance(data, list) else []:
        if not isinstance(row, dict) or row.get("t") is None:
            continue
        values = tuple(_num(row.get(key)) for key in ("t", "o", "h", "l", "c", "v"))
        if any(value is None for value in values):
            continue
        out.append(tuple(float(value) for value in values))
    return tuple(out)


def _merge_candle_paths(paths: tuple[Path, ...]) -> list[dict[str, float]]:
    rows: dict[int, tuple[float, ...]] = {}
    for path in paths:
        for values in _read_candle_path(str(path)):
            rows[int(values[0])] = values
    return [
        {"t": values[0], "o": values[1], "h": values[2], "l": values[3], "c": values[4], "v": values[5]}
        for _, values in sorted(rows.items())
    ]


@lru_cache(maxsize=512)
def _pack_for_paths(path_texts: tuple[str, ...]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rows = _merge_candle_paths(tuple(Path(path) for path in path_texts))
    if not rows:
        return _to_numpy([]), {}
    candles = _to_numpy(rows)
    feat = compute_features(candles["o"], candles["h"], candles["l"], candles["c"], candles["v"])
    return candles, feat


def _to_numpy(rows: list[dict[str, float]]) -> np.ndarray:
    data = np.zeros(len(rows), dtype=[("t", "i8"), ("o", "f8"), ("h", "f8"), ("l", "f8"), ("c", "f8"), ("v", "f8")])
    for key in data.dtype.names or ():
        data[key] = np.asarray([row[key] for row in rows], dtype=data.dtype[key])
    return data


def simulate_policy(
    path: CasePath,
    policy: PolicySpec,
    cfg: ReplayConfig,
    *,
    reason_fn: Callable[..., str | None] = check_exit_conditions,
) -> dict[str, Any]:
    event = path.event
    baseline_price = float(event["exit_price"])
    entry_price = float(event["entry_price"])
    baseline_pnl = _event_pnl(event)
    if not _in_scope(event, policy.scope):
        return _unchanged_outcome(baseline_price, baseline_pnl, "out_of_scope")

    idx = path.decision_idx
    bars_held = max(0, int(event.get("bars_held") or 0))
    if policy.kind == "grace" and bars_held >= int(policy.value):
        return _unchanged_outcome(baseline_price, baseline_pnl, "grace_already_elapsed")
    if policy.kind == "structure" and not _structure_intact(path.feat, idx, float(path.candles["c"][idx])):
        return _unchanged_outcome(baseline_price, baseline_pnl, "structure_not_intact")
    if policy.kind in {"mtf", "partial_mtf"} and not bool((path.context_1h or {}).get("intact")):
        return _unchanged_outcome(baseline_price, baseline_pnl, "one_hour_context_not_intact")

    trail_k = float(event.get("trail_k") or 1.8)
    entry_idx = max(0, idx - bars_held)
    trail_stop = 0.0
    for bar in range(entry_idx + 1, idx + 1):
        atr = _finite(path.feat["atr"][bar])
        if atr is not None and atr > 0:
            trail_stop = max(trail_stop, float(path.candles["c"][bar]) - trail_k * atr)

    trail_kinds = {"tighten", "structure", "mtf", "partial_mtf"}
    tight_k = policy.value if policy.kind in trail_kinds else trail_k
    if policy.kind in trail_kinds:
        atr = _finite(path.feat["atr"][idx])
        if atr is not None and atr > 0:
            trail_stop = max(trail_stop, float(path.candles["c"][idx]) - tight_k * atr)
        if baseline_pnl >= cfg.profit_floor_pct:
            trail_stop = max(trail_stop, entry_price * (1.0 + cfg.profit_floor_pct / 100.0))

    weak_streak = 1
    max_favorable = max(float(path.candles["h"][bar]) for bar in range(entry_idx, idx + 1))
    min_after_exit = baseline_price
    exit_price = baseline_price
    exit_step = 0
    exit_reason = "baseline"
    for step in range(1, cfg.horizon_bars + 1):
        bar = idx + step
        close = float(path.candles["c"][bar])
        max_favorable = max(max_favorable, float(path.candles["h"][bar]))
        min_after_exit = min(min_after_exit, float(path.candles["l"][bar]))
        atr = _finite(path.feat["atr"][bar])
        effective_k = min(trail_k, tight_k) if policy.kind in trail_kinds else trail_k
        if atr is not None and atr > 0:
            trail_stop = max(trail_stop, close - effective_k * atr)
        if trail_stop > 0 and close < trail_stop:
            exit_price, exit_step, exit_reason = close, step, "atr_trail"
            break

        reason = reason_fn(
            path.feat,
            bar,
            path.candles["c"].astype(float),
            mode=str(event.get("mode") or "trend"),
            bars_elapsed=bars_held + step,
            tf=str(event.get("tf") or "15m"),
        )
        weak = _is_weak_reason(reason)
        if reason and not weak:
            exit_price, exit_step, exit_reason = close, step, "hard_exit"
            break
        if weak:
            weak_streak += 1
            if policy.kind == "grace" and bars_held + step >= int(policy.value):
                exit_price, exit_step, exit_reason = close, step, "weak_after_grace"
                break
            if policy.kind == "confirm2" and weak_streak >= 2:
                exit_price, exit_step, exit_reason = close, step, "weak_confirmed"
                break
            if policy.kind == "structure" and not _structure_intact(path.feat, bar, close):
                exit_price, exit_step, exit_reason = close, step, "structure_lost"
                break
        else:
            weak_streak = 0
    else:
        exit_step = cfg.horizon_bars
        exit_price = float(path.candles["c"][idx + cfg.horizon_bars])
        exit_reason = "horizon"

    tail_pnl = (exit_price / entry_price - 1.0) * 100.0
    policy_pnl = (
        (1.0 - policy.tail_fraction) * baseline_pnl + policy.tail_fraction * tail_pnl
        if policy.kind == "partial_mtf"
        else tail_pnl
    )
    delta = policy_pnl - baseline_pnl
    raw_mfe_pct = (max_favorable / entry_price - 1.0) * 100.0
    mfe_pct = (
        (1.0 - policy.tail_fraction) * baseline_pnl + policy.tail_fraction * raw_mfe_pct
        if policy.kind == "partial_mtf"
        else raw_mfe_pct
    )
    return {
        "applicable": True,
        "exit_price": round(exit_price, 10),
        "exit_step": exit_step,
        "exit_reason": exit_reason,
        "pnl_pct": round(policy_pnl, 4),
        "delta_pct": round(delta, 4),
        "net_delta_pct": round(delta - cfg.policy_change_cost_bps / 100.0 * policy.tail_fraction, 4),
        "post_exit_adverse_pct": round(
            (min_after_exit / baseline_price - 1.0) * 100.0 * policy.tail_fraction,
            4,
        ),
        "mfe_pct": round(mfe_pct, 4),
        "giveback_pct": round(max(0.0, mfe_pct - policy_pnl), 4),
        "exit_efficiency": round(policy_pnl / mfe_pct, 4) if mfe_pct > 0 else None,
    }


def _structure_intact(feat: dict[str, np.ndarray], idx: int, close: float) -> bool:
    ema20 = _finite(feat["ema_fast"][idx])
    ema50 = _finite(feat["ema_slow"][idx])
    adx = _finite(feat["adx"][idx])
    slope = _finite(feat["slope"][idx])
    macd = _finite(feat["macd_hist"][idx])
    return bool(
        ema20 is not None and ema50 is not None and adx is not None and slope is not None and macd is not None
        and close >= ema20 and ema20 >= ema50 and adx >= 24.0 and slope >= 0.10 and macd > 0.0
    )


def _in_scope(event: dict[str, Any], scope: str) -> bool:
    if scope == "all":
        return True
    return scope == "retest15" and event.get("tf") == "15m" and event.get("mode") == "retest"


def _is_weak_reason(reason: Any) -> bool:
    text = str(reason or "").lower()
    return "weak:" in text or "диверг" in text or "diverg" in text


def chronological_day_splits(days: Iterable[str]) -> dict[str, str]:
    unique = sorted(set(days))
    if not unique:
        return {}
    train_end = max(1, int(len(unique) * 0.60))
    validation_end = max(train_end + 1, int(len(unique) * 0.80)) if len(unique) > 2 else train_end
    validation_end = min(validation_end, len(unique))
    out = {}
    for i, day in enumerate(unique):
        out[day] = "train" if i < train_end else "validation" if i < validation_end else "holdout"
    return out


def _base_row(path: CasePath) -> dict[str, Any]:
    event = path.event
    baseline_pnl = _event_pnl(event)
    idx = path.decision_idx
    decision_close = float(path.candles["c"][idx])
    return {
        "day": str(event.get("ts"))[:10],
        "exit_ts": event.get("ts"),
        "sym": event.get("sym"),
        "tf": event.get("tf"),
        "mode": event.get("mode"),
        "bars_held": int(event.get("bars_held") or 0),
        "entry_price": _num(event.get("entry_price")),
        "exit_price": _num(event.get("exit_price")),
        "baseline_pnl_pct": round(baseline_pnl, 4),
        "decision_close": round(decision_close, 10),
        "decision_price_error_pct": round((decision_close / float(event["exit_price"]) - 1.0) * 100.0, 4),
        "structure_intact": _structure_intact(path.feat, idx, decision_close),
        "one_hour_context_intact": bool((path.context_1h or {}).get("intact")),
    }


def _event_pnl(event: dict[str, Any]) -> float:
    value = _num(event.get("pnl_pct"))
    if value is not None:
        return value
    return (float(event["exit_price"]) / float(event["entry_price"]) - 1.0) * 100.0


def _unchanged_outcome(price: float, pnl: float, reason: str) -> dict[str, Any]:
    return {
        "applicable": False,
        "exit_price": price,
        "exit_step": 0,
        "exit_reason": reason,
        "pnl_pct": round(pnl, 4),
        "delta_pct": 0.0,
        "net_delta_pct": 0.0,
        "post_exit_adverse_pct": 0.0,
        "mfe_pct": None,
        "giveback_pct": None,
        "exit_efficiency": None,
    }


def _baseline_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pnls = [_num(row.get("baseline_pnl_pct")) for row in rows]
    values = [value for value in pnls if value is not None]
    return {
        "n": len(values),
        "avg_pnl_pct": _avg(values),
        "median_pnl_pct": _median(values),
        "win_rate_pct": _rate(value > 0 for value in values),
        "by_mode": dict(Counter(str(row.get("mode") or "unknown") for row in rows)),
        "by_tf": dict(Counter(str(row.get("tf") or "unknown") for row in rows)),
        "bars_le_2": sum(1 for row in rows if int(row.get("bars_held") or 0) <= 2),
    }


def _policy_report(rows: list[dict[str, Any]], policy: PolicySpec, cfg: ReplayConfig) -> dict[str, Any]:
    splits = {
        split: _policy_metrics([row for row in rows if row.get("split") == split], policy.name)
        for split in ("train", "validation", "holdout")
    }
    slices = {
        "all": _policy_metrics(rows, policy.name),
        "retest15": _policy_metrics([row for row in rows if row.get("tf") == "15m" and row.get("mode") == "retest"], policy.name),
        "bars_le_2": _policy_metrics([row for row in rows if int(row.get("bars_held") or 0) <= 2], policy.name),
        "retest15_bars_le_2": _policy_metrics([
            row for row in rows
            if row.get("tf") == "15m" and row.get("mode") == "retest" and int(row.get("bars_held") or 0) <= 2
        ], policy.name),
    }
    validation = splits["validation"]
    holdout = splits["holdout"]
    gate = "pass" if all((
        int(validation.get("n") or 0) >= 10,
        int(holdout.get("n") or 0) >= 10,
        _metric_gt(validation, "avg_net_delta_pct", 0.0),
        _metric_gt(holdout, "avg_net_delta_pct", 0.0),
        _metric_ge(validation, "median_net_delta_pct", 0.0),
        _metric_ge(holdout, "median_net_delta_pct", 0.0),
        _metric_le(holdout, "worse_rate_pct", 45.0),
        _metric_ge(holdout, "p10_net_delta_pct", -1.0),
    )) else "fail"
    return {
        "spec": policy.__dict__,
        "splits": splits,
        "slices": slices,
        "portfolio_replay_gate": gate,
        "policy_change_cost_bps": cfg.policy_change_cost_bps,
    }


def _policy_metrics(rows: list[dict[str, Any]], name: str) -> dict[str, Any]:
    outcomes = [row.get(name) or {} for row in rows]
    outcomes = [outcome for outcome in outcomes if outcome.get("applicable")]
    deltas = [_num(outcome.get("delta_pct")) for outcome in outcomes]
    net = [_num(outcome.get("net_delta_pct")) for outcome in outcomes]
    pnls = [_num(outcome.get("pnl_pct")) for outcome in outcomes]
    adverse = [_num(outcome.get("post_exit_adverse_pct")) for outcome in outcomes]
    giveback = [_num(outcome.get("giveback_pct")) for outcome in outcomes]
    efficiency = [_num(outcome.get("exit_efficiency")) for outcome in outcomes]
    d = [value for value in deltas if value is not None]
    n = [value for value in net if value is not None]
    p = [value for value in pnls if value is not None]
    return {
        "n": len(n),
        "avg_delta_pct": _avg(d),
        "median_delta_pct": _median(d),
        "avg_net_delta_pct": _avg(n),
        "median_net_delta_pct": _median(n),
        "p10_net_delta_pct": _percentile(n, 10),
        "worse_rate_pct": _rate(value < 0 for value in n),
        "avg_pnl_pct": _avg(p),
        "median_pnl_pct": _median(p),
        "win_rate_pct": _rate(value > 0 for value in p),
        "median_adverse_pct": _median(value for value in adverse if value is not None),
        "median_giveback_pct": _median(value for value in giveback if value is not None),
        "median_exit_efficiency": _median(value for value in efficiency if value is not None),
    }


def _top_cases(rows: list[dict[str, Any]], policy_name: str | None, *, best: bool) -> list[dict[str, Any]]:
    if not policy_name:
        return []
    selected = []
    for row in rows:
        outcome = row.get(policy_name) or {}
        if not outcome.get("applicable"):
            continue
        selected.append({
            key: row.get(key)
            for key in ("day", "exit_ts", "sym", "tf", "mode", "bars_held", "baseline_pnl_pct", "structure_intact")
        } | {
            "policy": policy_name,
            "policy_pnl_pct": outcome.get("pnl_pct"),
            "net_delta_pct": outcome.get("net_delta_pct"),
            "exit_step": outcome.get("exit_step"),
            "exit_reason": outcome.get("exit_reason"),
        })
    selected.sort(key=lambda row: _num(row.get("net_delta_pct")) or 0.0, reverse=best)
    return selected[:20]


def _metric_gt(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _num(row.get(key))
    return value is not None and value > threshold


def _metric_ge(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _num(row.get(key))
    return value is not None and value >= threshold


def _metric_le(row: dict[str, Any], key: str, threshold: float) -> bool:
    value = _num(row.get(key))
    return value is not None and value <= threshold


def render_text(report: dict[str, Any]) -> str:
    coverage = report.get("coverage") or {}
    lines = [
        "Early RSI-WEAK exit causal replay (research-only)",
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
            f"val n={validation.get('n')} avg_net={validation.get('avg_net_delta_pct')} med_net={validation.get('median_net_delta_pct')} | "
            f"holdout n={holdout.get('n')} avg_net={holdout.get('avg_net_delta_pct')} "
            f"med_net={holdout.get('median_net_delta_pct')} worse={holdout.get('worse_rate_pct')}% p10={holdout.get('p10_net_delta_pct')}"
        )
    return "\n".join(lines) + "\n"


def _parse_ts_ms(value: Any) -> int | None:
    try:
        return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() * 1000)
    except Exception:
        return None


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
    parser = argparse.ArgumentParser(description="Causal replay for early RSI-WEAK exits")
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
