from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_post_block_causal_discriminator import _feat, _load_rows, _split_by_day
from build_post_block_causal_discriminator_dataset import _first_bar_at_or_after, _load_history, _ts_ms


ROOT = Path(__file__).resolve().parent.parent
FILES = ROOT / "files"
DEFAULT_DATASET = ROOT / ".runtime" / "reports" / "post_block_causal_discriminator_dataset_15m.jsonl"
DEFAULT_HISTORY = FILES / ".runtime" / "v2_history"
DEFAULT_OUTPUT = ROOT / ".runtime" / "reports" / "post_block_delayed_confirmation_replay_15m.json"
CONFIRM_BARS = 8
FORWARD_BARS = {"60m": 4, "120m": 8, "240m": 16}


def build(dataset: Path = DEFAULT_DATASET, history_root: Path = DEFAULT_HISTORY, output: Path = DEFAULT_OUTPUT, train_fraction: float = 0.70) -> dict:
    rows = _load_rows(dataset)
    train, holdout = _split_by_day(rows, train_fraction)
    cache: dict[str, list[dict]] = {}
    selected_holdout = [row for row in holdout if _delayed_confirmation_rule(row)]
    baseline_holdout = holdout
    selected_trades = [_trade_from_row(row, history_root, cache) for row in selected_holdout]
    baseline_trades = [_trade_from_row(row, history_root, cache) for row in baseline_holdout]
    selected_trades = [trade for trade in selected_trades if trade]
    baseline_trades = [trade for trade in baseline_trades if trade]
    payload = {
        "dataset": str(dataset),
        "history_root": str(history_root),
        "rows": len(rows),
        "train_rows": len(train),
        "holdout_rows": len(holdout),
        "rule": {"rel_ret_120m_pct_gte": 2.0, "volume_x_120m_gte": 1.5, "entry_delay_bars": CONFIRM_BARS},
        "selected": _summary(selected_trades),
        "baseline_all_post_block_holdout": _summary(baseline_trades),
        "top_selected_examples": sorted(selected_trades, key=lambda item: item.get("ret_240m_pct", -999), reverse=True)[:20],
    }
    payload["decision"] = _decision(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _delayed_confirmation_rule(row: dict) -> bool:
    return _feat(row, "rel_ret_120m_pct") >= 2.0 and _feat(row, "volume_x_120m") >= 1.5


def _trade_from_row(row: dict, history_root: Path, cache: dict[str, list[dict]]) -> dict | None:
    symbol = str(row.get("symbol") or "")
    bars = _load_history(history_root, symbol, cache)
    if not bars:
        return None
    idx = _first_bar_at_or_after(bars, _ts_ms(row.get("candidate_ts")))
    if idx is None:
        return None
    entry_idx = idx + CONFIRM_BARS
    if entry_idx >= len(bars):
        return None
    entry = bars[entry_idx]
    entry_price = float(entry["close"])
    out = {
        "local_day": row.get("local_day"),
        "symbol": symbol,
        "candidate_ts": row.get("candidate_ts"),
        "entry_bar_ts_ms": entry.get("open_ts_ms"),
        "entry_price": entry_price,
        "label_top15": bool(row.get("label_top15")),
        "label_useful_missed_winner": bool(row.get("label_useful_missed_winner")),
        "rule_features": {"rel_ret_120m_pct": _feat(row, "rel_ret_120m_pct"), "volume_x_120m": _feat(row, "volume_x_120m")},
    }
    future = bars[entry_idx:]
    out["mfe_to_eod_pct"] = round(_ret_pct(max(float(bar["high"]) for bar in future), entry_price), 6) if future else None
    out["mae_to_eod_pct"] = round(_ret_pct(min(float(bar["low"]) for bar in future), entry_price), 6) if future else None
    out["eod_ret_pct"] = round(_ret_pct(float(future[-1]["close"]), entry_price), 6) if future else None
    for name, step in FORWARD_BARS.items():
        j = entry_idx + step
        if j < len(bars):
            window = bars[entry_idx : j + 1]
            out[f"ret_{name}_pct"] = round(_ret_pct(float(bars[j]["close"]), entry_price), 6)
            out[f"mfe_{name}_pct"] = round(_ret_pct(max(float(bar["high"]) for bar in window), entry_price), 6)
            out[f"mae_{name}_pct"] = round(_ret_pct(min(float(bar["low"]) for bar in window), entry_price), 6)
        else:
            out[f"ret_{name}_pct"] = None
            out[f"mfe_{name}_pct"] = None
            out[f"mae_{name}_pct"] = None
    return out


def _summary(trades: list[dict]) -> dict:
    n = len(trades)
    return {
        "count": n,
        "top15_count": sum(1 for trade in trades if trade.get("label_top15")),
        "top15_precision": _ratio(sum(1 for trade in trades if trade.get("label_top15")), n),
        "mean_ret_60m_pct": _mean_key(trades, "ret_60m_pct"),
        "mean_ret_120m_pct": _mean_key(trades, "ret_120m_pct"),
        "mean_ret_240m_pct": _mean_key(trades, "ret_240m_pct"),
        "mean_eod_ret_pct": _mean_key(trades, "eod_ret_pct"),
        "mean_mfe_to_eod_pct": _mean_key(trades, "mfe_to_eod_pct"),
        "mean_mae_to_eod_pct": _mean_key(trades, "mae_to_eod_pct"),
        "positive_120m_rate": _ratio(sum(1 for trade in trades if _num(trade.get("ret_120m_pct")) > 0), n),
        "positive_240m_rate": _ratio(sum(1 for trade in trades if _num(trade.get("ret_240m_pct")) > 0), n),
    }


def _decision(payload: dict) -> str:
    selected = payload["selected"]
    baseline = payload["baseline_all_post_block_holdout"]
    if selected["count"] < 10:
        return "research_only_rejected_low_support"
    if selected["mean_ret_240m_pct"] <= baseline["mean_ret_240m_pct"]:
        return "research_only_rejected_no_forward_edge_after_confirmation"
    if selected["mean_mfe_to_eod_pct"] <= abs(selected["mean_mae_to_eod_pct"]):
        return "research_only_rejected_poor_upside_vs_adverse_excursion"
    return "advance_to_fee_slippage_exit_replay"


def _mean_key(rows: list[dict], key: str) -> float:
    values = [_num(row.get(key)) for row in rows if row.get(key) is not None]
    return round(sum(values) / len(values), 6) if values else 0.0


def _ratio(a: int, b: int) -> float:
    return round(a / b, 6) if b else 0.0


def _num(value) -> float:
    try:
        value = float(value)
        return value if value == value else 0.0
    except Exception:
        return 0.0


def _ret_pct(value: float, base: float) -> float:
    return (value / base - 1.0) * 100.0 if base else 0.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.dataset, args.history_root, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else {"decision": payload["decision"], "selected": payload["selected"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
