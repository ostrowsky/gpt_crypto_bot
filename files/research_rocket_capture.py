"""Offline, fail-closed rocket measurement and isolated rule ablations."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, median
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TZ = ZoneInfo("Europe/Budapest")
DAY = 86_400_000
ARMS = ("baseline", "chase_off", "score_off", "cluster_off", "cooldown_off",
        "rsi_exit_off", "weak_exit_off", "rsi_weak_exit_off", "combined_off")


def ms(value):
    return int(datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() * 1000)


def bounds(day):
    dt = datetime.fromisoformat(day).replace(tzinfo=TZ)
    return int(dt.timestamp()*1000), int((dt+timedelta(days=1)).timestamp()*1000)


def load_rockets(reports):
    rows, manifest = [], []
    for path in sorted(reports.glob("top_gainer_critic_*_final.json")):
        raw = path.read_bytes()
        d = json.loads(raw)
        day = d.get("target_day_local")
        if not day:
            continue
        manifest.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        seen = set()
        # Exchange top movers, not a hindsight-selected watchlist-only universe.
        for row in d.get("exchange_top_gainers", []):
            sym = row.get("symbol")
            if sym in seen or float(row.get("day_change_pct") or 0) < 10:
                continue
            seen.add(sym)
            rows.append(dict(row, day=day))
    return rows, manifest


def pair_events(events):
    """Fail closed on ambiguous chains. No first-entry/last-exit pairing."""
    active, invalid, pairs, problems, seen = {}, set(), [], [], set()
    for r in sorted(events, key=lambda r: ms(r["ts"])):
        key = (r["source_file"], r.get("sym", r.get("symbol")), r.get("tf"))
        sig = (key, r["event"], r["ts"], r.get("price"), r.get("exit_price"), r.get("entry_price"))
        if sig in seen:
            continue
        seen.add(sig)
        if r["event"] == "entry":
            if key in active:
                invalid.add(key)
                problems.append(dict(r, issue="overlapping_entry"))
            active[key] = r
            continue
        entry = active.pop(key, None)
        ambiguous = key in invalid
        invalid.discard(key)
        ep = float((entry or {}).get("price") or 0)
        stated = float(r.get("entry_price") or 0)
        xp = float(r.get("exit_price") or 0)
        if ambiguous or not all(math.isfinite(x) and x>0 for x in (ep,xp,stated)) or not np.isclose(ep, stated, rtol=1e-6, atol=1e-10):
            problems.append(dict(r, issue="unmatched_or_ambiguous_exit"))
            continue
        pairs.append({"symbol": key[1], "tf": key[2], "source": key[0],
                      "entry_ts": ms(entry["ts"]), "exit_ts": ms(r["ts"]),
                      "entry_price": ep, "exit_price": xp,
                      "gross_return_pct": (xp/ep-1)*100, "exit_reason": r.get("reason"),
                      "time_basis": "event_emission_not_exchange_fill"})
    problems.extend(dict(r, issue="open_at_snapshot") for r in active.values())
    return pairs, problems


def capture_day(rocket, pairs, problems=()):
    start, end = bounds(rocket["day"])
    selected = sorted([p for p in pairs if p["symbol"] == rocket["symbol"]
                       and p["entry_ts"] < end and p["exit_ts"] >= start], key=lambda p: p["entry_ts"])
    reason = None
    if any(p["entry_ts"] < start or p["exit_ts"] >= end for p in selected):
        reason = "cross_day_position"
    if any(b["entry_ts"] < a["exit_ts"] for a, b in zip(selected, selected[1:])):
        reason = "overlapping_positions"
    if any(r.get("sym", r.get("symbol")) == rocket["symbol"] and start <= ms(r["ts"]) < end for r in problems):
        reason = "ambiguous_event_chain"
    if not selected:
        reason = "no_matched_trade_unknown_coverage"
    move = float(rocket.get("day_close") or 0) - float(rocket.get("day_open") or 0)
    if not math.isfinite(move) or move <= 0:
        reason = "invalid_denominator"
    earned = sum(p["exit_price"]-p["entry_price"] for p in selected)
    return {"day": rocket["day"], "symbol": rocket["symbol"],
            "in_watchlist":rocket.get("in_watchlist"),
            "day_change_pct": rocket["day_change_pct"], "status": reason or "paired_same_day",
            "trade_count": len(selected), "price_gain_sum": earned if not reason else None,
            "day_price_move": move, "capture_pct": 100*earned/move if not reason else None,
            "trades": selected}


def summarize_capture(rows):
    values = [r["capture_pct"] for r in rows if r["capture_pct"] is not None]
    return {"rocket_days": len(rows), "measured_days": len(values), "unknown_days": len(rows)-len(values),
            "median_capture_pct": median(values) if values else None,
            "mean_capture_pct": mean(values) if values else None,
            "negative_days": sum(x < 0 for x in values), "statuses": dict(Counter(r["status"] for r in rows))}


def audit_actual(rockets, files):
    events, source_manifest, blockers = [], [], Counter()
    keys = {(r["day"], r["symbol"]) for r in rockets}
    for name in ("bot_events.jsonl", "agent_events.jsonl"):
        print(f"audit events: {name}", flush=True)
        path = files/name
        limit = path.stat().st_size
        digest = hashlib.sha256()
        read_bytes = 0
        invalid_rows = 0
        with path.open("rb") as handle:
            while read_bytes < limit:
                raw = handle.readline(limit-read_bytes)
                if not raw:
                    break
                read_bytes += len(raw)
                digest.update(raw)
                if not any(marker in raw for marker in (b'"event": "entry"', b'"event": "exit"', b'"event": "blocked"', b'"event":"entry"', b'"event":"exit"', b'"event":"blocked"')):
                    continue
                try:
                    r = json.loads(raw)
                    ms(r["ts"])
                except (ValueError, KeyError, TypeError):
                    invalid_rows += 1
                    continue
                if r.get("event") in ("entry", "exit"):
                    events.append(dict(r, source_file=name))
                if r.get("event") == "blocked":
                    day = datetime.fromtimestamp(ms(r["ts"])/1000, TZ).date().isoformat()
                    key = (day, r.get("sym", r.get("symbol")))
                    if key in keys:
                        blockers[(key[0], key[1], r.get("reason_code", "unknown"))] += 1
        source_manifest.append({"path": str(path), "snapshot_bytes": limit,"read_bytes":read_bytes,
                                "invalid_relevant_rows":invalid_rows,
                                "coverage":"complete_prefix" if read_bytes>=limit else "truncated_during_read",
                                "sha256": digest.hexdigest()})
    pairs, problems = pair_events(events)
    print(f"paired trades={len(pairs)} ambiguous/open={len(problems)} rocket_days={len(rockets)}",flush=True)
    by_symbol, problem_symbols = defaultdict(list), defaultdict(list)
    for p in pairs:
        by_symbol[p["symbol"]].append(p)
    for p in problems:
        problem_symbols[p.get("sym",p.get("symbol"))].append(p)
    rows = [capture_day(r, by_symbol[r["symbol"]], problem_symbols[r["symbol"]]) for r in rockets]
    return {"scope": "recorded_signals_not_fills; one-unit price-path capture, not portfolio return",
            "sources": source_manifest, "paired_trades_all": len(pairs),
            "pairing_problems": dict(Counter(r["issue"] for r in problems)),
            "thresholds": {str(t): summarize_capture([r for r in rows if r["day_change_pct"] >= t]) for t in (10, 15, 20)},
            "rows": rows, "blocker_coin_day_counts": dict(Counter(k[2] for k in blockers)),
            "blockers": [{"day": k[0], "symbol": k[1], "reason": k[2], "events": v} for k,v in blockers.items()]}


@contextmanager
def policy(arm):
    """Process-local overrides only; the running bot and config file are untouched."""
    import replay_backtest as rb
    import config
    import strategy
    if arm not in ARMS:
        raise ValueError(arm)
    both = arm == "combined_off"
    original_exit = rb.check_exit_conditions
    no_rsi = both or arm in ("rsi_exit_off", "rsi_weak_exit_off")
    no_weak = both or arm in ("weak_exit_off", "rsi_weak_exit_off")

    def exit_check(*args, **kwargs):
        # Evaluate later hard rules after suppressing RSI; do not mask them.
        with patch.object(strategy.config, "RSI_OVERBOUGHT", float("inf")) if no_rsi else ExitStack():
            reason = original_exit(*args, **kwargs)
        return None if no_weak and reason and "WEAK:" in reason else reason

    with ExitStack() as stack:
        for name, value in (("_ml_general_score_replay", None), ("_ml_trend_nonbull_score_replay", None),
                            ("_ml_candidate_ranker_components", {}), ("_ml_candidate_ranker_runtime_bonus", 0.0)):
            stack.enter_context(patch.object(rb, name, return_value=value))
        stack.enter_context(patch.object(config, "PORTFOLIO_REPLACE_RANKER_ENABLED", False, create=True))
        if both or arm == "chase_off":
            stack.enter_context(patch.object(rb, "_chase_guard_reason_for_replay_variant", return_value=None))
        if both or arm == "score_off":
            stack.enter_context(patch.object(rb, "_top_gainer_score_min_for_mode", return_value=-float("inf")))
        if both or arm == "cluster_off":
            stack.enter_context(patch.object(rb, "_signal_cluster_cap_replay", return_value=10))
        if both or arm == "cooldown_off":
            stack.enter_context(patch.object(rb, "_cooldown_bars_after_exit", return_value=0))
        stack.enter_context(patch.object(rb, "check_exit_conditions", side_effect=exit_check))
        yield


def blocks(start, end):
    out = []
    while start < end:
        stop = min(start+30*DAY, end)
        out.append((start, stop))
        start = stop
    return [(a,b,"discovery" if i < max(1, int(len(out)*.6)) else
             "validation" if i < max(2, int(len(out)*.8)) else "holdout")
            for i,(a,b) in enumerate(out)]


def complete_series(data, start, end, step):
    expected = np.arange((start+step-1)//step*step, end, step, dtype=np.int64)
    if data is None or len(expected) == 0:
        return False
    actual = data["t"][(data["t"] >= start) & (data["t"] < end)]
    return np.array_equal(actual, expected)


def strict_cached_series(index, sym, tf, start, end):
    """Reject conflicting cached OHLCV values rather than silently last-write-wins."""
    import replay_backtest as rb
    candles, conflicts, errors = {}, set(), 0
    for a,b,path in index.get((sym,tf),[]):
        if b < start or a >= end:
            continue
        try:
            payload=json.loads(path.read_bytes())
            if not isinstance(payload,list):
                raise ValueError("not candle list")
            for row in payload:
                t=int(row["t"])
                # A cache ending mid-bar may contain a mutable active candle.
                if not start <= t < end or t+rb.BAR_MS[tf] > b:
                    continue
                values=tuple(float(row[k]) for k in ("o","h","l","c","v"))
                o,h,l,c,v=values
                if not all(math.isfinite(x) for x in values) or min(o,h,l,c)<=0 or v<0 or h<max(o,c,l) or l>min(o,c,h):
                    conflicts.add(t)
                    continue
                if t in candles and candles[t] != values and not np.allclose(candles[t],values,rtol=1e-8,atol=1e-12):
                    conflicts.add(t)
                candles[t]=values
        except (ValueError,KeyError,TypeError,OSError):
            errors+=1
    if conflicts or errors or not candles:
        return None,{"conflicting_timestamps":len(conflicts),"read_errors":errors}
    result=np.zeros(len(candles),dtype=rb._KLINE_DTYPE)
    for i,(t,values) in enumerate(sorted(candles.items())):
        result[i]=tuple([t,*values])
    return result,{"conflicting_timestamps":0,"read_errors":0}


def post_entry_retention(trade, data, end, step=900_000):
    """No entry-candle high or pre-entry peak; incomplete bars yield unknown."""
    start = (trade["entry_ts"]+step-1)//step*step
    if not complete_series(data,start,end,step):
        return {"status":"unknown_incomplete_post_entry_candles","retention_pct":None}
    rows = data[(data["t"] >= start) & (data["t"]+step <= end)]
    if not len(rows):
        return {"status":"unknown_no_full_post_entry_candle","retention_pct":None}
    peak = max(float(np.max(rows["h"])),trade["entry_price"])
    potential = peak-trade["entry_price"]
    return {"status":"full_candles_only_excludes_entry_partial_bar", "peak_price":peak,
            "potential_price_move":potential,
            "retention_pct":100*(trade["exit_price"]-trade["entry_price"])/potential if potential>0 else None}


def enrich_actual(report, cache_dir):
    import replay_backtest as rb
    index=rb._build_market_cache_index(cache_dir)
    grouped=defaultdict(list)
    for r in report["actual"]["rows"]:
        if r["status"]=="paired_same_day":
            grouped[r["symbol"]].append(r)
    for sym,rows in grouped.items():
        for r in rows:
            start,end=bounds(r["day"])
            data,quality=strict_cached_series(index,sym,"15m",start,end)
            for trade in r["trades"]:
                trade["after_entry_retention"]=post_entry_retention(trade,data,bounds(r["day"])[1])
                trade["after_entry_retention"]["cache_quality"]=quality


def experiment_summary(report):
    result={}
    for split in ("discovery","validation","holdout"):
        groups=[b for b in report["blocks"] if b.get("split")==split and b.get("arms")]
        result[split]={}
        for arm in ARMS:
            rows=[b["arms"][arm] for b in groups if arm in b["arms"]]
            if not rows:
                continue
            returns=[r["portfolio"]["portfolio"]["net_return_after_costs_pct"] for r in rows]
            base=[b["arms"]["baseline"]["portfolio"]["portfolio"]["net_return_after_costs_pct"] for b in groups]
            captures=[r for a in rows for r in a["rocket_rows"]]
            result[split][arm]={"blocks":len(rows),"mean_block_net_return_pct":mean(returns),
                "paired_mean_block_delta_pp":mean(a-b for a,b in zip(returns,base)),
                "positive_delta_blocks":sum(a>b for a,b in zip(returns,base)),
                "worst_block_drawdown_pct":max(r["portfolio"]["portfolio"]["max_drawdown_after_costs_pct"] for r in rows),
                "trades":sum(r["trades"] for r in rows),"capture":summarize_capture(captures)}
    return result


def render_report(report):
    lines=["# Rocket capture research", "", f"Status: {report['status']}; production promotion: UNKNOWN.",
        "", "Actual metric: one-unit price-path capture on matched same-day signal trades, not fills or capital return.",
        "Unknown days are excluded, not treated as zero. Rockets are ex-post exchange top movers.",
        "", "| Day gain threshold | Rocket days | Measured | Unknown | Median capture |", "|---|---:|---:|---:|---:|"]
    for t,r in report["actual"]["thresholds"].items():
        lines.append(f"| {t}% | {r['rocket_days']} | {r['measured_days']} | {r['unknown_days']} | {r['median_capture_pct']} |")
    lines += ["", "## Rule-only backtests", "", "30-day reset blocks; 10 slots; fee 7.5 bps + slippage 5 bps per side.",
        "Learned scores disabled in all arms. Agent-only entry gates are NOT tested by this engine.",
        "", "| Split | Arm | Blocks | Mean block net % | Delta vs control, pp | Worst block drawdown % |", "|---|---|---:|---:|---:|---:|"]
    for split,arms in report.get("summary",{}).items():
        for arm,r in arms.items():
            lines.append(f"| {split} | {arm} | {r['blocks']} | {r['mean_block_net_return_pct']:.3f} | {r['paired_mean_block_delta_pp']:.3f} | {r['worst_block_drawdown_pct']:.3f} |")
    lines += ["", "## QNT: last complete week", ""]
    for r in report["actual"]["rows"]:
        if r["symbol"]=="QNTUSDT" and "2026-09-21"<=r["day"]<="2026-09-27":
            lines.append(f"- {r['day']}: {r['status']}; trades={r['trade_count']}; capture={r['capture_pct']}%.")
    lines += ["", "## Limits", ""]+[f"- {s}" for s in report["limitations"]]
    return "\n".join(lines)+"\n"


async def replay_block(start, end, symbols, index, cache_dir, rockets, arms):
    import replay_backtest as rb
    from portfolio_alpha import closed_price_series, evaluate_portfolio_alpha
    cache, missing = {}, []
    warm = start-10*DAY
    for sym in sorted(set(symbols) | {"BTCUSDT"}):
        for tf in ("15m", "1h"):
            fragments=index.get((sym,tf),[])
            if not fragments or min(a for a,_,_ in fragments)>warm or max(b for _,b,_ in fragments)<end:
                missing.append({"symbol":sym,"tf":tf,"metadata_range_missing":True})
                continue
            data,quality = strict_cached_series(index,sym,tf,warm,end)
            if not complete_series(data, warm, end, rb.BAR_MS[tf]):
                missing.append({"symbol":sym,"tf":tf,**quality})
                continue
            cache[sym,tf] = (data, rb.compute_features(data["o"],data["h"],data["l"],data["c"],data["v"]))
    eligible = [s for s in symbols if (s,"15m") in cache and (s,"1h") in cache]
    for sym in eligible:
        data = rb._aggregate_1h_to_4h(cache[sym,"1h"][0])
        cache[sym,"4h"] = (data,rb.compute_features(data["o"],data["h"],data["l"],data["c"],data["v"]))
    result = {"start": start, "end": end, "eligible_symbols": len(eligible), "requested_symbols":len(symbols),
              "missing_series": missing, "arms": {}}
    if not eligible or ("BTCUSDT","1h") not in cache or ("BTCUSDT","15m") not in cache:
        result["status"] = "unknown_no_complete_population_or_benchmark"
        return result
    c15 = {s:cache[s,"15m"] for s in eligible}
    c4 = {s:cache[s,"4h"] for s in eligible}
    ctx = rb._build_bull_day_context(cache["BTCUSDT","1h"][0])
    series = {s:closed_price_series(cache[s,"15m"][0],bar_ms=rb.BAR_MS["15m"],start_ms=start,end_ms=end) for s in eligible}
    btc = closed_price_series(cache["BTCUSDT","15m"][0],bar_ms=rb.BAR_MS["15m"],start_ms=start,end_ms=end)
    snapshots = {}
    for arm in arms:
        print(f"arm {arm} eligible={len(eligible)}", flush=True)
        with policy(arm):
            key = "chase_off" if arm in ("chase_off","combined_off") else "baseline"
            if key not in snapshots:
                raw, times, _ = await rb.build_replay_candidate_snapshot(eligible,["15m","1h"],cache,c15,c4,ctx,variant="score_replace_cluster")
                raw = {t:v for t,v in raw.items() if start <= t < end}
                snapshots[key] = (raw,{t for t in times if start <= t <= end},sum(map(len,raw.values())))
            # simulate_portfolio sorts lists in place; detach each arm's lists.
            raw, times, count = snapshots[key]
            trades, stats = await rb.simulate_portfolio(eligible,["15m","1h"],cache,c15,c4,ctx,
                max_open_positions=10,enable_replacement=True,replace_min_delta=8,
                variant="score_replace_cluster",top_gainer_score_min=34,
                candidate_snapshot=({t:list(v) for t,v in raw.items()},set(times),count))
        alpha = evaluate_portfolio_alpha(trades,price_series_by_symbol=series,benchmark_series=btc,
            window_start_ms=start,window_end_ms=end,requested_days=max(1,round((end-start)/DAY)),
            universe=eligible,variant=arm,fee_bps=7.5,slippage_bps=5,capacity=10)
        # The generic evaluator cannot certify historical universe/model parity.
        alpha["decision_grade"] = False
        alpha["evidence_grade"] = "diagnostic_rule_only_partial_population"
        pairs = [{"symbol":t.sym,"entry_ts":t.entry_ts,"exit_ts":t.exit_ts,"entry_price":t.entry_price,
                  "exit_price":t.exit_price,"gross_return_pct":t.pnl_pct,"exit_reason":t.exit_reason} for t in trades]
        labels = [r for r in rockets if start <= bounds(r["day"])[0] and bounds(r["day"])[1] <= end and r["symbol"] in eligible]
        captures = [capture_day(r,pairs) for r in labels]
        # Unlike logs, the complete simulation establishes no-position zeros.
        for r in captures:
            if r["status"] == "no_matched_trade_unknown_coverage":
                r.update(status="replay_no_trade",capture_pct=0.0,price_gain_sum=0.0)
        result["arms"][arm] = {"portfolio":alpha,"capture":summarize_capture(captures),"rocket_rows":captures,
            "negative_trades":sum(t.pnl_pct < .25 for t in trades),"trades":len(trades),
            "stats":{k:v for k,v in asdict(stats).items() if not isinstance(v,(list,dict))}}
    result["status"] = "diagnostic_only"
    return result


def write_report(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    temp.replace(path)


async def main_async(args):
    import replay_backtest as rb
    import config
    if args.enrich:
        result=json.loads(args.output.read_text(encoding="utf-8"))
        labels,_=load_rockets(ROOT/".runtime/reports")
        membership={(r["day"],r["symbol"]):r.get("in_watchlist") for r in labels}
        for row in result["actual"]["rows"]:
            row["in_watchlist"]=membership.get((row["day"],row["symbol"]))
        result["actual"]["watchlist_thresholds"]={str(t):summarize_capture([r for r in result["actual"]["rows"]
            if r["in_watchlist"] is True and r["day_change_pct"]>=t]) for t in (10,15,20)}
        enrich_actual(result,args.cache)
        result["summary"]=experiment_summary(result)
        write_report(args.output,result)
        args.output.with_suffix(".md").write_text(render_report(result),encoding="utf-8")
        return
    rockets, manifest = load_rockets(ROOT/".runtime/reports")
    if args.replay_only:
        result=json.loads(args.output.read_text(encoding="utf-8"))
        if result["report_manifest"] != manifest:
            raise ValueError("Report inputs changed since actual audit; use a new output and rerun")
        result["blocks"]=[]
        result["status"]="running"
    else:
        result = {"schema":1,"status":"running","promotion":"UNKNOWN_not_authorized_by_diagnostic",
              "limitations":["historical universe snapshots unavailable", "current rule-only baseline, learned scores disabled",
                  "agent-only gates not simulated", "calendar block resets", "retrospective holdout is not forward validation",
                  "local candle cache is not independently exchange-verified"],
                  "report_manifest":manifest,"actual":(
                      {"status":"not_requested_replay_only", "thresholds":{}, "rows":[]}
                      if args.skip_event_audit else audit_actual(rockets,ROOT/"files")),"blocks":[]}
    result["runner_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_report(args.output,result)
    if args.actual_only:
        result["status"]="actual_audit_complete_replay_not_run"
        write_report(args.output,result)
        return
    index = rb._build_market_cache_index(args.cache)
    dates = sorted(r["day"] for r in rockets)
    schedule = blocks(bounds(dates[0])[0],bounds(dates[-1])[1])
    result["schedule"] = schedule
    result["universe"] = config.load_watchlist()
    # Freeze file inventory; the live writer may create later files, ignored here.
    result["cache_manifest"] = [{"path":str(p),"bytes":p.stat().st_size,"mtime_ns":p.stat().st_mtime_ns}
                                for vv in index.values() for _,_,p in vv]
    for i,(start,end,split) in enumerate(schedule):
        print(f"block {i+1}/{len(schedule)} {datetime.fromtimestamp(start/1000,timezone.utc).isoformat()} {split}",flush=True)
        block = await replay_block(start,end,result["universe"],index,args.cache,rockets,args.arms)
        block["split"] = split
        result["blocks"].append(block)
        write_report(args.output,result)
    result["status"]="complete_diagnostic_only"
    result["summary"]=experiment_summary(result)
    write_report(args.output,result)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--actual-only",action="store_true")
    parser.add_argument("--skip-event-audit",action="store_true",help="Run replay comparison without repeating separate raw-log measurement; no actual capture claim")
    parser.add_argument("--replay-only",action="store_true",help="Reuse completed raw audit with matching report manifest; rerun all candle blocks")
    parser.add_argument("--enrich",action="store_true",help="Add full-candle post-entry retention and split summaries to completed output")
    parser.add_argument("--arms",nargs='+',choices=ARMS,default=list(ARMS),help="Registered arms; baseline required for paired comparison")
    parser.add_argument("--cache",type=Path,default=ROOT/".runtime/signal_quality_cache")
    parser.add_argument("--output",type=Path,default=ROOT/".runtime/reports/rocket_capture_ablation_latest.json")
    args = parser.parse_args()
    if 'baseline' not in args.arms:
        parser.error('--arms must include baseline')
    asyncio.run(main_async(args))
