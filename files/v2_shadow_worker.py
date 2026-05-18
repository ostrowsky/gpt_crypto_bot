from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import aiohttp

import config
from indicators import compute_features
from strategy import fetch_klines
from v2.shadow_observer import (
    FeatureSnapshot,
    append_shadow_event,
    append_decision_trace,
    estimate_shadow_state,
    material_transition,
    telegram_eligible,
)


log = logging.getLogger("v2_shadow_worker")
ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / ".runtime"
EVENTS_FILE = ROOT / "v2_shadow_events.jsonl"
TRACE_FILE = ROOT / "v2_shadow_decisions.jsonl"
STATE_FILE = RUNTIME / "v2_shadow_state.json"
STATUS_FILE = RUNTIME / "v2_shadow_status.json"
CHAT_IDS_FILE = ROOT / ".chat_ids"
BAR_MS = {"15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_status(*, running: bool, last_cycle: dict | None = None, last_error: str = "") -> None:
    RUNTIME.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(
        json.dumps(
            {
                "worker": {
                    "running": running,
                    "last_heartbeat": _now(),
                    "last_error": last_error,
                },
                "last_cycle": last_cycle or {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _load_chat_ids() -> list[int]:
    if not CHAT_IDS_FILE.exists():
        return []
    try:
        payload = json.loads(CHAT_IDS_FILE.read_text(encoding="utf-8"))
        return [int(value) for value in payload] if isinstance(payload, list) else []
    except Exception:
        return []


async def _send_shadow_alert(session: aiohttp.ClientSession, event: dict) -> None:
    token = getattr(config, "TELEGRAM_BOT_TOKEN", "")
    if not token or not bool(getattr(config, "V2_SHADOW_REALTIME_TELEGRAM_ENABLED", False)):
        return
    chat_ids = _load_chat_ids()
    if not chat_ids:
        return
    text = (
        f"🧪 V2 SHADOW [{event['tf']}]\n"
        f"{event['sym']}: {event['previous_state'] or 'none'} → {event['state']}\n"
        f"action: {event['action']} | confidence: {event['confidence']:.2f}\n"
        f"reason: {event['reason']}"
    )
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for chat_id in chat_ids:
        try:
            async with session.post(url, json={"chat_id": chat_id, "text": text}, timeout=12) as resp:
                resp.raise_for_status()
                await resp.text()
        except Exception as exc:
            log.warning("v2 shadow telegram failed for %s: %s", chat_id, exc)


def _snapshot(data, feat, i: int) -> FeatureSnapshot:
    ema20_arr = feat.get("ema20", feat.get("ema_fast"))
    daily_range_arr = feat.get("daily_range", feat.get("daily_range_pct"))
    return FeatureSnapshot(
        price=float(data["c"][i]),
        ema20=float(ema20_arr[i]),
        slope=float(feat["slope"][i]),
        adx=float(feat["adx"][i]),
        rsi=float(feat["rsi"][i]),
        vol_x=float(feat["vol_x"][i]),
        daily_range=float(daily_range_arr[i]),
        macd_hist=float(feat["macd_hist"][i]),
    )


async def run_once() -> dict:
    state = _load_state()
    emitted = 0
    scanned = 0
    errors = 0
    stale = 0
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        for sym in config.load_watchlist():
            for tf in config.TIMEFRAMES:
                try:
                    data = await fetch_klines(session, sym, tf, limit=120)
                    if data is None or len(data) < 30:
                        continue
                    i = len(data["c"]) - 2
                    bar_ts = int(data["t"][i])
                    max_age_ms = 3 * BAR_MS.get(tf, 60 * 60 * 1000)
                    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
                    if now_ms - bar_ts > max_age_ms:
                        stale += 1
                        continue
                    feat = compute_features(data["o"], data["h"], data["l"], data["c"].astype(float), data["v"])
                    snapshot = _snapshot(data, feat, i)
                    decision = estimate_shadow_state(snapshot)
                    key = f"{sym}|{tf}"
                    previous = state.get(key)
                    bootstrap = previous is None
                    changed = material_transition(previous, decision)
                    event = {
                        "event": "v2_shadow_signal",
                        "source": "v2_shadow_observer",
                        "ts": _now(),
                        "sym": sym,
                        "tf": tf,
                        "bar_ts": bar_ts,
                        "previous_state": None if not previous else previous.get("state"),
                        "state": decision.state.value,
                        "action": decision.action,
                        "confidence": decision.confidence,
                        "reason": decision.reason,
                        "bootstrap": bootstrap,
                        "features": snapshot.__dict__,
                    }
                    trace_event = {
                        **event,
                        "observed_at": event["ts"],
                        "material_transition": changed,
                    }
                    if not previous or previous.get("bar_ts") != bar_ts:
                        append_decision_trace(TRACE_FILE, trace_event)
                    if changed:
                        append_shadow_event(EVENTS_FILE, event)
                        if telegram_eligible(previous, decision):
                            await _send_shadow_alert(session, event)
                        emitted += 1
                    state[key] = {
                        "state": decision.state.value,
                        "action": decision.action,
                        "bar_ts": bar_ts,
                        "updated_at": event["ts"],
                    }
                    scanned += 1
                except Exception as exc:
                    errors += 1
                    log.debug("shadow scan failed for %s %s: %s", sym, tf, exc)
    _save_state(state)
    cycle = {"scanned": scanned, "emitted": emitted, "stale": stale, "errors": errors, "finished_at": _now()}
    _save_status(running=True, last_cycle=cycle)
    return cycle


async def run_forever() -> None:
    while True:
        try:
            await run_once()
        except Exception as exc:
            _save_status(running=True, last_error=f"{type(exc).__name__}: {exc}")
        await asyncio.sleep(60)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO))
    if args.once:
        print(json.dumps(asyncio.run(run_once()), ensure_ascii=False))
    else:
        asyncio.run(run_forever())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
