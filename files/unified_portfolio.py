# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
MAIN_POSITIONS_FILE = ROOT / "positions.json"
AGENT_POSITIONS_FILE = ROOT / "agent_positions.json"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _prediction_score(predictions: Any) -> float:
    if not isinstance(predictions, dict):
        return 0.0
    score = 0.0
    for value in predictions.values():
        if value is True:
            score += 1.0
        elif value is False:
            score -= 1.0
    return score


def position_score(position: dict[str, Any], source: str) -> float:
    """Comparable score for main bot and market-agent positions."""
    prediction_bonus = _prediction_score(position.get("predictions")) * 4.0
    four_h_bonus = _safe_float(position.get("four_h_context_score")) * 1.5
    slope_bonus = max(0.0, _safe_float(position.get("entry_slope"))) * 4.0
    adx_bonus = max(0.0, _safe_float(position.get("entry_adx")) - 18.0) * 0.2
    vol_bonus = max(0.0, _safe_float(position.get("entry_vol_x")) - 1.0) * 2.0

    if source == "agent":
        return (
            _safe_float(position.get("leader_score")) * 2.0
            + _safe_float(position.get("forecast_return_pct")) * 4.0
            + max(0.0, _safe_float(position.get("today_change_pct"))) * 2.0
            + four_h_bonus
            + slope_bonus
            + adx_bonus
            + vol_bonus
            + prediction_bonus
        )

    return (
        _safe_float(position.get("candidate_score_at_entry"))
        + _safe_float(position.get("ranker_final_score")) * 6.0
        + _safe_float(position.get("ranker_top_gainer_prob")) * 10.0
        + _safe_float(position.get("ranker_ev")) * 2.0
        + _safe_float(position.get("forecast_return_pct")) * 18.0
        + max(0.0, _safe_float(position.get("today_change_pct"))) * 1.5
        + four_h_bonus
        + prediction_bonus
    )


def _load_json_dict(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_json_dict(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_main_positions_raw(path: Path = MAIN_POSITIONS_FILE) -> dict[str, dict[str, Any]]:
    return {
        str(key): value
        for key, value in _load_json_dict(path).items()
        if isinstance(value, dict)
    }


def load_agent_positions_raw(path: Path = AGENT_POSITIONS_FILE) -> dict[str, dict[str, Any]]:
    return {
        str(key): value
        for key, value in _load_json_dict(path).items()
        if isinstance(value, dict)
    }


def ranked_unified_positions(
    main_positions: dict[str, dict[str, Any]],
    agent_positions: dict[str, dict[str, Any]],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, position in main_positions.items():
        symbol = str(position.get("symbol") or key)
        rows.append(
            {
                "source": "main",
                "key": key,
                "symbol": symbol,
                "position": position,
                "score": position_score(position, "main"),
            }
        )
    for key, position in agent_positions.items():
        symbol = str(position.get("symbol") or str(key).split("|", 1)[0])
        rows.append(
            {
                "source": "agent",
                "key": key,
                "symbol": symbol,
                "position": position,
                "score": position_score(position, "agent"),
            }
        )

    best_by_symbol: dict[str, dict[str, Any]] = {}
    for row in rows:
        symbol = str(row["symbol"])
        current = best_by_symbol.get(symbol)
        if current is None or float(row["score"]) > float(current["score"]):
            best_by_symbol[symbol] = row

    ranked = sorted(
        best_by_symbol.values(),
        key=lambda row: (
            float(row["score"]),
            _safe_float(row["position"].get("today_change_pct")),
            _safe_float(row["position"].get("forecast_return_pct")),
            str(row["symbol"]),
        ),
        reverse=True,
    )
    if limit is not None and limit >= 0:
        return ranked[:limit]
    return ranked


def load_ranked_unified_positions(*, limit: int | None = None) -> list[dict[str, Any]]:
    return ranked_unified_positions(
        load_main_positions_raw(),
        load_agent_positions_raw(),
        limit=limit,
    )


def external_agent_symbol_count(main_symbols: set[str] | None = None) -> int:
    main_symbols = main_symbols or set()
    agent_positions = load_agent_positions_raw()
    symbols = {
        str(pos.get("symbol") or str(key).split("|", 1)[0])
        for key, pos in agent_positions.items()
    }
    return len(symbols - set(main_symbols))


def prune_files_to_unified_limit(limit: int) -> dict[str, Any]:
    main_positions = load_main_positions_raw()
    agent_positions = load_agent_positions_raw()
    ranked = ranked_unified_positions(main_positions, agent_positions, limit=limit)
    keep = {(str(row["source"]), str(row["key"])) for row in ranked}

    kept_main = {key: pos for key, pos in main_positions.items() if ("main", key) in keep}
    kept_agent = {key: pos for key, pos in agent_positions.items() if ("agent", key) in keep}

    removed_main = [key for key in main_positions if key not in kept_main]
    removed_agent = [key for key in agent_positions if key not in kept_agent]

    if removed_main:
        _write_json_dict(MAIN_POSITIONS_FILE, kept_main)
    if removed_agent:
        _write_json_dict(AGENT_POSITIONS_FILE, kept_agent)

    return {
        "limit": int(limit),
        "kept": ranked,
        "removed_main": removed_main,
        "removed_agent": removed_agent,
        "kept_main_count": len(kept_main),
        "kept_agent_count": len(kept_agent),
    }
