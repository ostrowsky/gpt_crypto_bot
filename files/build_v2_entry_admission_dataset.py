from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from run_v2_belief_filter import _scaled
from run_v2_state_reconstruction import _build_confidence, _load_labels
from v2.belief_filter import filter_rows
from v2.entry_admission_dataset import V1StructuralFeatures, V1TemporalFeatures, build_row
from v2.history_store import LocalHistoryStore
from v2.state_reconstruction import build_rows, chronological_split, fit_centroids, fit_scaler


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_CRITIC = ROOT / "critic_dataset.jsonl"
DEFAULT_EVENTS = ROOT / "bot_events.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_dataset_audit_15m.json"
DEFAULT_DATASET_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_dataset_15m.jsonl"


def build(
    history_root: Path,
    labels_path: Path,
    critic_path: Path,
    events_path: Path,
    output: Path,
    dataset_output: Path,
) -> dict:
    labels, day_sizes = _load_labels(labels_path)
    confidence = _build_confidence(labels, day_sizes)
    store = LocalHistoryStore(history_root)
    rows = []
    for symbol, tf in store.keys():
        if tf != "15m":
            continue
        slice_ = store.load(symbol, tf)
        if not slice_.is_contiguous:
            continue
        labels_by_ts = {ts: label for (sym, ts), label in labels.items() if sym == symbol}
        conf_by_ts = {ts: conf for (sym, ts), conf in confidence.items() if sym == symbol}
        rows.extend(build_rows(slice_.bars, labels_by_ts, conf_by_ts))
    train, test = chronological_split(rows)
    means, stds = fit_scaler(train)
    scaled_train = [_scaled(row, means, stds) for row in train]
    scaled_test = [_scaled(row, means, stds) for row in test]
    centroids = fit_centroids(scaled_train)
    filtered = filter_rows(scaled_test, centroids, self_bias=0.85, temperature=0.75)

    structural = _load_structural(critic_path)
    events = _load_scout_events(events_path)
    admission_rows = []
    for item in filtered:
        key = (item.row.symbol, "15m", item.row.ts_ms)
        admission_rows.append(
            build_row(
                item,
                structural=structural.get(key),
                temporal=_temporal_features(events.get(item.row.symbol, []), item.row.ts_ms),
            )
        )
    dataset_output.parent.mkdir(parents=True, exist_ok=True)
    dataset_output.write_text(
        "\n".join(json.dumps(asdict(row), ensure_ascii=False, default=str) for row in admission_rows) + "\n",
        encoding="utf-8",
    )
    payload = _summarize(admission_rows)
    payload["dataset_output"] = str(dataset_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _load_structural(path: Path) -> dict[tuple[str, str, int], V1StructuralFeatures]:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("tf") != "15m":
            continue
        decision = row.get("decision") or {}
        try:
            key = (str(row["sym"]), str(row["tf"]), int(row["bar_ts"]))
        except Exception:
            continue
        out[key] = V1StructuralFeatures(
            candidate_score=_num(decision.get("candidate_score")),
            base_score=_num(decision.get("base_score")),
            score_floor=_num(decision.get("score_floor")),
            forecast_return_pct=_num(decision.get("forecast_return_pct")),
            today_change_pct=_num(decision.get("today_change_pct")),
            ml_proba=_num(decision.get("ml_proba")),
            mtf_soft_penalty=_num(decision.get("mtf_soft_penalty")),
            fresh_priority=bool(decision.get("fresh_priority")),
            catchup=bool(decision.get("catchup")),
            continuation_profile=bool(decision.get("continuation_profile")),
            near_miss=bool(decision.get("near_miss")),
            signal_flags={str(k): bool(v) for k, v in (decision.get("signal_flags") or {}).items()},
        )
    return out


def _load_scout_events(path: Path) -> dict[str, list[dict]]:
    out = defaultdict(list)
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            row = json.loads(line)
        except Exception:
            continue
        if row.get("event") != "scout_shadow":
            continue
        ts = _parse_ts(row.get("ts"))
        if ts is None:
            continue
        out[str(row.get("sym") or "")].append(
            {
                "ts_ms": int(ts.timestamp() * 1000),
                "is_wakeup": str(row.get("scout_profile") or "").startswith("wake_up"),
            }
        )
    for symbol in out:
        out[symbol].sort(key=lambda item: item["ts_ms"])
    return out


def _temporal_features(events: list[dict], ts_ms: int) -> V1TemporalFeatures:
    prior = [event for event in events if event["ts_ms"] < ts_ms]
    structural = [event for event in prior if not event["is_wakeup"]]
    wakeups = [event for event in prior if event["is_wakeup"]]
    return V1TemporalFeatures(
        prior_structural_scout=bool(structural),
        prior_wakeup_scout=bool(wakeups),
        minutes_since_first_structural_scout=_minutes(ts_ms, structural[0]["ts_ms"]) if structural else None,
        minutes_since_latest_wakeup_scout=_minutes(ts_ms, wakeups[-1]["ts_ms"]) if wakeups else None,
    )


def _summarize(rows) -> dict:
    states = Counter(row.true_state.value for row in rows)
    structural_rows = [row for row in rows if row.v1_structural is not None]
    temporal_structural = [row for row in rows if row.v1_temporal.prior_structural_scout]
    temporal_wakeup = [row for row in rows if row.v1_temporal.prior_wakeup_scout]
    return {
        "rows": len(rows),
        "state_counts": dict(states),
        "coverage": {
            "v1_structural_rows": len(structural_rows),
            "v1_structural_pct": round(len(structural_rows) / len(rows), 6) if rows else 0.0,
            "prior_structural_scout_rows": len(temporal_structural),
            "prior_structural_scout_pct": round(len(temporal_structural) / len(rows), 6) if rows else 0.0,
            "prior_wakeup_scout_rows": len(temporal_wakeup),
            "prior_wakeup_scout_pct": round(len(temporal_wakeup) / len(rows), 6) if rows else 0.0,
        },
        "state_counts_with_v1_structural": dict(Counter(row.true_state.value for row in structural_rows)),
        "sample_feature_keys": sorted(asdict(structural_rows[0].v1_structural).keys()) if structural_rows else [],
    }


def _num(value):
    try:
        return None if value is None else float(value)
    except Exception:
        return None


def _parse_ts(raw):
    if not raw:
        return None
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone(timezone.utc)


def _minutes(later_ms: int, earlier_ms: int) -> float:
    return round((later_ms - earlier_ms) / 60000.0, 6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--critic", type=Path, default=DEFAULT_CRITIC)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-output", type=Path, default=DEFAULT_DATASET_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.history_root, args.labels, args.critic, args.events, args.output, args.dataset_output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
