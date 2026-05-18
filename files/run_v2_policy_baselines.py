from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from run_v2_belief_filter import _scaled
from run_v2_state_reconstruction import _build_confidence, _load_labels
from v2.belief_filter import filter_rows
from v2.history_store import LocalHistoryStore
from v2.offline_env import DecisionFrame, OfflineDecisionEnvironment
from v2.policy_baselines import (
    always_flat_policy,
    belief_policy_v1,
    lifecycle_oracle_policy,
    rollout,
    summarize_policy,
)
from v2.state_reconstruction import build_rows, chronological_split, fit_centroids, fit_scaler


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_policy_baselines_15m.json"


def build(history_root: Path, labels_path: Path, output: Path) -> dict:
    episodes = _build_episodes(history_root, labels_path)
    policies = {
        "always_flat": always_flat_policy,
        "lifecycle_oracle": lifecycle_oracle_policy,
        "belief_policy_v1": belief_policy_v1,
    }
    payload = {
        "episodes": len(episodes),
        "parameters": {"self_bias": 0.85, "temperature": 0.75},
        "policies": {
            name: summarize_policy(name, [rollout(env, policy) for env in episodes])
            for name, policy in policies.items()
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _build_episodes(history_root: Path, labels_path: Path) -> list[OfflineDecisionEnvironment]:
    labels, day_sizes = _load_labels(labels_path)
    confidence = _build_confidence(labels, day_sizes)
    store = LocalHistoryStore(history_root)
    rows = []
    bars_by_key = {}
    labels_by_key = {}
    for symbol, tf in store.keys():
        if tf != "15m":
            continue
        slice_ = store.load(symbol, tf)
        if not slice_.is_contiguous:
            continue
        symbol_labels = {ts: label for (sym, ts), label in labels.items() if sym == symbol}
        symbol_conf = {ts: conf for (sym, ts), conf in confidence.items() if sym == symbol}
        rows.extend(build_rows(slice_.bars, symbol_labels, symbol_conf))
        for bar in slice_.bars:
            bars_by_key[(symbol, bar.open_ts_ms)] = bar
        for ts, label in symbol_labels.items():
            labels_by_key[(symbol, ts)] = label
    train, test = chronological_split(rows)
    means, stds = fit_scaler(train)
    scaled_train = [_scaled(row, means, stds) for row in train]
    scaled_test = [_scaled(row, means, stds) for row in test]
    centroids = fit_centroids(scaled_train)
    filtered = filter_rows(scaled_test, centroids, self_bias=0.85, temperature=0.75)

    grouped = defaultdict(list)
    for item in filtered:
        key = (item.row.symbol, item.row.local_day)
        bar = bars_by_key[(item.row.symbol, item.row.ts_ms)]
        label = labels_by_key[(item.row.symbol, item.row.ts_ms)]
        grouped[key].append(
            DecisionFrame(
                bar=bar,
                label=label,
                belief=item.belief,
                prediction=item.prediction,
            )
        )
    return [OfflineDecisionEnvironment(frames) for frames in grouped.values() if frames]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.history_root, args.labels, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["policies"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
