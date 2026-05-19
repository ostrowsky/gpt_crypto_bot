from __future__ import annotations

import argparse
import json
from pathlib import Path

from audit_v2_entry_admission_reward_replay import _load_admission_rows
from audit_v2_policy_gap import _summarize_trades
from audit_v2_reward_weighted_market_selector import _reward_weighted_knn_expected_delta, _select_samples
from audit_v2_temporal_exit_baselines import _build_temporal_rows
from audit_v2_temporal_exit_robustness import _base_policy, _grid_policy
from run_v2_belief_filter import _scaled
from run_v2_state_reconstruction import _build_confidence, _load_labels
from v2.belief_filter import filter_rows
from v2.history_store import LocalHistoryStore
from v2.offline_env import DecisionFrame, OfflineDecisionEnvironment
from v2.policy_baselines import rollout, summarize_policy
from v2.state_reconstruction import build_rows, chronological_split, fit_centroids, fit_scaler
from v2.belief import BeliefState
from v2.state import SymbolState
from collections import defaultdict


ROOT = Path(__file__).resolve().parent
DEFAULT_HISTORY = ROOT / ".runtime" / "v2_history"
DEFAULT_LABELS = ROOT.parent / ".runtime" / "reports" / "v2_lifecycle_labels_15m.jsonl"
DEFAULT_DATASET = ROOT.parent / ".runtime" / "reports" / "v2_entry_admission_dataset_15m.jsonl"
DEFAULT_MARKET_SAMPLES = ROOT.parent / ".runtime" / "reports" / "v2_market_breadth_observation_store_15m.json"
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_reward_weighted_market_selector_offline_replay_15m.json"
HORIZON = "2h"
SELECTOR_FEATURES = ("market_btc_ret4_pct", "market_volume_gt_mean20_share")
SELECTOR_K = 5
SELECTOR_DOWNSIDE = 3.0
SELECTOR_THRESHOLD = 0.0


def build(history_root: Path, labels_path: Path, dataset_path: Path, market_samples_path: Path, output: Path) -> dict:
    rows = _load_admission_rows(dataset_path)
    episodes = _build_episodes_from_admission(history_root, labels_path, rows)
    temporal_rows = _build_temporal_rows(rows)
    market_payload = json.loads(market_samples_path.read_text(encoding="utf-8"))
    samples = _select_samples(market_payload["samples"], HORIZON, list(SELECTOR_FEATURES))
    selector_choices = _build_selector_choices(samples)

    policies = {
        "fixed_base": _base_policy(rows),
        "fixed_candidate": _grid_policy(rows, temporal_rows, -0.20, 0.10),
        "reward_weighted_market_switch": _switched_policy(rows, temporal_rows, selector_choices),
    }
    rollouts = {name: [rollout(env, policy) for env in episodes] for name, policy in policies.items()}
    summaries = {name: summarize_policy(name, runs) for name, runs in rollouts.items()}
    exits = {name: _summarize_trades(runs) for name, runs in rollouts.items()}
    base_reward = summaries["fixed_base"]["total_reward"]
    candidate_reward = summaries["fixed_candidate"]["total_reward"]
    payload = {
        "episodes": len(episodes),
        "selector": {
            "horizon": HORIZON,
            "features": list(SELECTOR_FEATURES),
            "k": SELECTOR_K,
            "downside_multiplier": SELECTOR_DOWNSIDE,
            "edge_threshold": SELECTOR_THRESHOLD,
            "choices": _summarize_choices(selector_choices),
        },
        "policies": {
            name: {
                "summary": summaries[name],
                "exit": exits[name],
                "delta_vs_fixed_base": round(summaries[name]["total_reward"] - base_reward, 6),
                "delta_vs_fixed_candidate": round(summaries[name]["total_reward"] - candidate_reward, 6),
            }
            for name in policies
        },
        "decision": _decision(summaries),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _build_episodes_from_admission(
    history_root: Path,
    labels_path: Path,
    admission_rows: dict[tuple[str, int], dict],
) -> list[OfflineDecisionEnvironment]:
    labels, _ = _load_labels(labels_path)
    symbols = sorted({symbol for symbol, _ in admission_rows})
    store = LocalHistoryStore(history_root)
    bars_by_key = {}
    for symbol in symbols:
        try:
            slice_ = store.load(symbol, "15m")
        except Exception:
            continue
        for bar in slice_.bars:
            bars_by_key[(symbol, bar.open_ts_ms)] = bar

    grouped = defaultdict(list)
    for (symbol, ts_ms), row in sorted(admission_rows.items(), key=lambda item: (item[1].get("local_day", ""), item[0][0], item[0][1])):
        bar = bars_by_key.get((symbol, ts_ms))
        label = labels.get((symbol, ts_ms))
        if bar is None or label is None:
            continue
        probs = {SymbolState(name): float(value) for name, value in (row.get("belief") or {}).items()}
        prediction = SymbolState(row.get("predicted_state") or max(probs, key=probs.get).value)
        frame = DecisionFrame(
            bar=bar,
            label=label,
            belief=BeliefState(probs),
            prediction=prediction,
        )
        grouped[(symbol, row.get("local_day") or label.local_day)].append(frame)
    return [OfflineDecisionEnvironment(frames) for frames in grouped.values() if frames]


def _build_episodes_fast(history_root: Path, labels_path: Path) -> list[OfflineDecisionEnvironment]:
    labels, day_sizes = _load_labels(labels_path)
    confidence = _build_confidence(labels, day_sizes)
    labels_by_symbol = defaultdict(dict)
    confidence_by_symbol = defaultdict(dict)
    for (symbol, ts_ms), label in labels.items():
        labels_by_symbol[symbol][ts_ms] = label
    for (symbol, ts_ms), conf in confidence.items():
        confidence_by_symbol[symbol][ts_ms] = conf

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
        symbol_labels = labels_by_symbol.get(symbol, {})
        symbol_conf = confidence_by_symbol.get(symbol, {})
        rows.extend(build_rows(slice_.bars, symbol_labels, symbol_conf))
        for bar in slice_.bars:
            bars_by_key[(symbol, bar.open_ts_ms)] = bar
        for ts_ms, label in symbol_labels.items():
            labels_by_key[(symbol, ts_ms)] = label

    train, test = chronological_split(rows)
    means, stds = fit_scaler(train)
    scaled_train = [_scaled(row, means, stds) for row in train]
    scaled_test = [_scaled(row, means, stds) for row in test]
    centroids = fit_centroids(scaled_train)
    filtered = filter_rows(scaled_test, centroids, self_bias=0.85, temperature=0.75)

    grouped = defaultdict(list)
    for item in filtered:
        key = (item.row.symbol, item.row.local_day)
        bar = bars_by_key.get((item.row.symbol, item.row.ts_ms))
        label = labels_by_key.get((item.row.symbol, item.row.ts_ms))
        if bar is None or label is None:
            continue
        grouped[key].append(
            DecisionFrame(
                bar=bar,
                label=label,
                belief=item.belief,
                prediction=item.prediction,
            )
        )
    return [OfflineDecisionEnvironment(frames) for frames in grouped.values() if frames]


def _build_selector_choices(samples: list[dict]) -> dict[str, list[dict]]:
    history = []
    choices: dict[str, list[dict]] = {}
    for sample in samples:
        labels = {item["label"] for item in history}
        if labels == {"candidate_favorable", "candidate_unfavorable"}:
            expected, neighbors = _reward_weighted_knn_expected_delta(
                history,
                sample["features"],
                list(SELECTOR_FEATURES),
                SELECTOR_K,
                SELECTOR_DOWNSIDE,
            )
            choose_candidate = expected > SELECTOR_THRESHOLD
            choices.setdefault(sample["day"], []).append(
                {
                    "anchor_ts_ms": int(sample["anchor_ts_ms"]),
                    "policy": "candidate" if choose_candidate else "base",
                    "expected_delta": expected,
                    "actual_label": sample["label"],
                    "actual_reward_delta": sample["reward_delta"],
                    "neighbor_count": len(neighbors),
                }
            )
        history.append(sample)
    for day in choices:
        choices[day].sort(key=lambda item: item["anchor_ts_ms"])
    return choices


def _switched_policy(admission_rows, temporal_rows, selector_choices: dict[str, list[dict]]):
    base = _base_policy(admission_rows)
    candidate = _grid_policy(admission_rows, temporal_rows, -0.20, 0.10)

    def policy(env):
        day = env.current_frame.label.local_day
        ts = env.current_frame.bar.open_ts_ms
        choice = _latest_choice(selector_choices.get(day, []), ts)
        if choice is None or choice["policy"] != "candidate":
            return base(env)
        return candidate(env)

    return policy


def _latest_choice(choices: list[dict], ts_ms: int) -> dict | None:
    latest = None
    for choice in choices:
        if int(choice["anchor_ts_ms"]) <= ts_ms:
            latest = choice
        else:
            break
    return latest


def _summarize_choices(selector_choices: dict[str, list[dict]]) -> dict:
    flat = [choice for choices in selector_choices.values() for choice in choices]
    candidate = [choice for choice in flat if choice["policy"] == "candidate"]
    wrong_candidate = [choice for choice in candidate if choice["actual_reward_delta"] <= 0]
    return {
        "days": len(selector_choices),
        "anchors": len(flat),
        "candidate_anchors": len(candidate),
        "candidate_anchor_share": round(len(candidate) / len(flat), 6) if flat else 0.0,
        "wrong_candidate_anchors": len(wrong_candidate),
        "wrong_candidate_loss": round(sum(choice["actual_reward_delta"] for choice in wrong_candidate), 6),
    }


def _decision(summaries: dict) -> dict:
    base = summaries["fixed_base"]
    candidate = summaries["fixed_candidate"]
    switched = summaries["reward_weighted_market_switch"]
    beats_base = switched["total_reward"] > base["total_reward"]
    beats_candidate = switched["total_reward"] > candidate["total_reward"]
    trade_ratio = switched["trade_count"] / max(candidate["trade_count"], 1)
    return {
        "beats_fixed_base": beats_base,
        "beats_fixed_candidate": beats_candidate,
        "trade_count_ratio_vs_candidate": round(trade_ratio, 6),
        "promotion_gate_passed": bool(beats_base and beats_candidate and trade_ratio <= 1.25),
        "recommendation": "advance to live shadow telemetry only" if beats_base and beats_candidate and trade_ratio <= 1.25 else "keep research-only",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--market-samples", type=Path, default=DEFAULT_MARKET_SAMPLES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    payload = build(args.history_root, args.labels, args.dataset, args.market_samples, args.output)
    print(json.dumps(payload, ensure_ascii=False, indent=2) if args.as_json else payload["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
