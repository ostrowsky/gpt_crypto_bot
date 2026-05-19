from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_v2_entry_admission_reward_replay import _load_admission_rows
from audit_v2_policy_gap import _summarize_trades
from audit_v2_reward_weighted_market_selector_offline_replay import (
    DEFAULT_DATASET,
    DEFAULT_HISTORY,
    DEFAULT_LABELS,
    DEFAULT_MARKET_SAMPLES,
    HORIZON,
    SELECTOR_DOWNSIDE,
    SELECTOR_FEATURES,
    SELECTOR_K,
    SELECTOR_THRESHOLD,
    _build_episodes_from_admission,
    _build_selector_choices,
    _latest_choice,
    _switched_policy,
)
from audit_v2_reward_weighted_market_selector import _select_samples
from audit_v2_temporal_exit_baselines import _build_temporal_rows
from audit_v2_temporal_exit_robustness import _base_policy, _grid_policy
from v2.policy_baselines import rollout, summarize_policy
from v2.state import Action


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT.parent / ".runtime" / "reports" / "v2_selector_failure_decomposition_15m.json"
BAR_MS = 15 * 60 * 1000


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
    rollouts_by_policy = {name: [rollout(env, policy) for env in episodes] for name, policy in policies.items()}
    summaries = {name: summarize_policy(name, runs) for name, runs in rollouts_by_policy.items()}
    decomposed = _decompose(episodes, rollouts_by_policy, selector_choices)
    payload = {
        "episodes": len(episodes),
        "selector": {
            "horizon": HORIZON,
            "features": list(SELECTOR_FEATURES),
            "k": SELECTOR_K,
            "downside_multiplier": SELECTOR_DOWNSIDE,
            "edge_threshold": SELECTOR_THRESHOLD,
        },
        "policy_summaries": summaries,
        "exit_summaries": {name: _summarize_trades(runs) for name, runs in rollouts_by_policy.items()},
        "decomposition": decomposed,
        "decision": _decision(decomposed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _decompose(episodes, rollouts_by_policy: dict, selector_choices: dict[str, list[dict]]) -> dict:
    switch_runs = rollouts_by_policy["reward_weighted_market_switch"]
    candidate_runs = rollouts_by_policy["fixed_candidate"]
    base_runs = rollouts_by_policy["fixed_base"]
    buckets_vs_candidate = defaultdict(float)
    buckets_vs_base = defaultdict(float)
    counts_vs_candidate = Counter()
    counts_vs_base = Counter()
    stale_buckets = defaultdict(float)
    state_buckets_vs_candidate = defaultdict(float)
    action_pair_vs_candidate = defaultdict(float)
    top_negative_vs_candidate = []
    top_negative_vs_base = []

    for env, switch_steps, candidate_steps, base_steps in zip(episodes, switch_runs, candidate_runs, base_runs):
        symbol = env.current_frame.bar.symbol
        day = env.current_frame.label.local_day
        choices = selector_choices.get(day, [])
        switch_pos_before = _position_before_flags(switch_steps)
        candidate_pos_before = _position_before_flags(candidate_steps)
        base_pos_before = _position_before_flags(base_steps)
        for idx, (sw, cand, base) in enumerate(zip(switch_steps, candidate_steps, base_steps)):
            ts = sw.frame.bar.open_ts_ms
            choice = _latest_choice(choices, ts)
            choice_policy = "none" if choice is None else str(choice["policy"])
            age_bars = None if choice is None else max(0, (ts - int(choice["anchor_ts_ms"])) // BAR_MS)
            stale_bucket = _stale_bucket(age_bars)
            phase = _phase(switch_pos_before[idx], candidate_pos_before[idx], base_pos_before[idx])
            state = sw.frame.label.state.value
            delta_vs_candidate = sw.reward.total - cand.reward.total
            delta_vs_base = sw.reward.total - base.reward.total
            bucket_candidate = _bucket_vs_reference(
                choice_policy=choice_policy,
                phase=phase,
                switch_action=sw.action,
                reference_action=cand.action,
                reference_name="candidate",
                age_bars=age_bars,
            )
            bucket_base = _bucket_vs_reference(
                choice_policy=choice_policy,
                phase=phase,
                switch_action=sw.action,
                reference_action=base.action,
                reference_name="base",
                age_bars=age_bars,
            )
            buckets_vs_candidate[bucket_candidate] += delta_vs_candidate
            counts_vs_candidate[bucket_candidate] += 1
            buckets_vs_base[bucket_base] += delta_vs_base
            counts_vs_base[bucket_base] += 1
            stale_buckets[stale_bucket] += delta_vs_candidate
            state_buckets_vs_candidate[state] += delta_vs_candidate
            action_pair_vs_candidate[f"switch:{sw.action.value}|candidate:{cand.action.value}"] += delta_vs_candidate
            if delta_vs_candidate < 0:
                _push_example(
                    top_negative_vs_candidate,
                    {
                        "symbol": symbol,
                        "day": day,
                        "ts_ms": ts,
                        "state": state,
                        "choice_policy": choice_policy,
                        "choice_age_bars": age_bars,
                        "phase": phase,
                        "switch_action": sw.action.value,
                        "candidate_action": cand.action.value,
                        "delta": round(delta_vs_candidate, 6),
                        "switch_reward": round(sw.reward.total, 6),
                        "candidate_reward": round(cand.reward.total, 6),
                        "bucket": bucket_candidate,
                    },
                )
            if delta_vs_base < 0:
                _push_example(
                    top_negative_vs_base,
                    {
                        "symbol": symbol,
                        "day": day,
                        "ts_ms": ts,
                        "state": state,
                        "choice_policy": choice_policy,
                        "choice_age_bars": age_bars,
                        "phase": phase,
                        "switch_action": sw.action.value,
                        "base_action": base.action.value,
                        "delta": round(delta_vs_base, 6),
                        "switch_reward": round(sw.reward.total, 6),
                        "base_reward": round(base.reward.total, 6),
                        "bucket": bucket_base,
                    },
                )

    return {
        "vs_fixed_candidate": _pack_buckets(buckets_vs_candidate, counts_vs_candidate),
        "vs_fixed_base": _pack_buckets(buckets_vs_base, counts_vs_base),
        "stale_age_vs_candidate": _pack_simple(stale_buckets),
        "state_vs_candidate": _pack_simple(state_buckets_vs_candidate),
        "action_pair_vs_candidate": _pack_simple(action_pair_vs_candidate),
        "top_negative_vs_candidate": top_negative_vs_candidate,
        "top_negative_vs_base": top_negative_vs_base,
    }


def _position_before_flags(steps) -> list[bool]:
    flags = []
    has_position = False
    for step in steps:
        flags.append(has_position)
        if step.action in {Action.OPEN_SMALL, Action.OPEN_FULL}:
            has_position = True
        elif step.action == Action.SELL:
            has_position = False
    return flags


def _phase(switch_pos: bool, candidate_pos: bool, base_pos: bool) -> str:
    if switch_pos and candidate_pos and base_pos:
        return "all_in_position"
    if not switch_pos and not candidate_pos and not base_pos:
        return "all_flat"
    if switch_pos:
        return "switch_in_position_only_or_partial"
    if candidate_pos:
        return "candidate_in_position_only_or_partial"
    if base_pos:
        return "base_in_position_only_or_partial"
    return "mixed_position"


def _bucket_vs_reference(*, choice_policy: str, phase: str, switch_action: Action, reference_action: Action, reference_name: str, age_bars: int | None) -> str:
    if choice_policy == "none":
        return "no_selector_choice"
    stale = age_bars is not None and age_bars >= 8
    if switch_action == reference_action:
        return f"same_action_{choice_policy}" + ("_stale" if stale else "")
    if choice_policy == "base" and reference_name == "candidate":
        if reference_action in {Action.OPEN_SMALL, Action.OPEN_FULL}:
            return "candidate_suppressed_open" + ("_stale" if stale else "")
        if reference_action == Action.HOLD:
            return "candidate_suppressed_hold" + ("_stale" if stale else "")
        if reference_action == Action.SELL:
            return "candidate_suppressed_sell" + ("_stale" if stale else "")
        return "candidate_suppressed_other" + ("_stale" if stale else "")
    if choice_policy == "candidate" and reference_name == "base":
        if switch_action in {Action.OPEN_SMALL, Action.OPEN_FULL}:
            return "candidate_enabled_open" + ("_stale" if stale else "")
        if switch_action == Action.HOLD:
            return "candidate_enabled_hold" + ("_stale" if stale else "")
        if switch_action == Action.SELL:
            return "candidate_enabled_sell" + ("_stale" if stale else "")
        return "candidate_enabled_other" + ("_stale" if stale else "")
    return f"action_mismatch_{phase}_{choice_policy}" + ("_stale" if stale else "")


def _stale_bucket(age_bars: int | None) -> str:
    if age_bars is None:
        return "no_choice"
    if age_bars <= 1:
        return "age_0_1"
    if age_bars <= 4:
        return "age_2_4"
    if age_bars <= 8:
        return "age_5_8"
    return "age_9_plus"


def _push_example(items: list[dict], example: dict, limit: int = 20) -> None:
    items.append(example)
    items.sort(key=lambda item: item["delta"])
    del items[limit:]


def _pack_buckets(values: dict, counts: Counter) -> list[dict]:
    rows = [
        {"bucket": key, "delta": round(value, 6), "count": counts[key]}
        for key, value in values.items()
    ]
    rows.sort(key=lambda item: item["delta"])
    return rows


def _pack_simple(values: dict) -> list[dict]:
    rows = [{"bucket": key, "delta": round(value, 6)} for key, value in values.items()]
    rows.sort(key=lambda item: item["delta"])
    return rows


def _decision(decomposition: dict) -> dict:
    worst = decomposition["vs_fixed_candidate"][0] if decomposition.get("vs_fixed_candidate") else {}
    stale = [row for row in decomposition.get("stale_age_vs_candidate", []) if row["bucket"] == "age_9_plus"]
    stale_delta = stale[0]["delta"] if stale else 0.0
    return {
        "dominant_failure_bucket_vs_candidate": worst,
        "stale_age_9_plus_delta_vs_candidate": stale_delta,
        "recommendation": _recommendation(str(worst.get("bucket", "")), stale_delta),
    }


def _recommendation(bucket: str, stale_delta: float) -> str:
    if stale_delta < -50:
        return "test selector TTL / stale-choice expiry before new policy learning"
    if "suppressed" in bucket:
        return "split entry and exit selectors; current switch suppresses useful candidate behavior"
    if "enabled" in bucket:
        return "add position-aware downside guard before candidate enablement"
    return "inspect top negative examples and build position-aware selector"


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
