"""Offline after-cost fixed-T5 diagnostic, never a production release gate."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import causal_entry_reconstruction as repair
import independent_signal_evaluator as evaluator
from historical_signal_evaluation import freeze, sha
from ml_candidate_ranker import predict_final_score_from_candidate_payload

CONTRACT = 'independent-ten-slot-fixed-t5-diagnostic-v1'
BLOCKERS = ['production_admission_and_sell_not_replayed',
            'point_in_time_universe_coverage_unknown',
            'continuous_mark_to_market_and_btc_benchmark_missing',
            'sealed_historical_protocol_missing',
            'forward_and_canary_adoption_not_delivered',
            'newer_live_history_outside_frozen_snapshot']


def number(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('nonfinite numeric input')
    return value


def simulate(events, score_key, fee_bps=10, slippage_bps=5):
    """Rank only simultaneous executable arrivals; no outcome-based selection."""
    fee, slip = number(fee_bps)/10000, number(slippage_bps)/10000
    if not 0 <= fee < 1 or not 0 <= slip < 1:
        raise ValueError('invalid costs')
    arrivals, exits, ids = defaultdict(list), defaultdict(list), set()
    for event in events:
        e = dict(event)
        if not e.get('id') or not e.get('sym') or e['id'] in ids:
            raise ValueError('unknown or duplicate identity')
        ids.add(e['id'])
        for key in ('entry_ms', 'exit_ms', 'entry_price', 'exit_price', score_key):
            e[key] = number(e[key])
        if (e['entry_ms'] != int(e['entry_ms']) or e['exit_ms'] != int(e['exit_ms'])
                or e['entry_ms'] >= e['exit_ms']
                or min(e['entry_price'], e['exit_price']) <= 0):
            raise ValueError('invalid execution timing or price')
        arrivals[e['entry_ms']].append(e)
        exits[e['exit_ms']].append(e)
    cash, open_positions, fills, skips = 1.0, {}, [], []
    costs, peak, drawdown, competitions, batches = 0.0, 1.0, 0.0, 0, 0
    for ts in sorted(set(arrivals) | set(exits)):
        for e in sorted(exits[ts], key=lambda e: e['id']):
            position = open_positions.get(e['sym'])
            if not position or position['id'] != e['id']:
                continue
            gross = position['qty'] * e['exit_price'] * (1-slip)
            exit_fee = gross*fee
            cash += gross-exit_fee
            costs += exit_fee
            fills.append({**position, 'exit_ms': ts, 'exit_price': e['exit_price']*(1-slip),
                          'exit_fee': exit_fee, 'net_proceeds': gross-exit_fee,
                          'net_pnl': gross-exit_fee-position['allocation']})
            del open_positions[e['sym']]
        available = max(0, min(10-len(open_positions), int((cash+1e-12)/0.1)))
        batch = arrivals[ts]
        if batch:
            batches += 1
            competitions += len({e['sym'] for e in batch}) > available
        for e in sorted(batch, key=lambda e: (-e[score_key], e['id'])):
            reason = ('symbol_already_open' if e['sym'] in open_positions else
                      'capacity' if len(open_positions) >= 10 else
                      'cash' if cash+1e-12 < 0.1 else None)
            if reason:
                skips.append({'id': e['id'], 'at_ms': ts, 'reason': reason})
                continue
            # Allocation includes entry fee; never borrow or compound position size.
            gross = 0.1/(1+fee)
            entry_fee = gross*fee
            price = e['entry_price']*(1+slip)
            open_positions[e['sym']] = {'id': e['id'], 'sym': e['sym'],
                                        'entry_ms': ts, 'entry_price': price,
                                        'qty': gross/price, 'entry_fee': entry_fee,
                                        'allocation': 0.1}
            cash -= 0.1
            costs += entry_fee
        settled_equity = cash+0.1*len(open_positions)
        peak = max(peak, settled_equity)
        drawdown = max(drawdown, 1-settled_equity/peak)
        if cash < -1e-10 or len(open_positions) > 10:
            raise ValueError('portfolio invariant failed')
    if open_positions or abs(cash-(1+sum(f['net_pnl'] for f in fills))) > 1e-9:
        raise ValueError('unliquidated or unreconciled portfolio')
    return {'final_equity': cash, 'net_return_pct': (cash-1)*100,
            'settled_equity_drawdown_pct': drawdown*100,
            'mark_to_market_drawdown_pct': None, 'fees_initial_capital': costs,
            'trades': len(fills), 'positive_trades': {
                'numerator': sum(f['net_pnl'] > 0 for f in fills), 'denominator': len(fills)},
            'capacity_competition_batches': {'numerator': competitions, 'denominator': batches},
            'fills': fills, 'skips': skips}


def build(run, fee_bps=10, slippage_bps=5, progress=None):
    # Verify raw exchange response hashes and source identity before using evidence.
    source_hash = sha(Path(__file__))
    input_hashes = {name: sha(run/name) for name in (
        'receipt.json', 'candidate.json', 'manifest.json', 'repaired.jsonl', 'repair_manifest.json')}
    source = repair.verify(run)
    if progress:
        progress('exchange_receipts_verified')
    manifest = json.loads((run/'repair_manifest.json').read_bytes())
    for name, expected in manifest['sources'].items():
        if Path(name).name != name or sha(Path(__file__).with_name(name)) != expected:
            raise ValueError('reconstruction source changed: '+name)
    now = evaluator.provenance.parse_utc(source['generated_at'])
    scores = {}
    def score(payload, causal):
        identity = causal['id']
        if identity not in scores:
            scores[identity] = predict_final_score_from_candidate_payload(payload, causal)
        return scores[identity]
    checked = evaluator.evaluate(run, run/'repaired.jsonl', predictor=score, now=now,
                                 historical=True, execution=True)
    if progress:
        progress('post_exposure_cohort_recomputed')
    # Recompute whole-group exclusion; never trust selected IDs from a saved report.
    groups = {p['group'] for p in checked['pairs']}
    payload = json.loads((run/'candidate.json').read_bytes())
    events = []
    for line in (run/'repaired.jsonl').read_bytes().splitlines():
        row = json.loads(line)
        group = str((row.get('provenance') or {}).get('feature_time'))+'|'+str(row.get('tf'))
        if group not in groups:
            continue
        repair.execution_return(row, now)
        proof = row['execution_reconstruction']
        minute = proof['minute_bars'][1]
        target = proof['target_bars'][5]
        causal = {k: v for k, v in row.items() if k not in {
            'labels', 'teacher', 'label_provenance', 'execution_reconstruction',
            'reconstruction_integrity'}}
        events.append({'id': row['id'], 'sym': row['sym'],
                       'entry_ms': int(minute[0]), 'entry_price': float(minute[1]),
                       'exit_ms': int(target[6])+1, 'exit_price': float(target[4]),
                       'baseline_score': row['decision']['candidate_score'],
                       'candidate_score': score(payload, causal)})
    baseline = simulate(events, 'baseline_score', fee_bps, slippage_bps)
    candidate = simulate(events, 'candidate_score', fee_bps, slippage_bps)
    if source_hash != sha(Path(__file__)) or any(
            expected != sha(run/name) for name, expected in input_hashes.items()):
        raise ValueError('replay inputs/source changed during evaluation')
    return {'contract': CONTRACT, 'generated_at': datetime.now(timezone.utc).isoformat(),
            'state': 'BLOCKED', 'runtime_eligible': False, 'achievement_claimed': False,
            'evidence_status': 'retrospective_fixed_horizon_portfolio_diagnostic',
            'baseline_name': 'candidate_score_arrival_order_not_production_champion',
            'candidate_sha256': sha(run/'candidate.json'),
            'evaluator_source_sha256': source_hash, 'input_hashes': input_hashes,
            'reconstruction_receipt_sha256': sha(run/'receipt.json'),
            'input_scope': 'entire_verified_frozen_file_without_date_or_symbol_filter',
            'available_history': checked['available_history'],
            'post_exposure_start': checked['evaluation_start'],
            'eligible_events': len(events), 'group_count': len(groups),
            'excluded_data_quality': checked['learning_quality'],
            'execution_window_ms': [min(e['entry_ms'] for e in events),
                                    max(e['exit_ms'] for e in events)] if events else None,
            'costs': {'fee_bps_per_side': fee_bps, 'slippage_bps_per_side': slippage_bps},
            'baseline': baseline, 'candidate': candidate,
            'net_return_delta_pp': (candidate['net_return_pct']-baseline['net_return_pct'])
                                  if events else None,
            'release_blockers': BLOCKERS, 'simulated_events': events}


def publish(output, report):
    if report.get('evaluator_source_sha256') != sha(Path(__file__)):
        raise ValueError('report evaluator changed before publication')
    output.mkdir(parents=True, exist_ok=False)
    freeze(output/'result.json', json.dumps(report, sort_keys=True, allow_nan=False).encode())
    receipt = {'result.json': sha(output/'result.json'),
               'evaluator_source_sha256': sha(Path(__file__))}
    freeze(output/'receipt.json', json.dumps(receipt, sort_keys=True).encode())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reconstruction', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fee-bps', type=float, default=10)
    parser.add_argument('--slippage-bps', type=float, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('immutable output already exists')
    result = build(args.reconstruction, args.fee_bps, args.slippage_bps,
                   progress=lambda state: print(state, file=sys.stderr, flush=True))
    publish(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k in {
        'state', 'eligible_events', 'group_count', 'net_return_delta_pp', 'release_blockers'}}))
